"""
Pydantic models for VLM request/response handling.

These models define the data structures for Vision-Language Model
interactions, including element detection, classification, and metadata.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class ElementType(str, Enum):
    """
    Enhanced EHR semantic element types (15+ types vs original 4).

    Based on VLM_REFACTOR.md Section 3.3.1: Expanded Element Types
    """

    # Document Structure
    DOCUMENT_HEADER = "document_header"
    PATIENT_DEMOGRAPHICS = "patient_demographics"
    PAGE_METADATA = "page_metadata"
    SECTION_HEADER = "section_header"
    SUBSECTION_HEADER = "subsection_header"

    # Clinical Content
    CLINICAL_PARAGRAPH = "clinical_paragraph"
    MEDICATION_TABLE = "medication_table"
    LAB_RESULTS_TABLE = "lab_results_table"
    VITAL_SIGNS_TABLE = "vital_signs_table"
    RECOMMENDATIONS_TABLE = "recommendations_table"
    GENERAL_TABLE = "general_table"
    PROBLEM_LIST = "problem_list"
    ASSESSMENT_PLAN = "assessment_plan"
    LIST_ITEMS = "list_items"

    # Special Content
    HANDWRITTEN_ANNOTATION = "handwritten_annotation"
    STAMP_SIGNATURE = "stamp_signature"
    MEDICAL_FIGURE = "medical_figure"
    FORM_FIELD_GROUP = "form_field_group"

    # Administrative
    MARGIN_CONTENT = "margin_content"
    UNCATEGORIZED = "uncategorized"

    # Document Summary (excluded from querying)
    DOCUMENT_SUMMARY = "document_summary"


class BoundingBox(BaseModel):
    """Bounding box coordinates in pixel space."""

    x0: float = Field(..., description="Left x-coordinate")
    y0: float = Field(..., description="Top y-coordinate")
    x1: float = Field(..., description="Right x-coordinate")
    y1: float = Field(..., description="Bottom y-coordinate")

    @field_validator('x0', 'y0', 'x1', 'y1')
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Ensure coordinates are non-negative."""
        if v < 0:
            raise ValueError(f"Coordinate must be non-negative, got {v}")
        return v

    def to_list(self) -> List[float]:
        """Convert to [x0, y0, x1, y2] list format."""
        return [self.x0, self.y0, self.x1, self.y1]

    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def center(self) -> tuple[float, float]:
        """Calculate center point (x, y)."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


class ConfidenceScores(BaseModel):
    """Multi-dimensional confidence scores for element quality."""

    extraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in text extraction accuracy"
    )
    classification: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in semantic element type"
    )
    clinical_context: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in clinical metadata and relationships"
    )

    def overall(self) -> float:
        """Calculate overall confidence as weighted average."""
        # Weight extraction and classification more heavily than clinical context
        return (
            self.extraction * 0.4 +
            self.classification * 0.4 +
            self.clinical_context * 0.2
        )

    def meets_threshold(self, threshold: float = 0.85) -> bool:
        """Check if overall confidence meets minimum threshold."""
        return self.overall() >= threshold


class ClinicalMetadata(BaseModel):
    """Clinical semantic metadata for enhanced processing."""

    temporal_qualifier: Optional[str] = Field(
        default=None,
        description="Temporal context: 'current', 'historical', 'planned'"
    )
    clinical_domain: Optional[str] = Field(
        default=None,
        description="Clinical domain: 'pharmacology', 'laboratory', 'vitals', etc."
    )
    cross_references: List[str] = Field(
        default_factory=list,
        description="Element IDs of related content"
    )
    requires_validation: bool = Field(
        default=False,
        description="Flag for human review requirement"
    )
    validation_reason: Optional[str] = Field(
        default=None,
        description="Reason for validation requirement"
    )


class ProcessingMetadata(BaseModel):
    """Provenance and processing metadata for auditability."""

    model_name: str = Field(..., description="VLM model used (e.g., 'gemini-1.5-flash')")
    processing_timestamp: str = Field(..., description="ISO 8601 timestamp")
    api_latency_ms: Optional[float] = Field(
        default=None,
        description="API call latency in milliseconds"
    )
    cost_estimate_usd: Optional[float] = Field(
        default=None,
        description="Estimated API cost in USD"
    )
    human_reviewed: bool = Field(default=False)
    review_flags: List[str] = Field(default_factory=list)


class ElementDetection(BaseModel):
    """
    Single detected element from VLM analysis.

    Represents one semantic element (text block, table, figure, etc.)
    extracted from a document page.
    """

    element_id: str = Field(..., description="Unique element identifier")
    semantic_type: ElementType = Field(..., description="Semantic element classification")
    bbox: BoundingBox = Field(..., description="Bounding box in pixel coordinates")
    content: str = Field(..., description="Extracted text content")
    confidence_scores: ConfidenceScores = Field(..., description="Multi-dimensional confidence")

    # Optional rich metadata
    clinical_metadata: Optional[ClinicalMetadata] = Field(
        default=None,
        description="Clinical semantic metadata"
    )
    processing_metadata: Optional[ProcessingMetadata] = Field(
        default=None,
        description="Processing provenance"
    )

    # Additional attributes for compatibility
    page: int = Field(..., description="Page number (0-indexed)")
    column: Optional[int] = Field(default=None, description="Column assignment")
    z_order: Optional[int] = Field(default=None, description="Global reading order")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format compatible with existing pipeline.

        Maintains backward compatibility with LayoutParser-based pipeline.
        """
        result = {
            "id": self.element_id,
            "type": self.semantic_type.value,
            "page": self.page,
            "bbox_px": self.bbox.to_list(),
            "payload": {
                "text": self.content,
                "confidence": self.confidence_scores.overall(),
            }
        }

        if self.column is not None:
            result["column"] = self.column

        if self.z_order is not None:
            result["z_order"] = self.z_order

        if self.clinical_metadata:
            result["clinical_metadata"] = self.clinical_metadata.model_dump(exclude_none=True)

        if self.processing_metadata:
            result["provenance"] = self.processing_metadata.model_dump(exclude_none=True)

        return result


class DocumentContext(BaseModel):
    """Context information provided to VLM for enhanced understanding."""

    document_type: Optional[str] = Field(
        default=None,
        description="Document type hint: 'Clinical Notes', 'Lab Results', etc."
    )
    section_hierarchy: List[str] = Field(
        default_factory=list,
        description="Hierarchical section path, e.g., ['Notes', 'Progress Notes']"
    )
    patient_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Patient metadata (age, gender, relevant clinical context)"
    )
    page_number: int = Field(..., description="Current page number")
    total_pages: int = Field(..., description="Total pages in document")
    preceding_summary: Optional[str] = Field(
        default=None,
        description="Summary of preceding sections for context"
    )


class VLMRequest(BaseModel):
    """Request payload for VLM element detection."""

    # Image data (base64 encoded or PIL Image handled separately)
    image_path: Optional[str] = Field(
        default=None,
        description="Path to image file (alternative to image_data)"
    )

    # Context for semantic understanding
    context: DocumentContext = Field(..., description="Document context for VLM")

    # Processing parameters
    max_tokens: int = Field(default=8192, description="Maximum response tokens")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")

    # Optional constraints
    expected_element_types: Optional[List[ElementType]] = Field(
        default=None,
        description="Hint for expected element types on this page"
    )


class VLMResponse(BaseModel):
    """Response from VLM element detection."""

    elements: List[ElementDetection] = Field(
        default_factory=list,
        description="Detected elements on page"
    )

    # Response metadata
    processing_metadata: ProcessingMetadata = Field(
        ...,
        description="Processing provenance and performance metrics"
    )

    # Quality indicators
    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average confidence across all elements"
    )
    requires_human_review: bool = Field(
        default=False,
        description="Flag if any elements require human review"
    )
    review_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons for human review requirement"
    )

    # Raw VLM response (for debugging)
    raw_response: Optional[str] = Field(
        default=None,
        description="Raw VLM response text for debugging"
    )

    def low_confidence_elements(self, threshold: float = 0.85) -> List[ElementDetection]:
        """Return elements below confidence threshold."""
        return [
            elem for elem in self.elements
            if not elem.confidence_scores.meets_threshold(threshold)
        ]

    def elements_by_type(self, element_type: ElementType) -> List[ElementDetection]:
        """Filter elements by semantic type."""
        return [elem for elem in self.elements if elem.semantic_type == element_type]

    def to_pipeline_format(self) -> List[Dict[str, Any]]:
        """
        Convert to format expected by existing pipeline.

        Returns list of element dictionaries compatible with
        column detection, hierarchy generation, and serialization.
        """
        return [elem.to_dict() for elem in self.elements]


class VLMError(BaseModel):
    """Structured error response from VLM processing."""

    error_type: str = Field(..., description="Error category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error context"
    )
    timestamp: str = Field(..., description="ISO 8601 error timestamp")
    recoverable: bool = Field(
        default=True,
        description="Whether error is recoverable via retry"
    )
