"""
Tests for VLM Pydantic models.

Tests cover data validation, serialization, and model methods.
"""

import pytest
from datetime import datetime

from ehrx.vlm.models import (
    ElementType,
    BoundingBox,
    ConfidenceScores,
    ClinicalMetadata,
    ProcessingMetadata,
    ElementDetection,
    DocumentContext,
    VLMRequest,
    VLMResponse,
)


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_valid_bbox(self):
        """Test valid bounding box creation."""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0)
        assert bbox.x0 == 10.0
        assert bbox.y0 == 20.0
        assert bbox.x1 == 100.0
        assert bbox.y1 == 200.0

    def test_bbox_to_list(self):
        """Test conversion to list format."""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0)
        assert bbox.to_list() == [10.0, 20.0, 100.0, 200.0]

    def test_bbox_area(self):
        """Test area calculation."""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=120.0)
        assert bbox.area() == 10000.0  # 100 * 100

    def test_bbox_center(self):
        """Test center point calculation."""
        bbox = BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0)
        assert bbox.center() == (50.0, 50.0)

    def test_negative_coordinates_rejected(self):
        """Test that negative coordinates are rejected."""
        with pytest.raises(ValueError):
            BoundingBox(x0=-10.0, y0=20.0, x1=100.0, y1=200.0)


class TestConfidenceScores:
    """Tests for ConfidenceScores model."""

    def test_valid_confidence_scores(self):
        """Test valid confidence score creation."""
        scores = ConfidenceScores(
            extraction=0.95,
            classification=0.88,
            clinical_context=0.90
        )
        assert scores.extraction == 0.95
        assert scores.classification == 0.88
        assert scores.clinical_context == 0.90

    def test_overall_confidence_calculation(self):
        """Test overall confidence weighted average."""
        scores = ConfidenceScores(
            extraction=1.0,
            classification=1.0,
            clinical_context=1.0
        )
        assert scores.overall() == 1.0

        scores = ConfidenceScores(
            extraction=0.8,
            classification=0.9,
            clinical_context=0.5
        )
        # 0.8*0.4 + 0.9*0.4 + 0.5*0.2 = 0.32 + 0.36 + 0.1 = 0.78
        assert abs(scores.overall() - 0.78) < 0.01

    def test_meets_threshold(self):
        """Test confidence threshold checking."""
        scores = ConfidenceScores(
            extraction=0.90,
            classification=0.90,
            clinical_context=0.90
        )
        assert scores.meets_threshold(0.85)
        assert not scores.meets_threshold(0.95)

    def test_out_of_range_scores_rejected(self):
        """Test that scores outside 0-1 range are rejected."""
        with pytest.raises(ValueError):
            ConfidenceScores(
                extraction=1.5,
                classification=0.9,
                clinical_context=0.9
            )


class TestClinicalMetadata:
    """Tests for ClinicalMetadata model."""

    def test_minimal_clinical_metadata(self):
        """Test clinical metadata with minimal fields."""
        metadata = ClinicalMetadata()
        assert metadata.temporal_qualifier is None
        assert metadata.clinical_domain is None
        assert metadata.cross_references == []
        assert metadata.requires_validation is False

    def test_full_clinical_metadata(self):
        """Test clinical metadata with all fields."""
        metadata = ClinicalMetadata(
            temporal_qualifier="current",
            clinical_domain="pharmacology",
            cross_references=["E_0001", "E_0002"],
            requires_validation=True,
            validation_reason="Unclear dosage"
        )
        assert metadata.temporal_qualifier == "current"
        assert metadata.clinical_domain == "pharmacology"
        assert len(metadata.cross_references) == 2
        assert metadata.requires_validation is True


class TestElementDetection:
    """Tests for ElementDetection model."""

    def test_minimal_element_detection(self):
        """Test element detection with minimal required fields."""
        element = ElementDetection(
            element_id="E_0001",
            semantic_type=ElementType.CLINICAL_PARAGRAPH,
            bbox=BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0),
            content="Patient reports feeling better.",
            confidence_scores=ConfidenceScores(
                extraction=0.95,
                classification=0.90,
                clinical_context=0.88
            ),
            page=0
        )
        assert element.element_id == "E_0001"
        assert element.semantic_type == ElementType.CLINICAL_PARAGRAPH
        assert element.page == 0

    def test_element_to_dict(self):
        """Test conversion to pipeline-compatible dictionary."""
        element = ElementDetection(
            element_id="E_0001",
            semantic_type=ElementType.MEDICATION_TABLE,
            bbox=BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0),
            content="Metformin 500mg PO BID",
            confidence_scores=ConfidenceScores(
                extraction=0.92,
                classification=0.88,
                clinical_context=0.85
            ),
            page=1,
            column=0,
            z_order=5
        )

        result = element.to_dict()

        assert result["id"] == "E_0001"
        assert result["type"] == "medication_table"
        assert result["page"] == 1
        assert result["column"] == 0
        assert result["z_order"] == 5
        assert result["bbox_px"] == [10.0, 20.0, 100.0, 200.0]
        assert result["payload"]["text"] == "Metformin 500mg PO BID"
        assert "confidence" in result["payload"]


class TestDocumentContext:
    """Tests for DocumentContext model."""

    def test_minimal_document_context(self):
        """Test document context with minimal fields."""
        context = DocumentContext(
            page_number=0,
            total_pages=10
        )
        assert context.page_number == 0
        assert context.total_pages == 10
        assert context.document_type is None

    def test_full_document_context(self):
        """Test document context with all fields."""
        context = DocumentContext(
            document_type="Clinical Notes",
            section_hierarchy=["Notes", "Progress Notes", "Daily Assessment"],
            patient_context={"age": 65, "gender": "M"},
            page_number=5,
            total_pages=20,
            preceding_summary="Patient admitted for chest pain..."
        )
        assert context.document_type == "Clinical Notes"
        assert len(context.section_hierarchy) == 3
        assert context.patient_context["age"] == 65


class TestVLMRequest:
    """Tests for VLMRequest model."""

    def test_vlm_request_with_image_path(self):
        """Test VLM request with image path."""
        context = DocumentContext(page_number=0, total_pages=1)
        request = VLMRequest(
            image_path="/path/to/image.png",
            context=context
        )
        assert request.image_path == "/path/to/image.png"
        assert request.max_tokens == 8192
        assert request.temperature == 0.1

    def test_vlm_request_custom_parameters(self):
        """Test VLM request with custom parameters."""
        context = DocumentContext(page_number=0, total_pages=1)
        request = VLMRequest(
            image_path="/path/to/image.png",
            context=context,
            max_tokens=4096,
            temperature=0.2
        )
        assert request.max_tokens == 4096
        assert request.temperature == 0.2


class TestVLMResponse:
    """Tests for VLMResponse model."""

    def test_empty_vlm_response(self):
        """Test VLM response with no elements."""
        metadata = ProcessingMetadata(
            model_name="gemini-1.5-flash",
            processing_timestamp=datetime.utcnow().isoformat()
        )
        response = VLMResponse(
            elements=[],
            processing_metadata=metadata,
            overall_confidence=0.0,
            requires_human_review=False
        )
        assert len(response.elements) == 0
        assert response.overall_confidence == 0.0

    def test_vlm_response_with_elements(self):
        """Test VLM response with multiple elements."""
        element1 = ElementDetection(
            element_id="E_0001",
            semantic_type=ElementType.SECTION_HEADER,
            bbox=BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=50.0),
            content="MEDICATIONS",
            confidence_scores=ConfidenceScores(
                extraction=0.98,
                classification=0.95,
                clinical_context=0.90
            ),
            page=0
        )

        element2 = ElementDetection(
            element_id="E_0002",
            semantic_type=ElementType.MEDICATION_TABLE,
            bbox=BoundingBox(x0=10.0, y0=60.0, x1=700.0, y1=300.0),
            content="Metformin 500mg PO BID",
            confidence_scores=ConfidenceScores(
                extraction=0.92,
                classification=0.88,
                clinical_context=0.85
            ),
            page=0
        )

        metadata = ProcessingMetadata(
            model_name="gemini-1.5-flash",
            processing_timestamp=datetime.utcnow().isoformat(),
            api_latency_ms=1234.5
        )

        response = VLMResponse(
            elements=[element1, element2],
            processing_metadata=metadata,
            overall_confidence=0.90,
            requires_human_review=False
        )

        assert len(response.elements) == 2
        assert response.overall_confidence == 0.90

    def test_low_confidence_elements(self):
        """Test filtering of low confidence elements."""
        high_conf_element = ElementDetection(
            element_id="E_0001",
            semantic_type=ElementType.CLINICAL_PARAGRAPH,
            bbox=BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0),
            content="High confidence content",
            confidence_scores=ConfidenceScores(
                extraction=0.95,
                classification=0.95,
                clinical_context=0.95
            ),
            page=0
        )

        low_conf_element = ElementDetection(
            element_id="E_0002",
            semantic_type=ElementType.HANDWRITTEN_ANNOTATION,
            bbox=BoundingBox(x0=10.0, y0=220.0, x1=100.0, y1=250.0),
            content="Unclear handwriting",
            confidence_scores=ConfidenceScores(
                extraction=0.60,
                classification=0.70,
                clinical_context=0.65
            ),
            page=0
        )

        metadata = ProcessingMetadata(
            model_name="gemini-1.5-flash",
            processing_timestamp=datetime.utcnow().isoformat()
        )

        response = VLMResponse(
            elements=[high_conf_element, low_conf_element],
            processing_metadata=metadata,
            overall_confidence=0.80
        )

        low_conf = response.low_confidence_elements(threshold=0.85)
        assert len(low_conf) == 1
        assert low_conf[0].element_id == "E_0002"

    def test_elements_by_type(self):
        """Test filtering elements by semantic type."""
        element1 = ElementDetection(
            element_id="E_0001",
            semantic_type=ElementType.MEDICATION_TABLE,
            bbox=BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0),
            content="Medications",
            confidence_scores=ConfidenceScores(
                extraction=0.95,
                classification=0.95,
                clinical_context=0.95
            ),
            page=0
        )

        element2 = ElementDetection(
            element_id="E_0002",
            semantic_type=ElementType.LAB_RESULTS_TABLE,
            bbox=BoundingBox(x0=10.0, y0=220.0, x1=100.0, y1=400.0),
            content="Lab results",
            confidence_scores=ConfidenceScores(
                extraction=0.95,
                classification=0.95,
                clinical_context=0.95
            ),
            page=0
        )

        metadata = ProcessingMetadata(
            model_name="gemini-1.5-flash",
            processing_timestamp=datetime.utcnow().isoformat()
        )

        response = VLMResponse(
            elements=[element1, element2],
            processing_metadata=metadata,
            overall_confidence=0.95
        )

        med_tables = response.elements_by_type(ElementType.MEDICATION_TABLE)
        assert len(med_tables) == 1
        assert med_tables[0].element_id == "E_0001"

    def test_to_pipeline_format(self):
        """Test conversion to pipeline-compatible format."""
        element = ElementDetection(
            element_id="E_0001",
            semantic_type=ElementType.CLINICAL_PARAGRAPH,
            bbox=BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0),
            content="Test content",
            confidence_scores=ConfidenceScores(
                extraction=0.95,
                classification=0.95,
                clinical_context=0.95
            ),
            page=0
        )

        metadata = ProcessingMetadata(
            model_name="gemini-1.5-flash",
            processing_timestamp=datetime.utcnow().isoformat()
        )

        response = VLMResponse(
            elements=[element],
            processing_metadata=metadata,
            overall_confidence=0.95
        )

        pipeline_format = response.to_pipeline_format()
        assert isinstance(pipeline_format, list)
        assert len(pipeline_format) == 1
        assert pipeline_format[0]["id"] == "E_0001"
        assert pipeline_format[0]["type"] == "clinical_paragraph"


class TestElementType:
    """Tests for ElementType enum."""

    def test_all_element_types_valid(self):
        """Test that all element types are valid enum values."""
        # Document structure
        assert ElementType.DOCUMENT_HEADER.value == "document_header"
        assert ElementType.PATIENT_DEMOGRAPHICS.value == "patient_demographics"
        assert ElementType.SECTION_HEADER.value == "section_header"

        # Clinical content
        assert ElementType.CLINICAL_PARAGRAPH.value == "clinical_paragraph"
        assert ElementType.MEDICATION_TABLE.value == "medication_table"
        assert ElementType.LAB_RESULTS_TABLE.value == "lab_results_table"

        # Special content
        assert ElementType.HANDWRITTEN_ANNOTATION.value == "handwritten_annotation"
        assert ElementType.MEDICAL_FIGURE.value == "medical_figure"

        # Administrative
        assert ElementType.UNCATEGORIZED.value == "uncategorized"

    def test_element_type_from_string(self):
        """Test creating ElementType from string value."""
        elem_type = ElementType("medication_table")
        assert elem_type == ElementType.MEDICATION_TABLE

    def test_invalid_element_type_rejected(self):
        """Test that invalid element type strings are rejected."""
        with pytest.raises(ValueError):
            ElementType("invalid_type")
