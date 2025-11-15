"""
Sub-document detection and hierarchical grouping for multi-page EHRs.

Organizes flat page lists into hierarchical sub-documents (Labs, Medications, etc.)
based on VLM-detected section headers.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


logger = logging.getLogger(__name__)


# Keyword mappings for sub-document type detection
SUBDOC_KEYWORDS = {
    "laboratory_results": [
        "lab", "laboratory", "blood", "urinalysis", "specimen", "pathology",
        "microbiology", "culture", "test results"
    ],
    "medications": [
        "medication", "pharmacy", "prescription", "rx", "drug", "med list",
        "discharge meds", "home medications"
    ],
    "radiology_imaging": [
        "radiology", "imaging", "x-ray", "ct", "mri", "ultrasound", "scan",
        "pet scan", "mammogram", "fluoroscopy"
    ],
    "vital_signs": [
        "vital signs", "vitals", "temperature", "blood pressure", "pulse",
        "respiratory rate", "oxygen saturation"
    ],
    "progress_notes": [
        "progress note", "clinical note", "h&p", "history and physical",
        "consultation", "discharge summary", "admission note"
    ],
    "procedures": [
        "procedure", "operation", "surgery", "operative note", "procedure note",
        "intervention"
    ],
    "orders": [
        "orders", "physician orders", "nursing orders", "diet order",
        "activity order"
    ],
    "immunizations": [
        "immunization", "vaccination", "vaccine", "immunization record"
    ],
    "allergies": [
        "allergy", "allergies", "adverse reaction", "drug allergy",
        "food allergy"
    ],
    "problem_list": [
        "problem list", "diagnosis", "diagnoses", "active problems",
        "past medical history", "pmh"
    ],
    "appointments": [
        "appointment", "scheduling", "visit summary", "encounter"
    ],
    "referrals": [
        "referral", "specialist referral", "consult request"
    ],
    "patient_instructions": [
        "patient instructions", "discharge instructions", "after visit summary",
        "patient education"
    ]
}


@dataclass
class SubDocument:
    """Represents a sub-document within a larger EHR."""
    id: str
    type: str
    title: str
    page_range: Tuple[int, int]  # (start, end) 0-indexed
    pages: List[Dict[str, Any]]
    confidence: float = 1.0


class SubDocumentGrouper:
    """
    Group pages into clinical sub-documents based on section headers.

    Uses VLM-detected section headers and keyword matching to identify
    document boundaries and organize multi-page clinical sections.
    """

    def __init__(self, confidence_threshold: float = 0.80):
        """
        Initialize sub-document grouper.

        Args:
            confidence_threshold: Minimum confidence for section header detection
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

    def group_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Organize flat page list into hierarchical sub-documents.

        Args:
            document_data: Document dictionary from DocumentPipeline

        Returns:
            Enhanced document dictionary with sub-document structure:
            {
                "document_id": "...",
                "patient_demographics": {...},  # If detected
                "sub_documents": [SubDocument, ...],
                "pages": [...],  # Original pages preserved
                "processing_stats": {...}
            }
        """
        self.logger.info("Starting sub-document grouping")

        pages = document_data.get("pages", [])
        if not pages:
            self.logger.warning("No pages to process")
            return document_data

        # Extract patient demographics (usually on first page)
        demographics = self._extract_demographics(pages)

        # Detect section boundaries
        section_boundaries = self._detect_section_boundaries(pages)

        # Group pages into sub-documents
        sub_documents = self._group_pages_by_sections(pages, section_boundaries)

        # Build enhanced document structure
        enhanced_doc = {
            **document_data,  # Preserve original data
            "patient_demographics": demographics,
            "sub_documents": [self._subdoc_to_dict(sd) for sd in sub_documents],
            "grouping_metadata": {
                "total_subdocuments": len(sub_documents),
                "section_boundaries_detected": len(section_boundaries),
                "confidence_threshold": self.confidence_threshold
            }
        }

        self.logger.info(f"Grouping complete: {len(sub_documents)} sub-documents detected")

        return enhanced_doc

    def _extract_demographics(self, pages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract patient demographics from document (usually first page).

        Args:
            pages: List of page dictionaries

        Returns:
            Demographics dictionary or None
        """
        if not pages:
            return None

        # Check first few pages for demographics
        for page in pages[:3]:
            for element in page.get("elements", []):
                if element.get("type") == "patient_demographics":
                    return {
                        "content": element.get("content"),
                        "page_number": page.get("page_number"),
                        "confidence": element.get("confidence", {}).get("overall", 0.0),
                        "bbox_pixel": element.get("bbox_pixel"),
                        "bbox_pdf": element.get("bbox_pdf")
                    }

        return None

    def _detect_section_boundaries(
        self,
        pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect section header boundaries across pages.

        Args:
            pages: List of page dictionaries

        Returns:
            List of section boundary markers:
            [{
                "page_number": int,
                "title": str,
                "subdoc_type": str,
                "confidence": float,
                "element": dict
            }]
        """
        boundaries = []

        for page in pages:
            page_num = page.get("page_number", 0)

            for element in page.get("elements", []):
                element_type = element.get("type")
                confidence = element.get("confidence", {}).get("overall", 0.0)

                # Look for section headers or document headers
                if element_type in ["section_header", "subsection_header", "document_header"]:
                    if confidence >= self.confidence_threshold:
                        title = element.get("content", "").strip()

                        # Classify sub-document type
                        subdoc_type = self._classify_subdoc_type(title)

                        boundaries.append({
                            "page_number": page_num,
                            "title": title,
                            "subdoc_type": subdoc_type,
                            "confidence": confidence,
                            "element": element
                        })

                        self.logger.debug(
                            f"Boundary detected on page {page_num}: "
                            f"{subdoc_type} - {title}"
                        )

        self.logger.info(f"Detected {len(boundaries)} section boundaries")
        return boundaries

    def _classify_subdoc_type(self, title: str) -> str:
        """
        Classify sub-document type based on section title.

        Args:
            title: Section header text

        Returns:
            Sub-document type key (e.g., "laboratory_results", "medications")
        """
        title_lower = title.lower()

        # Check against keyword mappings
        for subdoc_type, keywords in SUBDOC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return subdoc_type

        # Default to generic clinical content
        return "clinical_content"

    def _group_pages_by_sections(
        self,
        pages: List[Dict[str, Any]],
        section_boundaries: List[Dict[str, Any]]
    ) -> List[SubDocument]:
        """
        Group consecutive pages under same section heading.

        Args:
            pages: List of page dictionaries
            section_boundaries: List of detected section boundaries

        Returns:
            List of SubDocument objects
        """
        if not section_boundaries:
            # No boundaries detected - treat entire document as one sub-doc
            return [SubDocument(
                id="subdoc_001",
                type="clinical_content",
                title="Clinical Document",
                page_range=(pages[0]["page_number"], pages[-1]["page_number"]),
                pages=pages,
                confidence=0.5
            )]

        # Sort boundaries by page number
        boundaries_sorted = sorted(section_boundaries, key=lambda x: x["page_number"])

        sub_documents = []

        for i, boundary in enumerate(boundaries_sorted):
            start_page = boundary["page_number"]

            # Find end page (next boundary or last page)
            if i + 1 < len(boundaries_sorted):
                end_page = boundaries_sorted[i + 1]["page_number"] - 1
            else:
                end_page = pages[-1]["page_number"]

            # Extract pages for this sub-document
            subdoc_pages = [
                p for p in pages
                if start_page <= p["page_number"] <= end_page
            ]

            if subdoc_pages:
                subdoc_id = f"subdoc_{i+1:03d}"

                sub_documents.append(SubDocument(
                    id=subdoc_id,
                    type=boundary["subdoc_type"],
                    title=boundary["title"],
                    page_range=(start_page, end_page),
                    pages=subdoc_pages,
                    confidence=boundary["confidence"]
                ))

                self.logger.debug(
                    f"SubDoc {subdoc_id}: {boundary['subdoc_type']} "
                    f"(pages {start_page}-{end_page}, {len(subdoc_pages)} pages)"
                )

        return sub_documents

    def _subdoc_to_dict(self, subdoc: SubDocument) -> Dict[str, Any]:
        """Convert SubDocument to dictionary for JSON serialization."""
        return {
            "id": subdoc.id,
            "type": subdoc.type,
            "title": subdoc.title,
            "page_range": list(subdoc.page_range),
            "page_count": len(subdoc.pages),
            "pages": subdoc.pages,
            "confidence": subdoc.confidence,
            "element_count": sum(len(p.get("elements", [])) for p in subdoc.pages)
        }


def generate_hierarchical_index(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate lightweight hierarchical index without full page content.

    Useful for navigation and overview without loading full document.

    Args:
        document_data: Enhanced document with sub-documents

    Returns:
        Hierarchical index structure
    """
    sub_documents = document_data.get("sub_documents", [])

    index = {
        "document_id": document_data.get("document_id"),
        "source_pdf": document_data.get("source_pdf"),
        "total_pages": document_data.get("total_pages"),
        "patient_demographics": document_data.get("patient_demographics"),
        "sub_documents": [],
        "processing_stats": document_data.get("processing_stats")
    }

    for subdoc in sub_documents:
        # Build page summaries (without full elements)
        page_summaries = []
        for page in subdoc.get("pages", []):
            page_summaries.append({
                "page_number": page.get("page_number"),
                "element_count": len(page.get("elements", [])),
                "element_types": list(set(
                    e.get("type") for e in page.get("elements", [])
                ))
            })

        index["sub_documents"].append({
            "id": subdoc.get("id"),
            "type": subdoc.get("type"),
            "title": subdoc.get("title"),
            "page_range": subdoc.get("page_range"),
            "page_count": subdoc.get("page_count"),
            "pages": page_summaries,  # Lightweight summaries
            "confidence": subdoc.get("confidence"),
            "element_count": subdoc.get("element_count")
        })

    return index
