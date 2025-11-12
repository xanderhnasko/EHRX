"""
Tests for hierarchical document structuring.

This test suite validates:
1. Document label detection
2. Document grouping
3. Section detection
4. Category mapping
5. Visual debug output generation
"""
import json
import logging
from pathlib import Path

import pytest
import cv2
import numpy as np

from ehrx.hierarchy import (
    DocumentLabelDetector,
    DocumentGrouper,
    SectionDetector,
    CategoryMapper,
    HierarchyBuilder,
    CATEGORIES
)
from ehrx.visualize import HierarchyVisualizer
from ehrx.core.config import EHRXConfig
from ehrx.pdf.pager import PDFRasterizer
from ehrx.detect import LayoutDetector
from ehrx.ocr import OCREngine


logger = logging.getLogger(__name__)


@pytest.fixture
def test_config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    return EHRXConfig.from_yaml(config_path)


@pytest.fixture
def sample_pdf():
    """Path to sample PDF for testing."""
    # Use SENSITIVE_ehr2_copy.pdf as specified
    pdf_path = Path("/Users/justinjasper/Downloads/SampleEHR").glob("*ehr2*.pdf")
    pdf_list = list(pdf_path)
    
    if not pdf_list:
        pytest.skip("SENSITIVE_ehr2_copy.pdf not found")
    
    return pdf_list[0]


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    test_output = tmp_path / "hierarchy_test_output"
    test_output.mkdir(parents=True, exist_ok=True)
    return test_output


class TestDocumentLabelDetector:
    """Test document label detection."""
    
    def test_initialization(self, test_config):
        """Test detector initialization."""
        detector = DocumentLabelDetector(test_config.hierarchy)
        
        assert detector.top_region == 0.15
        assert detector.horizontal_start == 0.2
        assert detector.horizontal_end == 0.8
    
    def test_detect_label_simple(self):
        """Test label detection with simple mock data."""
        detector = DocumentLabelDetector()
        
        # Mock page info
        page_info = {"width_px": 1700, "height_px": 2200}
        
        # Mock elements - one label in detection region
        elements = [
            {
                "type": "text_block",
                "bbox_px": [700, 100, 1000, 130],  # Centered, near top
                "payload": {"text": "CLINICAL NOTES", "confidence": 0.95}
            },
            {
                "type": "text_block",
                "bbox_px": [100, 400, 800, 450],  # Below detection region
                "payload": {"text": "Patient information...", "confidence": 0.90}
            }
        ]
        
        label = detector.detect_label(elements, page_info)
        
        assert label == "CLINICAL NOTES"
    
    def test_detect_label_no_candidates(self):
        """Test when no label candidates are found."""
        detector = DocumentLabelDetector()
        
        page_info = {"width_px": 1700, "height_px": 2200}
        
        # Elements outside detection region
        elements = [
            {
                "type": "text_block",
                "bbox_px": [100, 1000, 800, 1050],  # Too far down
                "payload": {"text": "Some text", "confidence": 0.95}
            }
        ]
        
        label = detector.detect_label(elements, page_info)
        
        assert label is None
    
    def test_get_detection_region(self):
        """Test detection region calculation."""
        detector = DocumentLabelDetector()
        
        page_info = {"width_px": 1700, "height_px": 2200}
        x1, y1, x2, y2 = detector.get_detection_region(page_info)
        
        assert x1 == 1700 * 0.2  # 340
        assert y1 == 0
        assert x2 == 1700 * 0.8  # 1360
        assert y2 == 2200 * 0.15  # 330


class TestDocumentGrouper:
    """Test document grouping logic."""
    
    def test_group_consecutive_pages(self):
        """Test grouping of consecutive pages with same label."""
        grouper = DocumentGrouper()
        
        page_labels = [
            (1, "Demographics"),
            (2, "Demographics"),
            (3, "Clinical Notes"),
            (4, "Clinical Notes"),
            (5, "Clinical Notes"),
            (6, "Lab Results"),
        ]
        
        groups = grouper.group_pages(page_labels)
        
        assert len(groups) == 3
        assert groups[0].document_type == "Demographics"
        assert groups[0].pages == [1, 2]
        assert groups[1].document_type == "Clinical Notes"
        assert groups[1].pages == [3, 4, 5]
        assert groups[2].document_type == "Lab Results"
        assert groups[2].pages == [6]
    
    def test_group_with_unlabeled(self):
        """Test grouping with unlabeled pages."""
        grouper = DocumentGrouper()
        
        page_labels = [
            (1, None),
            (2, "Clinical Notes"),
            (3, None),
        ]
        
        groups = grouper.group_pages(page_labels)
        
        assert len(groups) == 3
        assert groups[0].document_type == "Unlabeled"
        assert groups[1].document_type == "Clinical Notes"
        assert groups[2].document_type == "Unlabeled"
    
    def test_empty_input(self):
        """Test with empty input."""
        grouper = DocumentGrouper()
        groups = grouper.group_pages([])
        
        assert len(groups) == 0


class TestSectionDetector:
    """Test section detection."""
    
    def test_initialization(self, test_config):
        """Test detector initialization."""
        detector = SectionDetector(test_config.hierarchy)
        
        assert detector.min_heading_height == 20
        assert detector.caps_ratio_threshold == 0.6
        assert len(detector.heading_patterns) > 0
    
    def test_classify_heading_keyword_match(self, test_config):
        """Test heading classification with keyword match."""
        detector = SectionDetector(test_config.hierarchy)
        
        # Mock element with keyword
        elem = {
            "type": "text_block",
            "bbox_px": [100, 200, 400, 230],  # Height = 30
            "payload": {"text": "MEDICATIONS", "confidence": 0.95},
            "page": 1
        }
        
        all_elements = [elem]
        
        is_heading, level, scores = detector._classify_heading(elem, all_elements, 0)
        
        assert is_heading == True
        assert scores["keyword_match"] == 1.0
    
    def test_classify_heading_caps_ratio(self, test_config):
        """Test heading classification with high caps ratio."""
        detector = SectionDetector(test_config.hierarchy)
        
        # Mock element with all caps but no keyword
        elem = {
            "type": "text_block",
            "bbox_px": [100, 200, 400, 225],  # Height = 25
            "payload": {"text": "PATIENT SUMMARY", "confidence": 0.95},
            "page": 1
        }
        
        all_elements = [elem]
        
        is_heading, level, scores = detector._classify_heading(elem, all_elements, 0)
        
        assert is_heading == True
        assert scores["caps_ratio"] >= 0.6
    
    def test_detect_sections_simple(self, test_config):
        """Test section detection with simple mock data."""
        detector = SectionDetector(test_config.hierarchy)
        
        # Mock elements: heading + content
        elements = [
            {
                "id": "E_001",
                "type": "text_block",
                "bbox_px": [100, 100, 400, 130],
                "payload": {"text": "MEDICATIONS", "confidence": 0.95},
                "page": 1
            },
            {
                "id": "E_002",
                "type": "text_block",
                "bbox_px": [100, 150, 600, 180],
                "payload": {"text": "Aspirin 81mg daily", "confidence": 0.92},
                "page": 1
            },
            {
                "id": "E_003",
                "type": "text_block",
                "bbox_px": [100, 200, 600, 230],
                "payload": {"text": "Lisinopril 10mg daily", "confidence": 0.93},
                "page": 1
            }
        ]
        
        sections = detector.detect_sections(elements, [1])
        
        # Should detect at least one section
        assert len(sections) >= 1


class TestCategoryMapper:
    """Test category mapping."""
    
    def test_map_medications(self):
        """Test mapping of medication document."""
        mapper = CategoryMapper()
        
        category = mapper.map_to_category("Medication List")
        
        assert category == "Meds"
    
    def test_map_lab_results(self):
        """Test mapping of lab results."""
        mapper = CategoryMapper()
        
        category = mapper.map_to_category("Laboratory Results")
        
        assert category == "Labs"
    
    def test_map_clinical_notes(self):
        """Test mapping of clinical notes."""
        mapper = CategoryMapper()
        
        category = mapper.map_to_category("Progress Notes")
        
        assert category == "Notes"
    
    def test_map_unlabeled(self):
        """Test mapping of unlabeled document."""
        mapper = CategoryMapper()
        
        category = mapper.map_to_category("Unlabeled")
        
        assert category == "Miscellaneous"
    
    def test_map_unknown(self):
        """Test mapping of unknown document type."""
        mapper = CategoryMapper()
        
        category = mapper.map_to_category("Random Document Type")
        
        assert category == "Miscellaneous"


class TestHierarchyBuilderIntegration:
    """Integration tests for full hierarchy building pipeline."""
    
    def test_build_hierarchy_end_to_end(
        self, 
        sample_pdf, 
        test_config, 
        output_dir
    ):
        """
        End-to-end test: process PDF and build hierarchy.
        
        This test:
        1. Loads a real EHR PDF
        2. Performs layout detection and OCR
        3. Builds hierarchical structure
        4. Generates visual debug output
        5. Validates the output structure
        """
        logger.info(f"Testing with PDF: {sample_pdf}")
        
        # Process only first 5 pages for testing
        max_pages = 5
        
        # Initialize components
        rasterizer = PDFRasterizer(sample_pdf)
        detector = LayoutDetector(test_config.detector)
        ocr_engine = OCREngine(test_config.ocr)
        hierarchy_builder = HierarchyBuilder(test_config.model_dump())
        visualizer = HierarchyVisualizer(output_dir)
        
        # Process pages
        pages_data = []
        page_images = {}
        
        for page_num in range(min(max_pages, rasterizer.page_count)):
            logger.info(f"Processing page {page_num}...")
            
            # Rasterize page
            page_image, page_info = rasterizer.rasterize_page(page_num, dpi=200)
            page_images[page_num] = page_image
            
            # Detect layout
            layout = detector.detect_layout(page_image)
            
            # Extract text for each block
            elements = []
            for block in layout:
                bbox_px = [block.block.x_1, block.block.y_1, 
                          block.block.x_2, block.block.y_2]
                
                elem = {
                    "id": f"E_{len(elements):04d}",
                    "page": page_num,
                    "type": getattr(block, 'type', 'text_block'),
                    "bbox_px": bbox_px,
                    "payload": {"text": "", "confidence": 0.0}
                }
                
                # OCR text blocks
                if elem["type"] == "text_block":
                    try:
                        x1, y1, x2, y2 = [int(c) for c in bbox_px]
                        cropped = page_image[y1:y2, x1:x2]
                        
                        if cropped.size > 0:
                            ocr_result = ocr_engine.extract_text(cropped, "text")
                            elem["payload"] = {
                                "text": ocr_result["text"],
                                "confidence": ocr_result.get("confidence", 0.0)
                            }
                    except Exception as e:
                        logger.warning(f"OCR failed: {e}")
                
                elements.append(elem)
            
            # Store page data
            page_data = {
                "page_num": page_num,
                "elements": elements,
                "page_info": {
                    "width_px": page_info.width_px,
                    "height_px": page_info.height_px,
                    "dpi": page_info.dpi
                }
            }
            pages_data.append(page_data)
            
            # Visualize label detection
            label = hierarchy_builder.label_detector.detect_label(
                elements, 
                page_data["page_info"]
            )
            detection_region = hierarchy_builder.label_detector.get_detection_region(
                page_data["page_info"]
            )
            
            # Find label element
            label_elem = None
            for elem in elements:
                if elem.get("payload", {}).get("text") == label:
                    label_elem = elem
                    break
            
            visualizer.visualize_label_detection(
                page_image,
                page_num,
                elements,
                detection_region,
                label,
                label_elem
            )
        
        # Build hierarchy
        hierarchy = hierarchy_builder.build_hierarchy(pages_data)
        
        # Validate structure
        assert "documents" in hierarchy
        assert "categories" in hierarchy
        assert "total_documents" in hierarchy
        assert "total_pages" in hierarchy
        
        assert hierarchy["total_pages"] == len(pages_data)
        assert len(hierarchy["documents"]) > 0
        
        # Check document structure
        for doc in hierarchy["documents"]:
            assert "document_type" in doc
            assert "category" in doc
            assert "page_range" in doc
            assert "pages" in doc
            assert "sections" in doc
            
            # Category should be valid
            assert doc["category"] in CATEGORIES
        
        # Save hierarchy to JSON
        hierarchy_file = output_dir / "hierarchy_output.json"
        with open(hierarchy_file, "w") as f:
            json.dump(hierarchy, f, indent=2)
        
        logger.info(f"Saved hierarchy to {hierarchy_file}")
        
        # Generate visual outputs
        visualizer.create_document_summary(hierarchy, sample_pdf.stem)
        visualizer.visualize_document_overview(hierarchy, page_images, sample_pdf.stem)
        
        # Visualize sections for each page
        for page_data in pages_data:
            page_num = page_data["page_num"]
            elements = page_data["elements"]
            
            # Detect headings for this page
            page_headings = []
            for elem in elements:
                if elem.get("type") != "text_block":
                    continue
                
                text = elem.get("payload", {}).get("text", "").strip()
                if not text:
                    continue
                
                # Check if it's a heading (simplified check)
                is_heading, level, scores = hierarchy_builder.section_detector._classify_heading(
                    elem, elements, elements.index(elem)
                )
                
                if is_heading:
                    page_headings.append({
                        "element": elem,
                        "text": text,
                        "level": level,
                        "scores": scores
                    })
            
            # Find document type for this page
            doc_type = None
            for doc in hierarchy["documents"]:
                if page_num in doc["pages"]:
                    doc_type = doc["document_type"]
                    break
            
            visualizer.visualize_sections(
                page_images[page_num],
                page_num,
                elements,
                page_headings,
                doc_type
            )
        
        logger.info("Integration test completed successfully!")
        logger.info(f"Visual outputs saved to: {output_dir}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("HIERARCHY TEST RESULTS")
        print("=" * 80)
        print(f"Total documents: {hierarchy['total_documents']}")
        print(f"Total pages: {hierarchy['total_pages']}")
        print(f"Categories: {', '.join(hierarchy['categories'])}")
        print("\nDocuments:")
        for doc in hierarchy["documents"]:
            print(f"  - {doc['document_type']} ({doc['category']})")
            print(f"    Pages: {doc['page_range'][0]}-{doc['page_range'][1]}")
            print(f"    Sections: {len(doc['sections'])}")
        print("=" * 80)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])

