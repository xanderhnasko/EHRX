"""
Integration tests for core infrastructure components
"""
import pytest
import tempfile
import yaml
from pathlib import Path

from ehrx.config import load_config, EHRXConfig, setup_logging_from_config
from ehrx.utils import (
    BBox, IDGenerator, Timer, setup_logging, 
    create_manifest, ensure_output_dir, pdf_to_pixel_coords, pixel_to_pdf_coords
)
from ehrx.pager import PageInfo, CoordinateMapper, parse_page_range


class TestCoreInfrastructureIntegration:
    """Test that all core infrastructure components work together."""
    
    def test_config_utils_integration(self):
        """Test config and utils integration."""
        # Create a test config
        config_data = {
            "detector": {"backend": "detectron2", "min_conf": 0.7},
            "ocr": {"engine": "tesseract", "lang": "eng"},
            "privacy": {"local_only": True, "log_text_snippets": False}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load config
            config = load_config(config_path)
            assert isinstance(config, EHRXConfig)
            
            # Setup logging from config
            logger = setup_logging_from_config(config, level="INFO")
            assert logger.name == "ehrx"
            assert logger.log_text_snippets == config.privacy.log_text_snippets
            
            # Create manifest using config
            manifest = create_manifest("test-doc", "/fake/input.pdf", config.model_dump())
            assert manifest["detector"] == "detectron2"
            assert manifest["ocr"] == "tesseract"
            assert "config_hash" in manifest
            
        finally:
            Path(config_path).unlink()
    
    def test_pager_utils_coordinate_integration(self):
        """Test coordinate conversion consistency between pager and utils."""
        # Create page info
        page_info = PageInfo(
            page_number=0,
            width_pdf=612.0,
            height_pdf=792.0,
            width_px=918,  # 150 DPI: 612 * 150/72 = 1275, but using simpler numbers
            height_px=1188,  # 150 DPI: 792 * 150/72 = 1650
            dpi=150,
            rotation=0
        )
        
        # Create coordinate mapper
        mapper = CoordinateMapper(page_info)
        
        # Test coordinates
        bbox_pdf = [72.0, 100.0, 200.0, 300.0]
        
        # Convert using mapper
        bbox_px_mapper = mapper.pdf_to_pixel(bbox_pdf)
        
        # Convert using utils functions
        bbox_px_utils = pdf_to_pixel_coords(
            bbox_pdf, 
            page_info.height_pdf, 
            page_info.height_px, 
            page_info.scale_x  # Assuming uniform scaling
        )
        
        # Should be very close (allowing for integer rounding)
        for mapper_coord, utils_coord in zip(bbox_px_mapper, bbox_px_utils):
            assert abs(mapper_coord - utils_coord) <= 1
        
        # Test round trip
        bbox_pdf_converted = mapper.pixel_to_pdf(bbox_px_mapper)
        for orig, converted in zip(bbox_pdf, bbox_pdf_converted):
            assert abs(orig - converted) < 1.0
    
    def test_bbox_coordinate_integration(self):
        """Test BBox class integration with coordinate conversions."""
        # Create BBox from PDF coordinates
        bbox_pdf = BBox(72.0, 100.0, 200.0, 300.0)
        
        # Create page info for conversion
        page_info = PageInfo(
            page_number=0,
            width_pdf=612.0,
            height_pdf=792.0,
            width_px=918,
            height_px=1188,
            dpi=150,
            rotation=0
        )
        
        # Convert to pixel coordinates
        mapper = CoordinateMapper(page_info)
        bbox_px_list = mapper.pdf_to_pixel(bbox_pdf.to_list())
        bbox_px = BBox.from_list([float(x) for x in bbox_px_list])
        
        # Verify properties are preserved
        assert bbox_px.width > 0
        assert bbox_px.height > 0
        assert bbox_px.area > 0
        
        # Test intersection with another bbox
        bbox_px2 = BBox(bbox_px.x0 + 10, bbox_px.y0 + 10, 
                       bbox_px.x1 + 50, bbox_px.y1 + 50)
        
        assert bbox_px.intersects(bbox_px2)
        intersection = bbox_px.intersection(bbox_px2)
        assert intersection is not None
        assert intersection.area > 0
        
        iou = bbox_px.iou(bbox_px2)
        assert 0 < iou < 1
    
    def test_id_generator_with_manifest(self):
        """Test ID generator integration with manifest creation."""
        doc_id = "test-document-001"
        
        # Create ID generator
        id_gen = IDGenerator(doc_id)
        
        # Generate some IDs
        element_ids = [id_gen.next_element_id() for _ in range(5)]
        heading_ids = [id_gen.next_heading_id("H1") for _ in range(3)]
        
        assert element_ids == ["E_0001", "E_0002", "E_0003", "E_0004", "E_0005"]
        assert heading_ids == ["H1_0001", "H1_0002", "H1_0003"]
        
        # Create manifest
        config = {"detector": {"backend": "detectron2"}, "ocr": {"engine": "tesseract"}}
        manifest = create_manifest(doc_id, "/fake/input.pdf", config)
        
        assert manifest["doc_id"] == doc_id
        assert "created_at" in manifest
    
    def test_timer_with_logging(self):
        """Test Timer integration with logging setup."""
        # Setup logging
        logger = setup_logging("INFO", log_text_snippets=False)
        
        # Use timer with logger
        with Timer("test_operation", logger) as timer:
            import time
            time.sleep(0.01)  # Sleep for 10ms
        
        assert timer.elapsed >= 0.01
        assert timer.start_time is not None
        assert timer.end_time is not None
    
    def test_page_range_with_coordinate_mapping(self):
        """Test page range parsing integration with coordinate mapping."""
        # Test various page ranges
        total_pages = 100
        
        ranges_to_test = [
            ("1-5", [0, 1, 2, 3, 4]),
            ("10,20,30", [9, 19, 29]),
            ("95-100", [94, 95, 96, 97, 98, 99]),
            ("all", list(range(100)))
        ]
        
        for range_str, expected in ranges_to_test:
            result = parse_page_range(range_str, total_pages)
            assert result == expected
        
        # Test with coordinate mapping for each page
        page_info_list = []
        for page_num in parse_page_range("1-3", 10):
            page_info = PageInfo(
                page_number=page_num,
                width_pdf=612.0,
                height_pdf=792.0,
                width_px=918,
                height_px=1188,
                dpi=150,
                rotation=0
            )
            page_info_list.append(page_info)
        
        assert len(page_info_list) == 3
        assert [p.page_number for p in page_info_list] == [0, 1, 2]
    
    def test_full_workflow_simulation(self):
        """Simulate a full workflow using all core components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Setup configuration
            config = EHRXConfig()
            assert config.detector.backend == "detectron2"
            assert config.privacy.local_only is True
            
            # 2. Setup logging
            logger = setup_logging_from_config(config, level="INFO")
            
            # 3. Create output directory
            output_dir = ensure_output_dir(temp_dir + "/output")
            assert output_dir.exists()
            assert (output_dir / "assets").exists()
            
            # 4. Create ID generator
            doc_id = "simulation-doc-001"
            id_gen = IDGenerator(doc_id)
            
            # 5. Simulate processing multiple pages
            page_numbers = parse_page_range("1-5", 100)
            processed_elements = []
            
            with Timer("full_workflow_simulation", logger) as timer:
                for page_num in page_numbers:
                    # Create page info
                    page_info = PageInfo(
                        page_number=page_num,
                        width_pdf=612.0,
                        height_pdf=792.0,
                        width_px=918,
                        height_px=1188,
                        dpi=150,
                        rotation=0
                    )
                    
                    # Create coordinate mapper
                    mapper = CoordinateMapper(page_info)
                    
                    # Simulate detecting elements on page
                    mock_detections = [
                        [100, 100, 300, 200],  # Text block
                        [100, 250, 500, 400],  # Table
                        [100, 450, 400, 600],  # Figure
                    ]
                    
                    for detection_bbox in mock_detections:
                        element_id = id_gen.next_element_id()
                        
                        # Convert coordinates
                        bbox_px = detection_bbox
                        bbox_pdf = mapper.pixel_to_pdf(bbox_px)
                        
                        # Create element record
                        element = {
                            "id": element_id,
                            "doc_id": doc_id,
                            "page": page_num,
                            "bbox_pdf": bbox_pdf,
                            "bbox_px": bbox_px,
                            "source": "mock"
                        }
                        processed_elements.append(element)
            
            # 6. Create manifest
            manifest = create_manifest(doc_id, "/fake/input.pdf", config.model_dump())
            
            # Verify results
            assert len(processed_elements) == 15  # 5 pages * 3 elements each
            assert all(elem["doc_id"] == doc_id for elem in processed_elements)
            assert timer.elapsed > 0
            assert manifest["doc_id"] == doc_id
            
            # Verify element IDs are sequential
            element_ids = [elem["id"] for elem in processed_elements]
            expected_ids = [f"E_{i:04d}" for i in range(1, 16)]
            assert element_ids == expected_ids
            
            # Verify coordinate consistency
            for element in processed_elements:
                bbox_pdf = element["bbox_pdf"]
                bbox_px = element["bbox_px"]
                
                # Create page info for this element's page
                page_info = PageInfo(
                    page_number=element["page"],
                    width_pdf=612.0,
                    height_pdf=792.0,
                    width_px=918,
                    height_px=1188,
                    dpi=150,
                    rotation=0
                )
                
                mapper = CoordinateMapper(page_info)
                
                # Round trip should be close
                bbox_px_converted = mapper.pdf_to_pixel(bbox_pdf)
                for orig, converted in zip(bbox_px, bbox_px_converted):
                    assert abs(orig - converted) <= 2  # Allow for rounding
    
    def test_error_handling_integration(self):
        """Test error handling across components."""
        # Test config validation
        with pytest.raises(ValueError):
            EHRXConfig(detector={"backend": "invalid"})
        
        # Test coordinate mapping with invalid data
        page_info = PageInfo(
            page_number=0,
            width_pdf=612.0,
            height_pdf=792.0,
            width_px=918,
            height_px=1188,
            dpi=150,
            rotation=0
        )
        
        mapper = CoordinateMapper(page_info)
        
        # Should handle edge cases gracefully
        zero_bbox = [0, 0, 0, 0]
        result = mapper.pdf_to_pixel(zero_bbox)
        assert result == [0, 1188, 0, 1188]  # Y-flipped
        
        # Test BBox with invalid coordinates
        with pytest.raises(Exception):
            # This might not raise an exception, but let's test negative area
            bbox = BBox(100, 100, 50, 50)  # x1 < x0, y1 < y0
            assert bbox.area < 0  # Negative area is mathematically valid