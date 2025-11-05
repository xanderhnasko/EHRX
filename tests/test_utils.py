"""
Tests for utils module
"""
import pytest
import logging
import tempfile
from pathlib import Path
from ehrx.utils import (
    BBox, pdf_to_pixel_coords, pixel_to_pdf_coords, 
    IDGenerator, Timer, setup_logging, safe_log_text,
    create_manifest, ensure_output_dir, validate_pdf_path
)


class TestBBox:
    """Test BBox utility class."""
    
    def test_bbox_creation(self):
        bbox = BBox(10, 20, 100, 200)
        assert bbox.x0 == 10
        assert bbox.y0 == 20
        assert bbox.x1 == 100
        assert bbox.y1 == 200
    
    def test_bbox_properties(self):
        bbox = BBox(10, 20, 100, 200)
        assert bbox.width == 90
        assert bbox.height == 180
        assert bbox.area == 16200
        assert bbox.center == (55, 110)
    
    def test_bbox_conversions(self):
        bbox = BBox(10, 20, 100, 200)
        
        # Test to_list
        assert bbox.to_list() == [10, 20, 100, 200]
        
        # Test to_dict
        expected_dict = {"x0": 10, "y0": 20, "x1": 100, "y1": 200}
        assert bbox.to_dict() == expected_dict
        
        # Test from_list
        bbox2 = BBox.from_list([10, 20, 100, 200])
        assert bbox2.x0 == 10 and bbox2.y0 == 20 and bbox2.x1 == 100 and bbox2.y1 == 200
    
    def test_bbox_scaling(self):
        bbox = BBox(10, 20, 100, 200)
        scaled = bbox.scale(2.0, 1.5)
        
        assert scaled.x0 == 20
        assert scaled.y0 == 30
        assert scaled.x1 == 200
        assert scaled.y1 == 300
    
    def test_bbox_intersection(self):
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(50, 50, 150, 150)
        bbox3 = BBox(200, 200, 300, 300)
        
        # Test intersects
        assert bbox1.intersects(bbox2)
        assert not bbox1.intersects(bbox3)
        
        # Test intersection
        intersection = bbox1.intersection(bbox2)
        assert intersection is not None
        assert intersection.x0 == 50 and intersection.y0 == 50
        assert intersection.x1 == 100 and intersection.y1 == 100
        
        no_intersection = bbox1.intersection(bbox3)
        assert no_intersection is None
    
    def test_bbox_iou(self):
        bbox1 = BBox(0, 0, 100, 100)
        bbox2 = BBox(50, 50, 150, 150)
        bbox3 = BBox(200, 200, 300, 300)
        
        # Overlapping boxes
        iou = bbox1.iou(bbox2)
        expected_iou = 2500 / (10000 + 10000 - 2500)  # intersection / union
        assert abs(iou - expected_iou) < 1e-6
        
        # Non-overlapping boxes
        assert bbox1.iou(bbox3) == 0.0
        
        # Identical boxes
        assert bbox1.iou(bbox1) == 1.0


class TestCoordinateConversions:
    """Test coordinate conversion functions."""
    
    def test_pdf_to_pixel_coords(self):
        # PDF: [x0, y0, x1, y1] with origin at bottom-left
        bbox_pdf = [72, 100, 200, 300]  # x0, y0, x1, y1
        page_height_pdf = 792  # Standard letter height
        page_height_px = 1056  # At some scale
        scale = page_height_px / page_height_pdf  # ~1.33
        
        bbox_px = pdf_to_pixel_coords(bbox_pdf, page_height_pdf, page_height_px, scale)
        
        # Check that coordinates are integers
        assert all(isinstance(coord, int) for coord in bbox_px)
        
        # Check X coordinates are scaled correctly
        assert bbox_px[0] == int(72 * scale)  # x0
        assert bbox_px[2] == int(200 * scale)  # x1
        
        # Check Y coordinates are flipped and scaled
        assert bbox_px[1] == int((page_height_pdf - 300) * scale)  # y0 (top in pixels)
        assert bbox_px[3] == int((page_height_pdf - 100) * scale)  # y1 (bottom in pixels)
    
    def test_pixel_to_pdf_coords(self):
        # Test reverse conversion
        bbox_px = [96, 656, 266, 922]  # pixel coordinates
        page_height_pdf = 792
        page_height_px = 1056
        scale = page_height_px / page_height_pdf
        
        bbox_pdf = pixel_to_pdf_coords(bbox_px, page_height_pdf, page_height_px, scale)
        
        # Check that coordinates are floats
        assert all(isinstance(coord, float) for coord in bbox_pdf)
        
        # Check X coordinates
        assert abs(bbox_pdf[0] - 96/scale) < 1e-6  # x0
        assert abs(bbox_pdf[2] - 266/scale) < 1e-6  # x1
        
        # Check Y coordinates (flipped)
        assert abs(bbox_pdf[1] - (page_height_pdf - 922/scale)) < 1e-6  # y0
        assert abs(bbox_pdf[3] - (page_height_pdf - 656/scale)) < 1e-6  # y1
    
    def test_coordinate_round_trip(self):
        # Test that converting PDF -> pixel -> PDF gives back original
        original_bbox_pdf = [72.5, 100.25, 200.75, 300.5]
        page_height_pdf = 792
        page_height_px = 1056
        scale = page_height_px / page_height_pdf
        
        bbox_px = pdf_to_pixel_coords(original_bbox_pdf, page_height_pdf, page_height_px, scale)
        bbox_pdf_converted = pixel_to_pdf_coords(bbox_px, page_height_pdf, page_height_px, scale)
        
        # Should be close to original (within rounding error)
        for orig, converted in zip(original_bbox_pdf, bbox_pdf_converted):
            assert abs(orig - converted) < 2.0  # Allow for integer rounding


class TestIDGenerator:
    """Test ID generation."""
    
    def test_element_id_generation(self):
        gen = IDGenerator("test-doc")
        
        id1 = gen.next_element_id()
        id2 = gen.next_element_id()
        
        assert id1 == "E_0001"
        assert id2 == "E_0002"
    
    def test_heading_id_generation(self):
        gen = IDGenerator("test-doc")
        
        h1_id = gen.next_heading_id("H1")
        h2_id = gen.next_heading_id("H2")
        h1_id2 = gen.next_heading_id("H1")
        
        assert h1_id == "H1_0001"
        assert h2_id == "H2_0002"
        assert h1_id2 == "H1_0003"
    
    def test_id_generator_reset(self):
        gen = IDGenerator("test-doc")
        
        gen.next_element_id()
        gen.next_heading_id("H1")
        
        gen.reset()
        
        assert gen.next_element_id() == "E_0001"
        assert gen.next_heading_id("H1") == "H1_0001"


class TestTimer:
    """Test Timer context manager."""
    
    def test_timer_context_manager(self):
        import time
        
        with Timer("test") as timer:
            time.sleep(0.01)  # Sleep for 10ms
        
        assert timer.elapsed >= 0.01
        assert timer.start_time is not None
        assert timer.end_time is not None
    
    def test_timer_elapsed_property(self):
        timer = Timer("test")
        
        # Before entering context
        assert timer.elapsed == 0.0
        
        # During context
        with timer:
            import time
            time.sleep(0.005)
            elapsed_during = timer.elapsed
            assert elapsed_during >= 0.005
        
        # After context
        assert timer.elapsed >= 0.005


class TestLogging:
    """Test logging setup and PHI safety."""
    
    def test_setup_logging(self):
        logger = setup_logging("DEBUG", log_text_snippets=True)
        
        assert logger.name == "ehrx"
        assert logger.level == logging.DEBUG
        assert hasattr(logger, 'log_text_snippets')
        assert logger.log_text_snippets is True
    
    def test_safe_log_text_with_snippets_disabled(self):
        logger = setup_logging("INFO", log_text_snippets=False)
        
        text = "Patient John Doe has diabetes"
        result = safe_log_text(logger, text)
        
        assert result == f"<text:{len(text)} chars>"
        assert "John Doe" not in result
    
    def test_safe_log_text_with_snippets_enabled(self):
        logger = setup_logging("INFO", log_text_snippets=True)
        
        # Short text
        short_text = "Short text"
        result = safe_log_text(logger, short_text)
        assert result == f'"{short_text}"'
        
        # Long text
        long_text = "This is a very long text that exceeds the default maximum length"
        result = safe_log_text(logger, long_text, max_length=20)
        assert result == f'"{long_text[:20]}..."'


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_manifest(self):
        config = {
            "detector": {"backend": "detectron2"},
            "ocr": {"engine": "tesseract"}
        }
        
        manifest = create_manifest("test-doc", "/path/to/input.pdf", config)
        
        assert manifest["doc_id"] == "test-doc"
        assert manifest["input_path"] == "/path/to/input.pdf"
        assert manifest["detector"] == "detectron2"
        assert manifest["ocr"] == "tesseract"
        assert "created_at" in manifest
        assert "config_hash" in manifest
        assert len(manifest["config_hash"]) == 8
    
    def test_ensure_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output"
            
            result_path = ensure_output_dir(str(output_path))
            
            assert result_path.exists()
            assert result_path.is_dir()
            assert (result_path / "assets").exists()
    
    def test_validate_pdf_path_success(self):
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp_path = tmp.name
        
        try:
            result = validate_pdf_path(tmp_path)
            assert result == Path(tmp_path)
        finally:
            Path(tmp_path).unlink()  # Clean up
    
    def test_validate_pdf_path_not_found(self):
        with pytest.raises(FileNotFoundError):
            validate_pdf_path("/nonexistent/file.pdf")
    
    def test_validate_pdf_path_wrong_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"not a pdf")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="Input file must be a PDF"):
                validate_pdf_path(tmp_path)
        finally:
            Path(tmp_path).unlink()  # Clean up