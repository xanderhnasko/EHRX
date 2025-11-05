"""
Tests for pager module - PDF rasterization and coordinate mapping
"""
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ehrx.pager import (
    PageInfo, PDFRasterizer, Pager, CoordinateMapper
)


class TestPageInfo:
    """Test PageInfo data class."""
    
    def test_page_info_creation(self):
        page_info = PageInfo(
            page_number=1,
            width_pdf=612.0,
            height_pdf=792.0,
            width_px=816,
            height_px=1056,
            dpi=150,
            rotation=0
        )
        
        assert page_info.page_number == 1
        assert page_info.width_pdf == 612.0
        assert page_info.height_pdf == 792.0
        assert page_info.width_px == 816
        assert page_info.height_px == 1056
        assert page_info.dpi == 150
        assert page_info.rotation == 0
    
    def test_page_info_scale_calculation(self):
        page_info = PageInfo(
            page_number=1,
            width_pdf=612.0,
            height_pdf=792.0,
            width_px=816,
            height_px=1056,
            dpi=150,
            rotation=0
        )
        
        # Scale should be pixels/points
        expected_scale_x = 816 / 612.0
        expected_scale_y = 1056 / 792.0
        
        assert abs(page_info.scale_x - expected_scale_x) < 1e-6
        assert abs(page_info.scale_y - expected_scale_y) < 1e-6


class TestCoordinateMapper:
    """Test coordinate mapping utilities."""
    
    def test_coordinate_mapper_creation(self):
        page_info = PageInfo(
            page_number=1,
            width_pdf=612.0,
            height_pdf=792.0,
            width_px=816,
            height_px=1056,
            dpi=150,
            rotation=0
        )
        
        mapper = CoordinateMapper(page_info)
        assert mapper.page_info == page_info
    
    def test_pdf_to_pixel_conversion(self):
        page_info = PageInfo(
            page_number=1,
            width_pdf=612.0,
            height_pdf=792.0,
            width_px=816,
            height_px=1056,
            dpi=150,
            rotation=0
        )
        
        mapper = CoordinateMapper(page_info)
        
        # Test conversion
        bbox_pdf = [72.0, 100.0, 200.0, 300.0]
        bbox_px = mapper.pdf_to_pixel(bbox_pdf)
        
        # Check that result is list of integers
        assert isinstance(bbox_px, list)
        assert len(bbox_px) == 4
        assert all(isinstance(coord, int) for coord in bbox_px)
        
        # Check coordinate transformation (Y should be flipped)
        scale_x = page_info.scale_x
        scale_y = page_info.scale_y
        
        expected_x0 = int(72.0 * scale_x)
        expected_x1 = int(200.0 * scale_x)
        expected_y0 = int((792.0 - 300.0) * scale_y)  # Y flipped
        expected_y1 = int((792.0 - 100.0) * scale_y)  # Y flipped
        
        assert bbox_px == [expected_x0, expected_y0, expected_x1, expected_y1]
    
    def test_pixel_to_pdf_conversion(self):
        page_info = PageInfo(
            page_number=1,
            width_pdf=612.0,
            height_pdf=792.0,
            width_px=816,
            height_px=1056,
            dpi=150,
            rotation=0
        )
        
        mapper = CoordinateMapper(page_info)
        
        # Test reverse conversion
        bbox_px = [96, 656, 266, 922]
        bbox_pdf = mapper.pixel_to_pdf(bbox_px)
        
        # Check that result is list of floats
        assert isinstance(bbox_pdf, list)
        assert len(bbox_pdf) == 4
        assert all(isinstance(coord, float) for coord in bbox_pdf)
    
    def test_coordinate_round_trip(self):
        page_info = PageInfo(
            page_number=1,
            width_pdf=612.0,
            height_pdf=792.0,
            width_px=816,
            height_px=1056,
            dpi=150,
            rotation=0
        )
        
        mapper = CoordinateMapper(page_info)
        
        # Test round trip conversion
        original_bbox_pdf = [72.5, 100.25, 200.75, 300.5]
        bbox_px = mapper.pdf_to_pixel(original_bbox_pdf)
        bbox_pdf_converted = mapper.pixel_to_pdf(bbox_px)
        
        # Should be close to original (within rounding error)
        for orig, converted in zip(original_bbox_pdf, bbox_pdf_converted):
            assert abs(orig - converted) < 2.0


class TestPDFRasterizer:
    """Test PDF rasterization functionality."""
    
    @patch('ehrx.pager.fitz')
    def test_pdf_rasterizer_creation_with_pymupdf(self, mock_fitz):
        # Mock PyMuPDF availability
        mock_doc = MagicMock()
        mock_fitz.open.return_value = mock_doc
        mock_doc.page_count = 5
        
        rasterizer = PDFRasterizer("/fake/path.pdf")
        
        assert rasterizer.pdf_path == Path("/fake/path.pdf")
        assert rasterizer.backend == "pymupdf"
        assert rasterizer.page_count == 5
        mock_fitz.open.assert_called_once_with("/fake/path.pdf")
    
    @patch('ehrx.pager.fitz', side_effect=ImportError)
    @patch('ehrx.pager.convert_from_path')
    def test_pdf_rasterizer_creation_with_pdf2image(self, mock_convert, mock_fitz):
        # Mock pdf2image as fallback
        mock_convert.return_value = [Mock(), Mock(), Mock()]  # 3 pages
        
        rasterizer = PDFRasterizer("/fake/path.pdf")
        
        assert rasterizer.pdf_path == Path("/fake/path.pdf")
        assert rasterizer.backend == "pdf2image"
        assert rasterizer.page_count == 3
    
    @patch('ehrx.pager.fitz', side_effect=ImportError)
    @patch('ehrx.pager.convert_from_path', side_effect=ImportError)
    def test_pdf_rasterizer_no_backend_available(self, mock_convert, mock_fitz):
        with pytest.raises(RuntimeError, match="No PDF backend available"):
            PDFRasterizer("/fake/path.pdf")
    
    @patch('ehrx.pager.fitz')
    def test_rasterize_page_pymupdf(self, mock_fitz):
        # Setup mocks
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_matrix = Mock()
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = mock_matrix
        mock_doc.page_count = 1
        mock_doc[0] = mock_page
        
        # Mock page properties
        mock_page.rect.width = 612.0
        mock_page.rect.height = 792.0
        mock_page.rotation = 0
        
        # Mock pixmap
        mock_pixmap = Mock()
        mock_pixmap.width = 816
        mock_pixmap.height = 1056
        mock_pixmap.n = 3  # RGB channels
        mock_pixmap.samples = b'fake_image_data'
        mock_page.get_pixmap.return_value = mock_pixmap
        
        # Mock numpy array
        fake_array = np.zeros((1056, 816, 3), dtype=np.uint8)
        # Need to return the right size for reshaping: height * width * channels
        expected_size = 1056 * 816 * 3
        fake_data = np.zeros(expected_size, dtype=np.uint8)
        with patch('numpy.frombuffer', return_value=fake_data):
            rasterizer = PDFRasterizer("/fake/path.pdf")
            image, page_info = rasterizer.rasterize_page(0, dpi=150)
        
        # Verify results
        assert isinstance(image, np.ndarray)
        assert image.shape == (1056, 816, 3)
        assert isinstance(page_info, PageInfo)
        assert page_info.page_number == 0
        assert page_info.width_pdf == 612.0
        assert page_info.height_pdf == 792.0
        assert page_info.width_px == 816
        assert page_info.height_px == 1056
        assert page_info.dpi == 150
        assert page_info.rotation == 0
    
    @patch('ehrx.pager.fitz', side_effect=ImportError)
    @patch('ehrx.pager.convert_from_path')
    def test_rasterize_page_pdf2image(self, mock_convert, mock_fitz):
        # Mock PIL Image
        mock_image = Mock()
        mock_image.size = (816, 1056)  # width, height
        fake_array = np.zeros((1056, 816, 3), dtype=np.uint8)
        mock_image.convert.return_value = mock_image
        
        mock_convert.return_value = [mock_image]
        
        with patch('numpy.array', return_value=fake_array):
            rasterizer = PDFRasterizer("/fake/path.pdf")
            image, page_info = rasterizer.rasterize_page(0, dpi=150)
        
        # Verify results
        assert isinstance(image, np.ndarray)
        assert image.shape == (1056, 816, 3)
        assert isinstance(page_info, PageInfo)
        assert page_info.page_number == 0
        assert page_info.width_px == 816
        assert page_info.height_px == 1056
        assert page_info.dpi == 150
    
    @patch('ehrx.pager.fitz')
    def test_extract_vector_text_pymupdf(self, mock_fitz):
        # Setup mocks
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_fitz.open.return_value = mock_doc
        mock_doc.page_count = 1
        mock_doc[0] = mock_page
        
        # Mock text blocks
        mock_text_blocks = [
            (10, 20, 100, 50, "Sample text block", 0, 0),
            (20, 60, 150, 90, "Another text block", 1, 0)
        ]
        mock_page.get_text.return_value = mock_text_blocks
        
        rasterizer = PDFRasterizer("/fake/path.pdf")
        text_blocks = rasterizer.extract_vector_text(0)
        
        assert len(text_blocks) == 2
        
        block1 = text_blocks[0]
        assert block1["bbox"] == [10, 20, 100, 50]
        assert block1["text"] == "Sample text block"
        assert block1["block_no"] == 0
        assert block1["line_no"] == 0
        
        block2 = text_blocks[1]
        assert block2["bbox"] == [20, 60, 150, 90]
        assert block2["text"] == "Another text block"
        assert block2["block_no"] == 1
        assert block2["line_no"] == 0


class TestPager:
    """Test main Pager class."""
    
    @patch('ehrx.pager.fitz')
    def test_pager_creation(self, mock_fitz):
        mock_doc = MagicMock()
        mock_fitz.open.return_value = mock_doc
        mock_doc.page_count = 10
        
        pager = Pager("/fake/path.pdf")
        
        assert pager.pdf_path == Path("/fake/path.pdf")
        assert pager.page_count == 10
        assert isinstance(pager.rasterizer, PDFRasterizer)
    
    @patch('ehrx.pager.fitz')
    def test_pager_pages_iterator(self, mock_fitz):
        # Setup mocks
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_fitz.open.return_value = mock_doc
        mock_doc.page_count = 3
        mock_doc[0] = mock_page
        mock_doc[1] = mock_page
        mock_doc[2] = mock_page
        
        # Mock page properties
        mock_page.rect.width = 612.0
        mock_page.rect.height = 792.0
        mock_page.rotation = 0
        
        # Mock pixmap
        mock_pixmap = Mock()
        mock_pixmap.width = 816
        mock_pixmap.height = 1056
        mock_pixmap.n = 3  # RGB channels
        mock_pixmap.samples = b'fake_image_data'
        mock_page.get_pixmap.return_value = mock_pixmap
        
        fake_array = np.zeros((1056, 816, 3), dtype=np.uint8)
        
        with patch('numpy.frombuffer', return_value=fake_array.flatten()):
            pager = Pager("/fake/path.pdf")
            
            # Test iterator
            pages = list(pager.pages())
            
            assert len(pages) == 3
            for i, (image, page_info, mapper) in enumerate(pages):
                assert isinstance(image, np.ndarray)
                assert isinstance(page_info, PageInfo)
                assert isinstance(mapper, CoordinateMapper)
                assert page_info.page_number == i
    
    @patch('ehrx.pager.fitz')
    def test_pager_pages_with_range(self, mock_fitz):
        # Setup mocks
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_fitz.open.return_value = mock_doc
        mock_doc.page_count = 10
        # Mock pages 1, 2, 3 (0-indexed)
        mock_doc[1] = mock_page
        mock_doc[2] = mock_page
        mock_doc[3] = mock_page
        
        # Mock page properties
        mock_page.rect.width = 612.0
        mock_page.rect.height = 792.0
        mock_page.rotation = 0
        
        # Mock pixmap
        mock_pixmap = Mock()
        mock_pixmap.width = 816
        mock_pixmap.height = 1056
        mock_pixmap.n = 3  # RGB channels
        mock_pixmap.samples = b'fake_image_data'
        mock_page.get_pixmap.return_value = mock_pixmap
        
        fake_array = np.zeros((1056, 816, 3), dtype=np.uint8)
        
        with patch('numpy.frombuffer', return_value=fake_array.flatten()):
            pager = Pager("/fake/path.pdf")
            
            # Test specific page range
            pages = list(pager.pages(page_range="2-4"))  # Pages 2, 3, 4 (0-indexed: 1, 2, 3)
            
            assert len(pages) == 3
            for i, (image, page_info, mapper) in enumerate(pages):
                assert page_info.page_number == i + 1  # 1, 2, 3
    
    @patch('ehrx.pager.fitz')
    def test_pager_get_page_vector_text(self, mock_fitz):
        # Setup mocks
        mock_doc = MagicMock()
        mock_page = Mock()
        mock_fitz.open.return_value = mock_doc
        mock_doc.page_count = 1
        mock_doc[0] = mock_page
        
        # Mock text blocks
        mock_text_blocks = [
            (10, 20, 100, 50, "Sample text", 0, 0)
        ]
        mock_page.get_text.return_value = mock_text_blocks
        
        pager = Pager("/fake/path.pdf")
        text_blocks = pager.get_page_vector_text(0)
        
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Sample text"


class TestIntegration:
    """Integration tests for pager functionality."""
    
    def test_page_range_parsing(self):
        # Test various page range formats
        from ehrx.pager import parse_page_range
        
        # Single page
        assert parse_page_range("5", 10) == [4]  # 0-indexed
        
        # Range
        assert parse_page_range("3-6", 10) == [2, 3, 4, 5]  # 0-indexed
        
        # Multiple ranges
        assert parse_page_range("1-3,7-8", 10) == [0, 1, 2, 6, 7]  # 0-indexed
        
        # All pages
        assert parse_page_range("all", 5) == [0, 1, 2, 3, 4]
        
        # Out of bounds (should be clamped)
        assert parse_page_range("8-12", 10) == [7, 8, 9]
    
    def test_dpi_scaling_calculations(self):
        # Test that DPI scaling is calculated correctly
        page_info = PageInfo(
            page_number=0,
            width_pdf=612.0,  # 8.5 inches at 72 DPI
            height_pdf=792.0,  # 11 inches at 72 DPI
            width_px=1275,    # 8.5 inches at 150 DPI
            height_px=1650,   # 11 inches at 150 DPI
            dpi=150,
            rotation=0
        )
        
        # Scale should be DPI / 72 (PDF default)
        expected_scale = 150 / 72
        assert abs(page_info.scale_x - expected_scale) < 1e-6
        assert abs(page_info.scale_y - expected_scale) < 1e-6