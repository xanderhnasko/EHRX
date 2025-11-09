"""
Unit tests for OCR engine and element routing
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Test imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ehrx.ocr import OCREngine, OCRError, VectorTextExtractor
from ehrx.route import ElementRouter, ElementRoutingError
from ehrx.core.config import load_default_config


class TestOCREngine:
    """Test cases for OCR engine functionality."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_default_config()
    
    @pytest.fixture
    def mock_image(self):
        """Create a mock RGB image."""
        return np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    
    def test_ocr_engine_initialization(self, config):
        """Test OCR engine initialization."""
        with patch('ehrx.ocr.LAYOUTPARSER_AVAILABLE', True):
            with patch('ehrx.ocr.lp') as mock_lp:
                mock_agent = Mock()
                mock_lp.TesseractAgent.return_value = mock_agent
                
                engine = OCREngine(config.ocr)
                assert engine.config == config.ocr
                assert engine.ocr_agent == mock_agent
                mock_lp.TesseractAgent.assert_called_once_with(languages='eng')
    
    def test_ocr_engine_initialization_failure(self, config):
        """Test OCR engine initialization failure."""
        with patch('ehrx.ocr.LAYOUTPARSER_AVAILABLE', False):
            with pytest.raises(OCRError, match="LayoutParser not available"):
                OCREngine(config.ocr)
    
    def test_extract_text_success(self, config, mock_image):
        """Test successful text extraction."""
        with patch('ehrx.ocr.LAYOUTPARSER_AVAILABLE', True):
            with patch('ehrx.ocr.lp') as mock_lp:
                # Mock TesseractAgent
                mock_agent = Mock()
                mock_agent.detect.return_value = {
                    'text': 'Sample extracted text',
                    'conf': 0.95
                }
                mock_lp.TesseractAgent.return_value = mock_agent
                
                engine = OCREngine(config.ocr)
                result = engine.extract_text(mock_image)
                
                assert result['text'] == 'Sample extracted text'
                assert result['confidence'] == 0.95
                assert result['method'] == 'tesseract'
                assert 'preprocessing_applied' in result
    
    def test_extract_text_fallback(self, config, mock_image):
        """Test text extraction with fallback to simple detection."""
        with patch('ehrx.ocr.LAYOUTPARSER_AVAILABLE', True):
            with patch('ehrx.ocr.lp') as mock_lp:
                # Mock TesseractAgent - first call returns non-dict, second returns text
                mock_agent = Mock()
                mock_agent.detect.side_effect = [
                    'Simple text response',  # First call with return_response=True
                    'Simple text response'   # Second call with return_only_text=True
                ]
                mock_lp.TesseractAgent.return_value = mock_agent
                
                engine = OCREngine(config.ocr)
                result = engine.extract_text(mock_image)
                
                assert result['text'] == 'Simple text response'
                assert result['confidence'] is None
                assert result['method'] == 'tesseract'
    
    def test_preprocessing(self, config, mock_image):
        """Test image preprocessing."""
        with patch('ehrx.ocr.LAYOUTPARSER_AVAILABLE', True):
            with patch('ehrx.ocr.lp') as mock_lp:
                mock_agent = Mock()
                mock_agent.detect.return_value = 'Processed text'
                mock_lp.TesseractAgent.return_value = mock_agent
                
                engine = OCREngine(config.ocr)
                
                # Test with preprocessing enabled
                result = engine.extract_text(mock_image, apply_preprocessing=True)
                assert result['preprocessing_applied'] == True
                
                # Test with preprocessing disabled
                result = engine.extract_text(mock_image, apply_preprocessing=False)
                assert result['preprocessing_applied'] == False
    
    def test_text_cleaning(self, config):
        """Test text cleaning functionality."""
        with patch('ehrx.ocr.LAYOUTPARSER_AVAILABLE', True):
            with patch('ehrx.ocr.lp'):
                engine = OCREngine(config.ocr)
                
                # Test various text cleaning scenarios
                assert engine._clean_text("  hello   world  ") == "hello world"
                assert engine._clean_text("text\n\nwith\r\n\twhitespace") == "text with whitespace"
                assert engine._clean_text("") == ""
                assert engine._clean_text(None) == ""


class TestElementRouter:
    """Test cases for element routing functionality."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_default_config()
    
    @pytest.fixture 
    def mock_layout_block(self):
        """Create a mock layout block."""
        block = Mock()
        block.type = 'text_block'
        block.score = 0.85
        block.block = Mock()
        block.block.x_1 = 100
        block.block.y_1 = 50
        block.block.x_2 = 300
        block.block.y_2 = 150
        return block
    
    @pytest.fixture
    def mock_image(self):
        """Create a mock page image."""
        return np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    @pytest.fixture
    def mock_page_info(self):
        """Create mock page info."""
        return {
            "page_number": 1,
            "width_px": 600,
            "height_px": 400
        }
    
    @pytest.fixture
    def mock_mapper(self):
        """Create mock coordinate mapper."""
        mapper = Mock()
        mapper.pixel_to_pdf.return_value = [72, 100, 216, 200]
        return mapper
    
    def test_element_router_initialization(self, config):
        """Test element router initialization."""
        with patch('ehrx.route.OCREngine') as mock_ocr_engine:
            router = ElementRouter(config, "test_doc")
            assert router.doc_id == "test_doc"
            assert router.config == config
            mock_ocr_engine.assert_called_once_with(config.ocr)
    
    def test_block_type_mapping(self, config):
        """Test block type mapping."""
        with patch('ehrx.route.OCREngine'):
            router = ElementRouter(config, "test_doc")
            
            assert router._map_block_type("Text") == "text_block"
            assert router._map_block_type("Table") == "table"
            assert router._map_block_type("Figure") == "figure"
            assert router._map_block_type("unknown") == "text_block"  # Default
    
    def test_crop_image_region(self, config, mock_image):
        """Test image region cropping."""
        with patch('ehrx.route.OCREngine'):
            router = ElementRouter(config, "test_doc")
            
            # Test valid crop
            bbox = [100, 50, 300, 150]
            cropped = router._crop_image_region(mock_image, bbox)
            assert cropped is not None
            assert cropped.shape == (100, 200, 3)  # height, width, channels
            
            # Test invalid crop (coordinates out of order)
            bbox_invalid = [300, 150, 100, 50]
            cropped_invalid = router._crop_image_region(mock_image, bbox_invalid)
            assert cropped_invalid is None
            
            # Test crop too small
            bbox_small = [100, 50, 102, 52]  # 2x2 region
            cropped_small = router._crop_image_region(mock_image, bbox_small)
            assert cropped_small is None
    
    def test_create_base_element(self, config, mock_layout_block, mock_page_info):
        """Test base element creation."""
        with patch('ehrx.route.OCREngine'):
            router = ElementRouter(config, "test_doc")
            
            element = router._create_base_element(
                "E_0001", "text_block", mock_page_info,
                [72, 100, 216, 200], [100, 50, 300, 150],
                5, mock_layout_block
            )
            
            assert element["id"] == "E_0001"
            assert element["doc_id"] == "test_doc"
            assert element["page"] == 1
            assert element["type"] == "text_block"
            assert element["bbox_pdf"] == [72, 100, 216, 200]
            assert element["bbox_px"] == [100, 50, 300, 150]
            assert element["z_order"] == 5
            assert element["detector_conf"] == 0.85
            assert "created_at" in element
    
    def test_text_block_processing(self, config, mock_image):
        """Test text block processing."""
        with patch('ehrx.route.OCREngine') as mock_ocr_class:
            # Mock OCR engine
            mock_ocr = Mock()
            mock_ocr.extract_text.return_value = {
                'text': 'Extracted text content',
                'confidence': 0.92,
                'preprocessing_applied': True
            }
            mock_ocr_class.return_value = mock_ocr
            
            router = ElementRouter(config, "test_doc")
            
            # Create test element
            element = {"source": "ocr"}
            
            # Process text block
            payload = router._process_text_block(mock_image, Mock(), element)
            
            assert payload["text"] == "Extracted text content"
            assert payload["ocr_confidence"] == 0.92
            assert payload["preprocessing_applied"] == True
            mock_ocr.extract_text.assert_called_once_with(
                mock_image, element_type="text", apply_preprocessing=True
            )


class TestVectorTextExtractor:
    """Test cases for vector text extraction."""
    
    def test_vector_text_extractor_initialization(self):
        """Test vector text extractor initialization."""
        extractor = VectorTextExtractor()
        assert extractor is not None
    
    def test_extract_text_from_region_placeholder(self):
        """Test placeholder implementation returns None."""
        extractor = VectorTextExtractor()
        result = extractor.extract_text_from_region(Mock(), [0, 0, 100, 100])
        assert result is None


@pytest.fixture
def sample_test_image():
    """Create a simple test image with text-like patterns."""
    # Create a simple black and white image that could contain text
    image = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White background
    
    # Add some black "text-like" rectangles
    image[20:30, 20:60] = 0   # Simulate text line 1
    image[40:50, 20:80] = 0   # Simulate text line 2
    image[60:70, 20:100] = 0  # Simulate text line 3
    
    return image


# Integration tests (require actual dependencies)
class TestIntegrationOCR:
    """Integration tests that require actual OCR dependencies."""
    
    @pytest.mark.integration
    def test_real_ocr_engine_creation(self):
        """Test creating real OCR engine (requires tesseract)."""
        try:
            config = load_default_config()
            engine = OCREngine(config.ocr)
            assert engine is not None
        except (OCRError, ImportError) as e:
            pytest.skip(f"OCR dependencies not available: {e}")
    
    @pytest.mark.integration
    def test_real_element_router_creation(self):
        """Test creating real element router (requires all dependencies)."""
        try:
            config = load_default_config()
            router = ElementRouter(config, "test_integration")
            assert router is not None
        except (OCRError, ElementRoutingError, ImportError) as e:
            pytest.skip(f"Routing dependencies not available: {e}")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])