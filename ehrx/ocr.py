"""
OCR processing using LayoutParser TesseractAgent with preprocessing
"""
import logging
from typing import Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import cv2
from pathlib import Path

# LayoutParser imports
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False

# Configuration and utilities
from .core.config import OCRConfig
from .core.utils import Timer, safe_log_text


class OCRError(Exception):
    """Custom exception for OCR processing errors."""
    pass


class OCREngine:
    """OCR processing engine using LayoutParser TesseractAgent."""
    
    def __init__(self, config: OCRConfig):
        """Initialize OCR engine with configuration.
        
        Args:
            config: OCRConfig object with OCR settings
            
        Raises:
            OCRError: If LayoutParser or Tesseract not available
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ocr_agent = None
        
        if not LAYOUTPARSER_AVAILABLE:
            raise OCRError(
                "LayoutParser not available. Install with: pip install layoutparser"
            )
        
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the LayoutParser TesseractAgent."""
        try:
            # Initialize TesseractAgent with language configuration
            self.ocr_agent = lp.TesseractAgent(languages=self.config.lang)
            
            self.logger.info(f"Initialized Tesseract OCR agent with language: {self.config.lang}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR agent: {e}")
            raise OCRError(f"OCR agent initialization failed: {e}")
    
    def extract_text(self, image: np.ndarray, element_type: str = "text", 
                    apply_preprocessing: bool = True) -> Dict[str, Any]:
        """Extract text from image using OCR.
        
        Args:
            image: RGB numpy array of cropped region
            element_type: Type of element ("text", "table", "handwriting")
            apply_preprocessing: Whether to apply preprocessing
            
        Returns:
            Dictionary with OCR results:
            {
                "text": str,
                "confidence": float,
                "preprocessing_applied": bool,
                "method": str
            }
            
        Raises:
            OCRError: If OCR processing fails
        """
        if self.ocr_agent is None:
            raise OCRError("OCR agent not initialized")
        
        try:
            # Apply preprocessing if requested
            processed_image = image
            preprocessing_applied = False
            
            if apply_preprocessing and self.config.preprocess:
                processed_image = self._preprocess_image(image)
                preprocessing_applied = True
            
            # Run OCR with timing
            with Timer(f"ocr_{element_type}", self.logger):
                # Use return_response=True to get confidence scores
                ocr_result = self.ocr_agent.detect(
                    processed_image, 
                    return_response=True,
                    return_only_text=False
                )
            
            # Extract text and confidence from result
            if isinstance(ocr_result, dict) and 'text' in ocr_result:
                text = ocr_result['text']
                confidence = ocr_result.get('conf', 0.0)
            else:
                # Fallback to simple text detection
                text = self.ocr_agent.detect(processed_image, return_only_text=True)
                confidence = None
            
            # Clean up text
            text = self._clean_text(text)
            
            # Log results (PHI-safe)
            text_info = safe_log_text(self.logger, text, max_length=30)
            self.logger.debug(f"OCR extracted text: {text_info}")
            if confidence is not None:
                self.logger.debug(f"OCR confidence: {confidence:.2f}")
            
            return {
                "text": text,
                "confidence": confidence,
                "preprocessing_applied": preprocessing_applied,
                "method": "tesseract"
            }
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            raise OCRError(f"Text extraction failed: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to image before OCR.
        
        Args:
            image: RGB input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        processed = gray
        
        # Apply deskewing if configured
        if self.config.preprocess.deskew:
            processed = self._deskew_image(processed)
        
        # Apply binarization if configured
        if self.config.preprocess.binarize:
            processed = self._binarize_image(processed)
        
        # Convert back to RGB for LayoutParser
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        return processed
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew image to correct rotation."""
        try:
            # Simple deskewing using Hough line detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                # Calculate average angle
                angles = []
                for line in lines[:10]:  # Use first 10 lines
                    if line is not None and len(line) >= 2:
                        rho, theta = line[0], line[1]
                        angle = (theta - np.pi/2) * 180 / np.pi
                        angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    
                    # Only correct if angle is significant but not too extreme
                    if abs(median_angle) > 0.5 and abs(median_angle) < 45:
                        # Rotate image
                        height, width = image.shape
                        center = (width // 2, height // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        
                        image = cv2.warpAffine(image, rotation_matrix, (width, height),
                                             flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REPLICATE)
                        
                        self.logger.debug(f"Deskewed image by {median_angle:.1f} degrees")
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Deskewing failed: {e}")
            return image
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive binarization to improve OCR accuracy."""
        try:
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            self.logger.debug("Applied adaptive binarization")
            return binary
            
        except Exception as e:
            self.logger.warning(f"Binarization failed: {e}")
            return image
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters but keep newlines
        text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')
        
        return text.strip()
    
    def can_extract_vector_text(self) -> bool:
        """Check if vector text extraction is available.
        
        Note: Vector text extraction would be implemented in pager.py
        This is a placeholder for the interface.
        """
        # This would be implemented with PyMuPDF text extraction
        # For now, return False as it's not implemented yet
        return False


class VectorTextExtractor:
    """Placeholder for vector text extraction from PDF."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_region(self, page, bbox_pdf: list) -> Optional[str]:
        """Extract vector text from PDF region.
        
        This would be implemented using PyMuPDF to extract text
        directly from PDF without OCR.
        
        Args:
            page: PDF page object
            bbox_pdf: Bounding box in PDF coordinates [x0, y0, x1, y1]
            
        Returns:
            Extracted text or None if no vector text found
        """
        # TODO: Implement with PyMuPDF
        # For now, return None to force OCR
        return None

