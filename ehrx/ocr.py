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
            
            # Run OCR with timing and optimized parameters
            with Timer(f"ocr_{element_type}", self.logger):
                # Get PSM setting based on element type
                psm = self._get_psm_for_element_type(element_type)
                
                # Use return_response=True to get confidence scores with custom PSM
                try:
                    ocr_result = self.ocr_agent.detect(
                        processed_image, 
                        return_response=True,
                        return_only_text=False,
                        psm=psm
                    )
                    self.logger.debug(f"OCR with PSM {psm} successful")
                except TypeError as e:
                    # Fallback if PSM parameter not supported in this version
                    self.logger.debug(f"PSM parameter not supported: {e}")
                    ocr_result = self.ocr_agent.detect(
                        processed_image, 
                        return_response=True,
                        return_only_text=False
                    )
                except Exception as e:
                    self.logger.warning(f"OCR with PSM failed: {e}, trying without PSM")
                    ocr_result = self.ocr_agent.detect(
                        processed_image, 
                        return_response=True,
                        return_only_text=False
                    )
            
            # Extract text and confidence from result
            if isinstance(ocr_result, dict) and 'text' in ocr_result:
                text = ocr_result['text']
                # Debug: Log all available keys for confidence extraction
                self.logger.debug(f"OCR result dict keys: {list(ocr_result.keys())}")
                
                # Extract confidence from the 'data' field which contains Tesseract TSV output
                confidence = 0.0
                if 'data' in ocr_result and ocr_result['data'] is not None:
                    try:
                        # Parse Tesseract TSV data to extract confidence scores
                        data = ocr_result['data']
                        if hasattr(data, 'empty') and not data.empty:  # pandas DataFrame
                            confidence = self._extract_confidence_from_dataframe(data)
                        elif isinstance(data, str) and data.strip():  # string TSV
                            confidence = self._extract_confidence_from_tsv(data)
                    except Exception as e:
                        self.logger.debug(f"Failed to extract confidence from TSV data: {e}")
                        
                # Fallback: try direct confidence fields
                if confidence == 0.0:
                    confidence = (ocr_result.get('conf') or 
                                 ocr_result.get('confidence') or 
                                 ocr_result.get('mean_confidence') or 
                                 0.0)
                
                self.logger.debug(f"Extracted confidence from dict: {confidence}")
            elif hasattr(ocr_result, 'text'):
                # LayoutParser TextBlock object
                text = ocr_result.text
                # Debug: Log all available attributes
                attrs = [attr for attr in dir(ocr_result) if not attr.startswith('_')]
                self.logger.debug(f"OCR result object attributes: {attrs}")
                confidence = getattr(ocr_result, 'confidence', getattr(ocr_result, 'score', 0.0))
                self.logger.debug(f"Extracted confidence from object: {confidence}")
            else:
                # Debug: Log what we actually got
                self.logger.debug(f"Unexpected OCR result type: {type(ocr_result)}, value: {ocr_result}")
                # Fallback to simple text detection
                text = self.ocr_agent.detect(processed_image, return_only_text=True)
                confidence = 0.0
            
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
    
    def _get_psm_for_element_type(self, element_type: str) -> int:
        """Get optimal PSM (Page Segmentation Mode) for element type.
        
        Args:
            element_type: Type of element ("text", "table", "handwriting")
            
        Returns:
            PSM value for Tesseract
        """
        psm_mapping = {
            "text": self.config.psm_text,      # Usually PSM 6 (uniform text block)
            "table": self.config.psm_table,  # Usually PSM 6 (uniform text block) 
            "handwriting": 8,                # PSM 8 (single word) - better for handwritten text
            "figure": 11                     # PSM 11 (sparse text) - for captions
        }
        
        return psm_mapping.get(element_type, self.config.psm_text)
    
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
        
        # Enhance contrast and reduce noise
        processed = self._enhance_image_quality(processed)
        
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
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better OCR results.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Enhanced grayscale image
        """
        try:
            # Apply slight Gaussian blur to reduce noise
            denoised = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Apply sharpening kernel for better text edges
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Ensure values are within valid range
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            return sharpened
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed, using original: {e}")
            return image
    
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
    
    def _extract_confidence_from_tsv(self, tsv_data: str) -> float:
        """Extract average confidence score from Tesseract TSV data.
        
        Args:
            tsv_data: TSV format data from Tesseract with confidence column
            
        Returns:
            Average confidence score (0-100 scale, converted to 0-1)
        """
        if not tsv_data or not tsv_data.strip():
            return 0.0
            
        lines = tsv_data.strip().split('\n')
        if len(lines) < 2:  # Need header + at least one data row
            return 0.0
            
        # Parse TSV header to find confidence column
        header = lines[0].split('\t')
        try:
            conf_idx = header.index('conf')
        except ValueError:
            # No confidence column found
            return 0.0
        
        # Extract confidence values from data rows
        confidences = []
        for line in lines[1:]:
            fields = line.split('\t')
            if len(fields) > conf_idx:
                try:
                    conf_val = float(fields[conf_idx])
                    if conf_val > 0:  # Only include positive confidence scores
                        confidences.append(conf_val)
                except (ValueError, IndexError):
                    continue
        
        if not confidences:
            return 0.0
            
        # Return average confidence, converted from 0-100 scale to 0-1 scale
        avg_confidence = sum(confidences) / len(confidences)
        return avg_confidence / 100.0
    
    def _extract_confidence_from_dataframe(self, df) -> float:
        """Extract average confidence score from Tesseract DataFrame.
        
        Args:
            df: pandas DataFrame from Tesseract with confidence column
            
        Returns:
            Average confidence score (0-100 scale, converted to 0-1)
        """
        try:
            if 'conf' not in df.columns:
                return 0.0
            
            # Filter out negative/zero confidence scores and get valid ones
            valid_conf = df['conf'][df['conf'] > 0]
            
            if len(valid_conf) == 0:
                return 0.0
            
            # Return average confidence, converted from 0-100 scale to 0-1 scale
            avg_confidence = valid_conf.mean()
            return avg_confidence / 100.0
            
        except Exception as e:
            self.logger.debug(f"Failed to extract confidence from DataFrame: {e}")
            return 0.0
    
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

