"""
Tesseract wrappers and preprocessing
"""

import numpy as np
import cv2
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available")

from .config import OCRConfig
from .utils import BBox


@dataclass
class OCRResult:
    """Result from OCR."""
    text: str
    confidence: float
    method: str  # "vector" or "ocr"


class OCREngine:
    """
    Tesseract OCR wrapper with preprocessing.
    """
    
    def __init__(
        self,
        config: OCRConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize OCR engine.
        
        Args:
            config: OCR configuration
            logger: Optional logger
        """
        if not PYTESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract is not installed")
        
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Test tesseract availability
        try:
            pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            self.logger.error(f"Tesseract not found: {e}")
            raise
    
    def ocr_region(
        self,
        image: np.ndarray,
        bbox_px: Optional[BBox] = None,
        psm: Optional[int] = None
    ) -> OCRResult:
        """
        Run OCR on image region.
        
        Args:
            image: RGB image as numpy array
            bbox_px: Bounding box to crop (if None, use full image)
            psm: Page segmentation mode (if None, use config default)
        
        Returns:
            OCRResult with text and confidence
        """
        # Crop region if bbox provided
        if bbox_px:
            x0, y0 = int(bbox_px.x0), int(bbox_px.y0)
            x1, y1 = int(bbox_px.x1), int(bbox_px.y1)
            
            # Clip to image bounds
            h, w = image.shape[:2]
            x0 = max(0, min(x0, w))
            y0 = max(0, min(y0, h))
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            
            if x1 <= x0 or y1 <= y0:
                return OCRResult(text="", confidence=0.0, method="ocr")
            
            crop = image[y0:y1, x0:x1]
        else:
            crop = image
        
        # Preprocess
        if self.config.preprocess.get("deskew") or self.config.preprocess.get("binarize"):
            crop = self._preprocess(crop)
        
        # Determine PSM
        if psm is None:
            psm = self.config.psm_text
        
        # Build tesseract config
        tess_config = f"--psm {psm}"
        if self.config.lang:
            tess_config = f"-l {self.config.lang} {tess_config}"
        
        # Run OCR
        try:
            text = pytesseract.image_to_string(crop, config=tess_config)
            
            # Get confidence
            try:
                data = pytesseract.image_to_data(crop, config=tess_config, output_type=pytesseract.Output.DICT)
                confidences = [c for c in data['conf'] if c != -1]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            except:
                avg_conf = 0.0
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_conf / 100.0,  # Normalize to 0-1
                method="ocr"
            )
        
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return OCRResult(text="", confidence=0.0, method="ocr")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR.
        
        Args:
            image: RGB image
        
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Deskew if enabled
        if self.config.preprocess.get("deskew"):
            gray = self._deskew(gray)
        
        # Binarize if enabled
        if self.config.preprocess.get("binarize"):
            gray = self._binarize(gray)
        
        return gray
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image using Hough transform.
        
        Args:
            image: Grayscale image
        
        Returns:
            Deskewed image
        """
        try:
            # Detect edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Detect lines
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                # Calculate median angle
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = (theta * 180 / np.pi) - 90
                    angles.append(angle)
                
                median_angle = np.median(angles)
                
                # Only rotate if angle is significant
                if abs(median_angle) > 0.5:
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    image = cv2.warpAffine(image, M, (w, h), 
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)
        except:
            pass  # Return original if deskew fails
        
        return image
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Binarize image using adaptive thresholding.
        
        Args:
            image: Grayscale image
        
        Returns:
            Binarized image
        """
        try:
            # Use Otsu's method
            _, binary = cv2.threshold(
                image, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary
        except:
            return image
    
    def get_ocr_info(self) -> Dict[str, Any]:
        """Get OCR engine information for manifest."""
        return {
            "engine": self.config.engine,
            "lang": self.config.lang,
            "psm_text": self.config.psm_text,
            "psm_table": self.config.psm_table,
            "preprocess": self.config.preprocess
        }
