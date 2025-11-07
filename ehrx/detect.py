"""
LayoutParser model wrapper for region detection

Detects layout regions (text blocks, tables, figures) using
LayoutParser with Detectron2 or PaddleDetection backend.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False
    logging.warning("LayoutParser not available")

from .utils import BBox
from .config import DetectorConfig


@dataclass
class DetectedBlock:
    """A detected layout block."""
    bbox_px: BBox  # Bounding box in pixel coordinates
    label: str  # Detected label (e.g., "Text", "Table", "Figure")
    label_mapped: str  # Mapped label (e.g., "text_block", "table", "figure")
    confidence: float  # Detection confidence score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox_px": self.bbox_px.to_list(),
            "label": self.label,
            "label_mapped": self.label_mapped,
            "confidence": self.confidence
        }


class LayoutDetector:
    """
    Layout detection using LayoutParser.
    
    Supports both Detectron2 and PaddleDetection backends.
    """
    
    def __init__(
        self,
        config: DetectorConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize layout detector.
        
        Args:
            config: Detector configuration
            logger: Optional logger
        """
        if not LAYOUTPARSER_AVAILABLE:
            raise RuntimeError("LayoutParser is not installed")
        
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.label_map = config.label_map
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the detection model."""
        from pathlib import Path
        
        backend = self.config.backend.lower()
        
        self.logger.info(f"Loading detector: {self.config.model} (backend: {backend})")
        
        # Check if we have a local model path
        model_path = getattr(self.config, 'model_path', None)
        if model_path:
            model_path = Path(model_path).expanduser()
            self.logger.info(f"Using local model: {model_path}")
        
        try:
            if backend == "detectron2":
                extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.config.min_conf]
                
                # Prefer passing model_path explicitly to avoid LP model zoo cache issues
                lp_model_path = None
                if model_path and model_path.exists():
                    lp_model_path = str(model_path)
                    # Also set as extra config for detectron2 cfg completeness
                    extra_config.extend(["MODEL.WEIGHTS", lp_model_path])
                
                self.model = lp.Detectron2LayoutModel(
                    config_path=self.config.model,
                    model_path=lp_model_path,
                    extra_config=extra_config,
                    label_map=self._get_original_label_map()
                )
            elif backend == "paddle":
                self.model = lp.PaddleDetectionLayoutModel(
                    config_path=self.config.model,
                    threshold=self.config.min_conf,
                    label_map=self._get_original_label_map()
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")
            
            self.logger.info("Detector loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load detector: {e}")
            raise
    
    def _get_original_label_map(self) -> Dict[int, str]:
        """
        Get original label map for LayoutParser.
        
        LayoutParser expects {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        for PubLayNet.
        """
        # PubLayNet standard labels
        return {
            0: "Text",
            1: "Title", 
            2: "List",
            3: "Table",
            4: "Figure"
        }
    
    def detect(
        self,
        image: np.ndarray,
        min_conf: Optional[float] = None
    ) -> List[DetectedBlock]:
        """
        Detect layout regions in an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            min_conf: Minimum confidence threshold (overrides config)
        
        Returns:
            List of DetectedBlock objects
        """
        if min_conf is None:
            min_conf = self.config.min_conf
        
        # Run detection
        layout = self.model.detect(image)
        
        # Convert to our format
        blocks = []
        for block in layout:
            # Get bounding box
            bbox_px = BBox(
                x0=block.coordinates[0],
                y0=block.coordinates[1],
                x1=block.coordinates[2],
                y1=block.coordinates[3]
            )
            
            # Get label
            label = block.type
            confidence = block.score
            
            # Filter by confidence
            if confidence < min_conf:
                continue
            
            # Map label
            label_mapped = self._map_label(label)
            
            blocks.append(DetectedBlock(
                bbox_px=bbox_px,
                label=label,
                label_mapped=label_mapped,
                confidence=confidence
            ))
        
        self.logger.debug(f"Detected {len(blocks)} blocks")
        
        # Apply NMS if needed
        if len(blocks) > 1 and self.config.nms_iou > 0:
            blocks = self._apply_nms(blocks, self.config.nms_iou)
            self.logger.debug(f"After NMS: {len(blocks)} blocks")
        
        return blocks
    
    def _map_label(self, label: str) -> str:
        """
        Map original label to standardized label.
        
        Args:
            label: Original label (e.g., "Text", "Table")
        
        Returns:
            Mapped label (e.g., "text_block", "table")
        """
        # Check if label is in our map
        if label in self.label_map:
            return self.label_map[label]
        
        # Try case-insensitive match
        for key, value in self.label_map.items():
            if key.lower() == label.lower():
                return value
        
        # Handle special cases
        if label.lower() in ["title", "list"]:
            return "text_block"
        
        # Default to text_block
        self.logger.warning(f"Unknown label '{label}', mapping to 'text_block'")
        return "text_block"
    
    def _apply_nms(
        self,
        blocks: List[DetectedBlock],
        iou_threshold: float
    ) -> List[DetectedBlock]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            blocks: List of detected blocks
            iou_threshold: IoU threshold for suppression
        
        Returns:
            Filtered list of blocks
        """
        if len(blocks) <= 1:
            return blocks
        
        # Sort by confidence (descending)
        blocks = sorted(blocks, key=lambda b: b.confidence, reverse=True)
        
        keep = []
        while blocks:
            # Keep the highest confidence block
            current = blocks.pop(0)
            keep.append(current)
            
            # Remove blocks that overlap too much
            remaining = []
            for block in blocks:
                iou = self._compute_iou(current.bbox_px, block.bbox_px)
                if iou < iou_threshold:
                    remaining.append(block)
            
            blocks = remaining
        
        return keep
    
    def _compute_iou(self, bbox1: BBox, bbox2: BBox) -> float:
        """Compute Intersection over Union between two boxes."""
        # Intersection
        x0 = max(bbox1.x0, bbox2.x0)
        y0 = max(bbox1.y0, bbox2.y0)
        x1 = min(bbox1.x1, bbox2.x1)
        y1 = min(bbox1.y1, bbox2.y1)
        
        if x1 < x0 or y1 < y0:
            return 0.0
        
        intersection = (x1 - x0) * (y1 - y0)
        
        # Union
        area1 = bbox1.area
        area2 = bbox2.area
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get detector information for manifest."""
        return {
            "backend": self.config.backend,
            "model": self.config.model,
            "min_conf": self.config.min_conf,
            "nms_iou": self.config.nms_iou
        }
