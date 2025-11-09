"""
LayoutParser model wrapper for region detection
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import layoutparser as lp
import numpy as np
from pathlib import Path

# Core LayoutParser imports
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False

# Configuration and utilities
from .core.config import DetectorConfig
from .core.utils import Timer, safe_log_text


class LayoutDetectionError(Exception):
    """Custom exception for layout detection errors."""
    pass


class LayoutDetector:
    """LayoutParser model wrapper for document region detection."""
    
    def __init__(self, config: DetectorConfig):
        """Initialize layout detector with configuration.
        
        Args:
            config: DetectorConfig object with model settings
            
        Raises:
            LayoutDetectionError: If LayoutParser or backend not available
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        
        if not LAYOUTPARSER_AVAILABLE:
            raise LayoutDetectionError(
                "LayoutParser not available. Install with: pip install layoutparser"
            )
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LayoutParser model based on config."""
        try:
            if self.config.backend == "detectron2":
                self._initialize_detectron2_model()
            else:
                raise LayoutDetectionError(f"Only Detectron2 backend is currently supported")
                
            self.logger.info(f"Initialized {self.config.backend} layout model: {self.config.model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize layout model: {e}")
            raise LayoutDetectionError(f"Model initialization failed: {e}")
    
    def _initialize_detectron2_model(self):
        """Initialize Detectron2 backend model."""
        try:
            import detectron2
        except ImportError:
            raise LayoutDetectionError(
                "Detectron2 not available. Install with: pip install detectron2"
            )
        
        # Initialize model with PubLayNet
        self.model = lp.Detectron2LayoutModel(
            config_path=self.config.model,
            label_map=self.config.label_map,
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 
                self.config.min_conf
            ]
        )
    
    def detect_layout(self, image: np.ndarray) -> "lp.Layout":
        """Run layout detection on image.
        
        Args:
            image: RGB numpy array of shape (H, W, 3)
            
        Returns:
            LayoutParser Layout object with detected elements
            
        Raises:
            LayoutDetectionError: If detection fails
        """
        if self.model is None:
            raise LayoutDetectionError("Model not initialized")
        
        try:
            # Ensure image is in correct format (RGB)
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise LayoutDetectionError(f"Invalid image shape: {image.shape}. Expected (H, W, 3)")
            
            # Run detection
            with Timer("layout_detection", self.logger):
                layout = self.model.detect(image)
            
            # Filter by confidence and apply NMS if needed
            filtered_layout = self._post_process_layout(layout)
            
            self.logger.info(f"Detected {len(filtered_layout)} layout elements")
            self.logger.debug(f"Element types: {[block.type for block in filtered_layout]}")
            
            return filtered_layout
            
        except Exception as e:
            self.logger.error(f"Layout detection failed: {e}")
            raise LayoutDetectionError(f"Detection failed: {e}")
    
    def _post_process_layout(self, layout: "lp.Layout") -> "lp.Layout":
        """Post-process layout results with filtering and NMS.
        
        Args:
            layout: Raw layout from model
            
        Returns:
            Filtered and processed layout
        """
        # Filter by confidence threshold
        filtered_blocks = []
        for block in layout:
            if hasattr(block, 'score') and block.score >= self.config.min_conf:
                filtered_blocks.append(block)
            elif not hasattr(block, 'score'):
                # Some models may not provide scores
                filtered_blocks.append(block)
        
        # Apply NMS if configured
        if self.config.nms_iou < 1.0 and len(filtered_blocks) > 1:
            filtered_blocks = self._apply_nms(filtered_blocks)
        
        return lp.Layout(filtered_blocks)
    
    def _apply_nms(self, blocks: List) -> List:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        # Simple NMS implementation
        if blocks and hasattr(blocks[0], 'score'):
            blocks = sorted(blocks, key=lambda x: x.score, reverse=True)
        
        kept_blocks = []
        for block in blocks:
            # Check overlap with already kept blocks
            should_keep = True
            for kept_block in kept_blocks:
                if self._calculate_iou(block, kept_block) > self.config.nms_iou:
                    should_keep = False
                    break
            
            if should_keep:
                kept_blocks.append(block)
        
        return kept_blocks
    
    def _calculate_iou(self, block1, block2) -> float:
        """Calculate Intersection over Union between two blocks."""
        # Get coordinates
        x1_min, y1_min = block1.block.x_1, block1.block.y_1
        x1_max, y1_max = block1.block.x_2, block1.block.y_2
        x2_min, y2_min = block2.block.x_1, block2.block.y_1
        x2_max, y2_max = block2.block.x_2, block2.block.y_2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def visualize_detection(self, image: np.ndarray, layout: "lp.Layout", 
                          box_width: int = 3) -> np.ndarray:
        """Visualize layout detection results on image.
        
        Args:
            image: Original RGB image
            layout: Detected layout elements
            box_width: Width of bounding box lines
            
        Returns:
            Image with overlaid bounding boxes and labels
        """
        try:
            # Use LayoutParser's built-in visualization
            annotated_image = lp.draw_box(image, layout, box_width=box_width)
            
            self.logger.info(f"Generated visualization with {len(layout)} detected elements")
            
            return annotated_image
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            raise LayoutDetectionError(f"Visualization failed: {e}")
    
    def get_detection_stats(self, layout: "lp.Layout") -> Dict[str, Any]:
        """Get statistics about detected layout elements.
        
        Args:
            layout: Detected layout
            
        Returns:
            Dictionary with detection statistics
        """
        stats = {
            "total_elements": len(layout),
            "element_types": {},
            "confidence_stats": {}
        }
        
        # Count element types
        for block in layout:
            block_type = block.type if hasattr(block, 'type') else 'unknown'
            stats["element_types"][block_type] = stats["element_types"].get(block_type, 0) + 1
        
        # Confidence statistics if available
        scores = [block.score for block in layout if hasattr(block, 'score')]
        if scores:
            stats["confidence_stats"] = {
                "min": min(scores),
                "max": max(scores),
                "mean": sum(scores) / len(scores),
                "count": len(scores)
            }
        
        return stats