"""
Utilities: bbox utils, id generation, logging, timers
"""

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass


# ============================================================================
# Logging Setup (PHI-safe)
# ============================================================================

def setup_logger(name: str, level: str = "INFO", log_text: bool = False) -> logging.Logger:
    """
    Setup a PHI-safe logger.
    
    Args:
        name: Logger name
        level: Log level (INFO, DEBUG, WARNING, ERROR)
        log_text: Whether to allow text snippets in logs (default: False for privacy)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler with formatting
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Store config in logger
    logger.allow_text = log_text
    
    return logger


# ============================================================================
# ID Generation
# ============================================================================

def generate_doc_id(prefix: str = "doc") -> str:
    """Generate a unique document ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{short_uuid}"


def generate_element_id(element_type: str, sequence: int) -> str:
    """
    Generate element ID following SPECS pattern.
    
    Args:
        element_type: Type of element (text_block, table, figure, handwriting)
        sequence: Sequential number
    
    Returns:
        Element ID like "E_0001", "T_0045", etc.
    """
    prefix_map = {
        "text_block": "E",
        "table": "E",
        "figure": "E",
        "handwriting": "E",
        "heading_h1": "H1",
        "heading_h2": "H2",
        "heading_h3": "H3",
    }
    prefix = prefix_map.get(element_type, "E")
    return f"{prefix}_{sequence:04d}"


# ============================================================================
# Bounding Box Utilities
# ============================================================================

@dataclass
class BBox:
    """Bounding box representation."""
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    def to_list(self) -> List[float]:
        """Convert to [x0, y0, x1, y1] format."""
        return [self.x0, self.y0, self.x1, self.y1]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}
    
    @classmethod
    def from_list(cls, coords: List[float]) -> 'BBox':
        """Create from [x0, y0, x1, y1] format."""
        return cls(coords[0], coords[1], coords[2], coords[3])
    
    def scale(self, scale_x: float, scale_y: float = None) -> 'BBox':
        """Scale bounding box coordinates."""
        if scale_y is None:
            scale_y = scale_x
        return BBox(
            self.x0 * scale_x,
            self.y0 * scale_y,
            self.x1 * scale_x,
            self.y1 * scale_y
        )
    
    def clip(self, width: float, height: float) -> 'BBox':
        """Clip bounding box to image dimensions."""
        return BBox(
            max(0, min(self.x0, width)),
            max(0, min(self.y0, height)),
            max(0, min(self.x1, width)),
            max(0, min(self.y1, height))
        )


def bbox_iou(box1: BBox, box2: BBox) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Calculate intersection
    x0 = max(box1.x0, box2.x0)
    y0 = max(box1.y0, box2.y0)
    x1 = min(box1.x1, box2.x1)
    y1 = min(box1.y1, box2.y1)
    
    if x1 < x0 or y1 < y0:
        return 0.0
    
    intersection = (x1 - x0) * (y1 - y0)
    union = box1.area + box2.area - intersection
    
    return intersection / union if union > 0 else 0.0


def bbox_overlap_ratio(box1: BBox, box2: BBox) -> float:
    """Calculate what fraction of box1 is covered by box2."""
    x0 = max(box1.x0, box2.x0)
    y0 = max(box1.y0, box2.y0)
    x1 = min(box1.x1, box2.x1)
    y1 = min(box1.y1, box2.y1)
    
    if x1 < x0 or y1 < y0:
        return 0.0
    
    intersection = (x1 - x0) * (y1 - y0)
    return intersection / box1.area if box1.area > 0 else 0.0


# ============================================================================
# Performance Timing
# ============================================================================

class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self, name: str = "Timer", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.logger:
            self.logger.info(f"{self.name}: {self.elapsed:.2f}s")
    
    def lap(self, label: str = ""):
        """Record a lap time."""
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        if self.logger:
            msg = f"{self.name}"
            if label:
                msg += f" - {label}"
            msg += f": {elapsed:.2f}s"
            self.logger.info(msg)


# ============================================================================
# File Utilities
# ============================================================================

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes."""
    return Path(path).stat().st_size / (1024 * 1024)

