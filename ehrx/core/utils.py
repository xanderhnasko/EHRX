"""
Utilities: bbox utils, id generation, logging, timers
"""
import hashlib
import logging
import time
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class BBox:
    """Bounding box utilities for coordinate transformations."""
    
    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
    
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
        """Convert to dict format."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}
    
    @classmethod
    def from_list(cls, coords: List[float]) -> "BBox":
        """Create BBox from [x0, y0, x1, y1] list."""
        return cls(coords[0], coords[1], coords[2], coords[3])
    
    def scale(self, scale_x: float, scale_y: float) -> "BBox":
        """Scale bbox coordinates."""
        return BBox(
            self.x0 * scale_x,
            self.y0 * scale_y,
            self.x1 * scale_x,
            self.y1 * scale_y
        )
    
    def intersects(self, other: "BBox") -> bool:
        """Check if this bbox intersects with another."""
        return not (self.x1 < other.x0 or other.x1 < self.x0 or 
                   self.y1 < other.y0 or other.y1 < self.y0)
    
    def intersection(self, other: "BBox") -> Optional["BBox"]:
        """Get intersection bbox if it exists."""
        if not self.intersects(other):
            return None
        
        return BBox(
            max(self.x0, other.x0),
            max(self.y0, other.y0),
            min(self.x1, other.x1),
            min(self.y1, other.y1)
        )
    
    def iou(self, other: "BBox") -> float:
        """Calculate Intersection over Union."""
        intersection = self.intersection(other)
        if not intersection:
            return 0.0
        
        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0


def pdf_to_pixel_coords(bbox_pdf: List[float], page_height_pdf: float, 
                       page_height_px: int, scale: float) -> List[int]:
    """Convert PDF coordinates to pixel coordinates.
    
    PDF coordinates have origin at bottom-left, pixels at top-left.
    """
    x0_pdf, y0_pdf, x1_pdf, y1_pdf = bbox_pdf
    
    # Convert to pixel coordinates with origin flip
    x0_px = int(x0_pdf * scale)
    y0_px = int((page_height_pdf - y1_pdf) * scale)  # Flip Y and use y1 for top
    x1_px = int(x1_pdf * scale)
    y1_px = int((page_height_pdf - y0_pdf) * scale)  # Flip Y and use y0 for bottom
    
    return [x0_px, y0_px, x1_px, y1_px]


def pixel_to_pdf_coords(bbox_px: List[int], page_height_pdf: float, 
                       page_height_px: int, scale: float) -> List[float]:
    """Convert pixel coordinates to PDF coordinates."""
    x0_px, y0_px, x1_px, y1_px = bbox_px
    
    # Convert to PDF coordinates with origin flip
    x0_pdf = x0_px / scale
    y1_pdf = page_height_pdf - (y0_px / scale)  # Flip Y and use y0 for top
    x1_pdf = x1_px / scale
    y0_pdf = page_height_pdf - (y1_px / scale)  # Flip Y and use y1 for bottom
    
    return [x0_pdf, y0_pdf, x1_pdf, y1_pdf]


class IDGenerator:
    """Generate deterministic element IDs."""
    
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.counters = {"element": 0, "heading": 0}
    
    def next_element_id(self) -> str:
        """Generate next element ID: E_XXXX format."""
        self.counters["element"] += 1
        return f"E_{self.counters['element']:04d}"
    
    def next_heading_id(self, level: str) -> str:
        """Generate next heading ID: H1_XXXX, H2_XXXX, etc."""
        self.counters["heading"] += 1
        return f"{level}_{self.counters['heading']:04d}"
    
    def reset(self):
        """Reset all counters."""
        self.counters = {"element": 0, "heading": 0}


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger.info(f"{self.name}: {duration:.2f}s")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


def setup_logging(level: str = "INFO", log_text_snippets: bool = False) -> logging.Logger:
    """Setup PHI-safe logging configuration."""
    
    # Create logger
    logger = logging.getLogger("ehrx")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Store PHI settings
    logger.log_text_snippets = log_text_snippets
    
    return logger


def safe_log_text(logger: logging.Logger, text: str, max_length: int = 50) -> str:
    """Safely log text snippets respecting PHI settings."""
    if not getattr(logger, 'log_text_snippets', False):
        return f"<text:{len(text)} chars>"
    
    if len(text) <= max_length:
        return f'"{text}"'
    else:
        return f'"{text[:max_length]}..."'


def create_manifest(doc_id: str, input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create run manifest with metadata."""
    return {
        "doc_id": doc_id,
        "input_path": str(input_path),
        "created_at": datetime.now().isoformat(),
        "detector": config.get("detector", {}).get("backend", "unknown"),
        "ocr": config.get("ocr", {}).get("engine", "unknown"),
        "config_hash": hashlib.md5(str(config).encode()).hexdigest()[:8]
    }


def ensure_output_dir(output_path: str) -> Path:
    """Ensure output directory exists and return Path object."""
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Create assets subdirectory
    (path / "assets").mkdir(exist_ok=True)
    
    return path


def validate_pdf_path(pdf_path: str) -> Path:
    """Validate PDF input path."""
    path = Path(pdf_path)
    
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if path.suffix.lower() != '.pdf':
        raise ValueError(f"Input file must be a PDF: {pdf_path}")
    
    return path
