"""
Load and validate YAML configuration
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from pydantic import BaseModel, Field, validator


# ============================================================================
# Configuration Models (using Pydantic for validation)
# ============================================================================

class DetectorConfig(BaseModel):
    """Configuration for layout detector."""
    backend: str = "detectron2"  # or "paddle"
    model: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    model_path: Optional[str] = None  # Local path to model weights
    label_map: Dict[str, str] = Field(default_factory=lambda: {
        "Text": "text_block",
        "Table": "table",
        "Figure": "figure"
    })
    min_conf: float = 0.5
    nms_iou: float = 0.3


class OCRConfig(BaseModel):
    """Configuration for OCR."""
    engine: str = "tesseract"
    psm_text: int = 6  # uniform block
    psm_table: int = 6
    lang: str = "eng"
    preprocess: Dict[str, bool] = Field(default_factory=lambda: {
        "deskew": True,
        "binarize": True
    })


class TablesConfig(BaseModel):
    """Configuration for table extraction."""
    csv_guess: bool = True
    min_row_height_px: int = 10
    y_cluster_tol_px: int = 6
    x_proj_bins: int = 40


class HierarchyConfig(BaseModel):
    """Configuration for hierarchy detection."""
    heading_regex: List[str] = Field(default_factory=lambda: [
        r"^(PROBLEMS?|PROBLEM LIST)\b",
        r"^(MEDICATIONS?|CURRENT MEDS?)\b",
        r"^(ALLERG(IES|Y))\b",
        r"^(LABS?|RESULTS?)\b",
        r"^(VITALS?)\b",
        r"^(IMAGING|RADIOLOGY)\b",
        r"^(ASSESSMENT|PLAN|A/P)\b",
        r"^NOTES?\b"
    ])
    caps_ratio_min: float = 0.6
    gap_above_px: int = 18
    left_margin_tolerance_px: int = 20
    levels: Dict[str, str] = Field(default_factory=lambda: {
        "H1": "strong_keyword|big_gap",
        "H2": "weaker_keyword|indented",
        "H3": "bullets|numbered"
    })


class PrivacyConfig(BaseModel):
    """Configuration for privacy settings."""
    local_only: bool = True
    log_text_snippets: bool = False


class EHRXConfig(BaseModel):
    """Main configuration for ehrx."""
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    tables: TablesConfig = Field(default_factory=TablesConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: Optional[Path] = None) -> EHRXConfig:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses defaults.
    
    Returns:
        Validated EHRXConfig object
    """
    if config_path is None:
        # Use default config
        return EHRXConfig()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Merge with defaults
    return EHRXConfig(**yaml_data)


def load_config_with_overrides(
    config_path: Optional[Path] = None,
    **overrides
) -> EHRXConfig:
    """
    Load config and apply command-line overrides.
    
    Args:
        config_path: Path to YAML config file
        **overrides: Key-value pairs to override config values
    
    Returns:
        Validated EHRXConfig with overrides applied
    """
    config = load_config(config_path)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def get_default_config_path() -> Path:
    """Get path to default config file."""
    return Path(__file__).parent.parent / "configs" / "default.yaml"

