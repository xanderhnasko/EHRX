"""
Load and validate YAML configuration
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import logging


# DetectorConfig removed - LayoutParser dependency eliminated


class OCRPreprocessConfig(BaseModel):
    """OCR preprocessing configuration."""
    deskew: bool = True
    binarize: bool = True


class OCRConfig(BaseModel):
    """Configuration for OCR processing."""
    engine: str = Field(default="tesseract")
    psm_text: int = Field(default=6, description="Page segmentation mode for text")
    psm_table: int = Field(default=6, description="Page segmentation mode for tables") 
    lang: str = Field(default="eng")
    preprocess: OCRPreprocessConfig = Field(default_factory=OCRPreprocessConfig)
    
    @field_validator('engine')
    @classmethod
    def validate_engine(cls, v):
        if v != 'tesseract':
            raise ValueError('Only tesseract engine is supported')
        return v
    
    @field_validator('psm_text', 'psm_table')
    @classmethod
    def validate_psm(cls, v):
        if not 0 <= v <= 13:
            raise ValueError('PSM must be between 0 and 13')
        return v


class TablesConfig(BaseModel):
    """Configuration for table processing."""
    csv_guess: bool = Field(default=True, description="Use heuristic row/col if grid not found")
    min_row_height_px: int = Field(default=10, ge=1)
    y_cluster_tol_px: int = Field(default=6, ge=1)
    x_proj_bins: int = Field(default=40, ge=10)


class HierarchyLevelsConfig(BaseModel):
    """Configuration for hierarchy levels."""
    H1: str = Field(default="strong_keyword|big_gap")
    H2: str = Field(default="weaker_keyword|indented") 
    H3: str = Field(default="bullets|numbered")


class HierarchyConfig(BaseModel):
    """Configuration for hierarchy building."""
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
    caps_ratio_min: float = Field(default=0.6, ge=0.0, le=1.0)
    gap_above_px: int = Field(default=18, ge=0)
    left_margin_tolerance_px: int = Field(default=20, ge=0)
    levels: HierarchyLevelsConfig = Field(default_factory=HierarchyLevelsConfig)


class PrivacyConfig(BaseModel):
    """Configuration for privacy settings."""
    local_only: bool = Field(default=True)
    log_text_snippets: bool = Field(default=False, description="Never log OCR text")


class EHRXConfig(BaseModel):
    """Main configuration model."""
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    tables: TablesConfig = Field(default_factory=TablesConfig) 
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    
    model_config = ConfigDict(extra="forbid")  # Prevent unknown fields
    
    @classmethod
    def from_yaml(cls, config_path: Optional[Union[str, Path]] = None) -> "EHRXConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to config YAML file. If None, uses default config.
            
        Returns:
            Validated configuration object.
        """
        return load_config(config_path)


def load_config(config_path: Optional[Union[str, Path]] = None) -> EHRXConfig:
    """Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file. If None, uses default config.
        
    Returns:
        Validated configuration object.
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValidationError: If config values are invalid
    """
    # Start with default config
    config_data = {}
    
    # Load from file if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    
    # Validate and return
    return EHRXConfig(**config_data)


def find_default_config() -> Optional[Path]:
    """Find default config file in standard locations."""
    # Look for config in standard locations
    search_paths = [
        Path.cwd() / "configs" / "default.yaml",
        Path.cwd() / "config.yaml",
        Path.cwd() / "ehrx.yaml",
        Path(__file__).parent.parent / "configs" / "default.yaml"
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def load_default_config() -> EHRXConfig:
    """Load configuration from default locations."""
    default_path = find_default_config()
    return load_config(default_path)


def validate_environment(config: EHRXConfig) -> List[str]:
    """Validate that required system dependencies are available.
    
    Returns:
        List of error messages. Empty if all dependencies are available.
    """
    errors = []
    
    # Check tesseract
    if config.ocr.engine == "tesseract":
        try:
            import pytesseract
            # Try to get tesseract version to ensure it's installed
            pytesseract.get_tesseract_version()
        except Exception as e:
            errors.append(f"Tesseract not available: {e}")
    
    # Check PDF processing
    try:
        import fitz  # PyMuPDF
    except ImportError:
        try:
            import pdf2image
        except ImportError:
            errors.append("Neither PyMuPDF nor pdf2image available for PDF processing")
    
    # Check OpenCV
    try:
        import cv2
    except ImportError:
        errors.append("OpenCV not available")
    
    return errors


def setup_logging_from_config(config: EHRXConfig, level: str = "INFO") -> logging.Logger:
    """Setup logging based on configuration."""
    from .utils import setup_logging
    
    return setup_logging(
        level=level,
        log_text_snippets=config.privacy.log_text_snippets
    )

