"""
Tests for config module
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from pydantic import ValidationError

from ehrx.config import (
    DetectorConfig, OCRConfig, TablesConfig, HierarchyConfig, PrivacyConfig,
    EHRXConfig, load_config, find_default_config, load_default_config,
    validate_environment, setup_logging_from_config
)


class TestDetectorConfig:
    """Test DetectorConfig validation."""
    
    def test_default_detector_config(self):
        config = DetectorConfig()
        
        assert config.backend == "detectron2"
        assert config.model == "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
        assert config.min_conf == 0.5
        assert config.nms_iou == 0.3
        assert "Text" in config.label_map
        assert config.label_map["Text"] == "text_block"
    
    def test_detector_config_validation(self):
        # Valid backend
        config = DetectorConfig(backend="paddle")
        assert config.backend == "paddle"
        
        # Invalid backend
        with pytest.raises(ValidationError):
            DetectorConfig(backend="invalid")
        
        # Invalid confidence range
        with pytest.raises(ValidationError):
            DetectorConfig(min_conf=1.5)
        
        with pytest.raises(ValidationError):
            DetectorConfig(min_conf=-0.1)


class TestOCRConfig:
    """Test OCRConfig validation."""
    
    def test_default_ocr_config(self):
        config = OCRConfig()
        
        assert config.engine == "tesseract"
        assert config.psm_text == 6
        assert config.psm_table == 6
        assert config.lang == "eng"
        assert config.preprocess.deskew is True
        assert config.preprocess.binarize is True
    
    def test_ocr_config_validation(self):
        # Valid PSM values
        config = OCRConfig(psm_text=3, psm_table=8)
        assert config.psm_text == 3
        assert config.psm_table == 8
        
        # Invalid PSM values
        with pytest.raises(ValidationError):
            OCRConfig(psm_text=15)
        
        with pytest.raises(ValidationError):
            OCRConfig(psm_table=-1)
        
        # Invalid engine
        with pytest.raises(ValidationError):
            OCRConfig(engine="invalid")


class TestTablesConfig:
    """Test TablesConfig validation."""
    
    def test_default_tables_config(self):
        config = TablesConfig()
        
        assert config.csv_guess is True
        assert config.min_row_height_px == 10
        assert config.y_cluster_tol_px == 6
        assert config.x_proj_bins == 40
    
    def test_tables_config_validation(self):
        # Valid values
        config = TablesConfig(min_row_height_px=5, x_proj_bins=20)
        assert config.min_row_height_px == 5
        assert config.x_proj_bins == 20
        
        # Invalid values
        with pytest.raises(ValidationError):
            TablesConfig(min_row_height_px=0)
        
        with pytest.raises(ValidationError):
            TablesConfig(x_proj_bins=5)


class TestHierarchyConfig:
    """Test HierarchyConfig validation."""
    
    def test_default_hierarchy_config(self):
        config = HierarchyConfig()
        
        assert len(config.heading_regex) == 8
        assert any("PROBLEM" in regex for regex in config.heading_regex)
        assert any("MEDICATION" in regex for regex in config.heading_regex)
        assert config.caps_ratio_min == 0.6
        assert config.gap_above_px == 18
        assert config.left_margin_tolerance_px == 20
    
    def test_hierarchy_config_validation(self):
        # Valid values
        config = HierarchyConfig(caps_ratio_min=0.8, gap_above_px=25)
        assert config.caps_ratio_min == 0.8
        assert config.gap_above_px == 25
        
        # Invalid values
        with pytest.raises(ValidationError):
            HierarchyConfig(caps_ratio_min=1.5)
        
        with pytest.raises(ValidationError):
            HierarchyConfig(gap_above_px=-5)


class TestPrivacyConfig:
    """Test PrivacyConfig validation."""
    
    def test_default_privacy_config(self):
        config = PrivacyConfig()
        
        assert config.local_only is True
        assert config.log_text_snippets is False


class TestEHRXConfig:
    """Test main EHRXConfig model."""
    
    def test_default_config(self):
        config = EHRXConfig()
        
        assert isinstance(config.detector, DetectorConfig)
        assert isinstance(config.ocr, OCRConfig)
        assert isinstance(config.tables, TablesConfig)
        assert isinstance(config.hierarchy, HierarchyConfig)
        assert isinstance(config.privacy, PrivacyConfig)
    
    def test_config_from_dict(self):
        config_dict = {
            "detector": {"backend": "paddle", "min_conf": 0.7},
            "ocr": {"psm_text": 3},
            "privacy": {"log_text_snippets": True}
        }
        
        config = EHRXConfig(**config_dict)
        
        assert config.detector.backend == "paddle"
        assert config.detector.min_conf == 0.7
        assert config.ocr.psm_text == 3
        assert config.privacy.log_text_snippets is True
        
        # Check defaults are still applied
        assert config.ocr.engine == "tesseract"
        assert config.tables.csv_guess is True
    
    def test_config_extra_forbidden(self):
        # Should reject unknown fields
        with pytest.raises(ValidationError):
            EHRXConfig(unknown_field="value")


class TestConfigLoading:
    """Test configuration loading functions."""
    
    def test_load_config_default(self):
        # Load without any file should return default config
        config = load_config()
        
        assert isinstance(config, EHRXConfig)
        assert config.detector.backend == "detectron2"
    
    def test_load_config_from_file(self):
        config_data = {
            "detector": {"backend": "paddle", "min_conf": 0.8},
            "ocr": {"lang": "fra"},
            "privacy": {"log_text_snippets": True}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            assert config.detector.backend == "paddle"
            assert config.detector.min_conf == 0.8
            assert config.ocr.lang == "fra"
            assert config.privacy.log_text_snippets is True
            
            # Check defaults are still applied
            assert config.detector.nms_iou == 0.3
            assert config.tables.csv_guess is True
            
        finally:
            Path(config_path).unlink()
    
    def test_load_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
    
    def test_load_config_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:")
            config_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_load_config_invalid_values(self):
        config_data = {
            "detector": {"backend": "invalid_backend"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValidationError):
                load_config(config_path)
        finally:
            Path(config_path).unlink()


class TestConfigUtilities:
    """Test configuration utility functions."""
    
    def test_find_default_config(self):
        # This test depends on the actual file structure
        # In a real repo, there should be a configs/default.yaml
        default_path = find_default_config()
        
        # Should either find the default config or return None
        if default_path:
            assert default_path.exists()
            assert default_path.suffix == '.yaml'
    
    def test_load_default_config(self):
        # Should not raise an error even if no default config exists
        config = load_default_config()
        assert isinstance(config, EHRXConfig)
    
    def test_validate_environment(self):
        config = EHRXConfig()
        errors = validate_environment(config)
        
        # Should return a list (may be empty or contain error messages)
        assert isinstance(errors, list)
        
        # All errors should be strings
        for error in errors:
            assert isinstance(error, str)
    
    def test_setup_logging_from_config(self):
        config = EHRXConfig()
        config.privacy.log_text_snippets = True
        
        logger = setup_logging_from_config(config, level="DEBUG")
        
        assert logger.name == "ehrx"
        assert hasattr(logger, 'log_text_snippets')
        assert logger.log_text_snippets is True


class TestIntegration:
    """Integration tests for config system."""
    
    def test_full_config_workflow(self):
        # Create a complete config file
        config_data = {
            "detector": {
                "backend": "detectron2",
                "model": "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                "min_conf": 0.6,
                "label_map": {
                    "Text": "text_block",
                    "Table": "table",
                    "Figure": "figure"
                }
            },
            "ocr": {
                "engine": "tesseract",
                "psm_text": 6,
                "lang": "eng",
                "preprocess": {
                    "deskew": True,
                    "binarize": False
                }
            },
            "tables": {
                "csv_guess": False,
                "min_row_height_px": 15
            },
            "hierarchy": {
                "heading_regex": [
                    r"^CUSTOM PATTERN\b"
                ],
                "caps_ratio_min": 0.7
            },
            "privacy": {
                "local_only": True,
                "log_text_snippets": False
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load and validate
            config = load_config(config_path)
            
            # Check all values
            assert config.detector.backend == "detectron2"
            assert config.detector.min_conf == 0.6
            assert config.ocr.preprocess.binarize is False
            assert config.tables.csv_guess is False
            assert config.tables.min_row_height_px == 15
            assert len(config.hierarchy.heading_regex) == 1
            assert config.hierarchy.caps_ratio_min == 0.7
            assert config.privacy.local_only is True
            
            # Validate environment
            errors = validate_environment(config)
            assert isinstance(errors, list)
            
            # Setup logging
            logger = setup_logging_from_config(config)
            assert logger.name == "ehrx"
            
        finally:
            Path(config_path).unlink()