"""
Tests for single-column detection (baseline/fallback)
"""
import pytest
from typing import List, Dict, Any


def test_single_column_detection_basic():
    """Test basic single-column detection with various inputs."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Test with empty blocks
    layout = detector.detect_single_column_layout([], page_width=100.0)
    assert layout.column_count == 1
    assert layout.boundaries == [0.0, 100.0]
    assert layout.page_width == 100.0
    
    # Test with single block
    blocks = [{"bbox_px": [10, 20, 50, 80]}]
    layout = detector.detect_single_column_layout(blocks, page_width=100.0)
    assert layout.column_count == 1
    assert layout.boundaries == [0.0, 100.0]


def test_single_column_detection_with_various_widths():
    """Test single-column detection respects different page widths."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Test different page widths
    for width in [50.0, 100.0, 200.0, 612.0]:  # 612 is typical PDF width
        layout = detector.detect_single_column_layout([], page_width=width)
        assert layout.column_count == 1
        assert layout.boundaries == [0.0, width]
        assert layout.page_width == width


def test_single_column_detection_robustness():
    """Test single-column detection handles edge cases gracefully."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Test with malformed blocks (missing bbox_px)
    malformed_blocks = [{"some_other_field": "value"}]
    layout = detector.detect_single_column_layout(malformed_blocks, page_width=100.0)
    assert layout.column_count == 1
    assert layout.boundaries == [0.0, 100.0]
    
    # Test with blocks containing invalid coordinates
    invalid_blocks = [{"bbox_px": []}]  # Empty bbox
    layout = detector.detect_single_column_layout(invalid_blocks, page_width=100.0)
    assert layout.column_count == 1
    assert layout.boundaries == [0.0, 100.0]
    
    # Test with negative page width (should handle gracefully)
    with pytest.raises(ValueError, match="page_width must be positive"):
        detector.detect_single_column_layout([], page_width=-10.0)


def test_single_column_detection_ignores_block_positions():
    """Test that single-column detection ignores actual block positions."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Blocks spread across what might be multiple columns
    # But single-column detector should ignore this
    wide_spread_blocks = [
        {"bbox_px": [10, 20, 30, 80]},    # Left side
        {"bbox_px": [200, 20, 220, 80]},  # Right side
        {"bbox_px": [100, 100, 120, 120]} # Middle
    ]
    
    layout = detector.detect_single_column_layout(wide_spread_blocks, page_width=250.0)
    assert layout.column_count == 1
    assert layout.boundaries == [0.0, 250.0]
    assert layout.page_width == 250.0


def test_document_column_detector_initialization():
    """Test DocumentColumnDetector can be initialized and configured."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    # Test default initialization
    detector = DocumentColumnDetector()
    assert detector is not None
    
    # Test with configuration (will be added later)
    # For now, just ensure it accepts configs without failing
    config = {"min_column_width": 50.0, "gap_threshold": 10.0}
    detector_with_config = DocumentColumnDetector(config)
    assert detector_with_config is not None