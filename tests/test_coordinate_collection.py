"""
Tests for coordinate collection from layout blocks
"""
import pytest
from typing import List, Dict, Any


def test_extract_left_edges_basic():
    """Test basic left-edge coordinate extraction."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Test with simple blocks
    blocks = [
        {"bbox_px": [10, 20, 50, 80]},    # Left edge at x=10
        {"bbox_px": [100, 30, 150, 90]},  # Left edge at x=100  
        {"bbox_px": [200, 40, 250, 100]}  # Left edge at x=200
    ]
    
    left_edges = detector.extract_left_edges(blocks)
    assert left_edges == [10.0, 100.0, 200.0]


def test_extract_left_edges_various_formats():
    """Test coordinate extraction handles various block formats."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Test different coordinate formats
    blocks = [
        # Standard format
        {"bbox_px": [10, 20, 50, 80]},
        
        # Float coordinates
        {"bbox_px": [100.5, 30.2, 150.8, 90.1]},
        
        # String coordinates (should convert)
        {"bbox_px": ["200", "40", "250", "100"]},
        
        # Nested in different structure (layoutparser format)
        {"block": {"x_1": 300, "y_1": 50, "x_2": 350, "y_2": 110}}
    ]
    
    left_edges = detector.extract_left_edges(blocks)
    expected = [10.0, 100.5, 200.0, 300.0]
    assert left_edges == expected


def test_extract_left_edges_robustness():
    """Test coordinate extraction handles malformed data gracefully."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Test with malformed blocks
    malformed_blocks = [
        {"bbox_px": [10, 20, 50, 80]},    # Good block
        {"bbox_px": []},                  # Empty bbox
        {"bbox_px": [100]},               # Incomplete bbox
        {"some_other_field": "value"},    # No bbox at all
        {"bbox_px": ["invalid", "data"]}, # Non-numeric data
        {"block": {}},                    # Empty block structure
        {"bbox_px": [200, 40, 250, 100]}  # Another good block
    ]
    
    left_edges = detector.extract_left_edges(malformed_blocks)
    # Should only extract from valid blocks
    assert left_edges == [10.0, 200.0]


def test_extract_left_edges_empty_input():
    """Test coordinate extraction with empty input."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Test with empty list
    left_edges = detector.extract_left_edges([])
    assert left_edges == []
    
    # Test with list of invalid blocks
    invalid_blocks = [
        {"some_field": "value"},
        {"another_field": 123}
    ]
    left_edges = detector.extract_left_edges(invalid_blocks)
    assert left_edges == []


def test_extract_left_edges_duplicate_coordinates():
    """Test coordinate extraction handles duplicates."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Blocks with duplicate left edges
    blocks = [
        {"bbox_px": [10, 20, 50, 80]},    # Left edge at x=10
        {"bbox_px": [10, 100, 50, 140]},  # Duplicate x=10
        {"bbox_px": [100, 30, 150, 90]},  # Left edge at x=100
        {"bbox_px": [10, 200, 50, 240]}   # Another x=10
    ]
    
    left_edges = detector.extract_left_edges(blocks)
    # Should preserve duplicates for clustering algorithm
    assert left_edges == [10.0, 10.0, 100.0, 10.0]


def test_extract_left_edges_sorting_independence():
    """Test that coordinate extraction doesn't assume input ordering."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Blocks in non-sorted order
    unsorted_blocks = [
        {"bbox_px": [200, 40, 250, 100]},  # x=200
        {"bbox_px": [10, 20, 50, 80]},     # x=10  
        {"bbox_px": [100, 30, 150, 90]}    # x=100
    ]
    
    left_edges = detector.extract_left_edges(unsorted_blocks)
    # Should preserve input order (don't sort yet)
    assert left_edges == [200.0, 10.0, 100.0]


def test_filter_coordinate_outliers():
    """Test outlier filtering for noisy coordinate data."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Coordinates with outliers
    coordinates = [10.0, 12.0, 11.0, 100.0, 101.0, 99.0, 500.0, 9.0, 102.0]
    
    # Test basic outlier filtering (if implemented)
    filtered = detector.filter_coordinate_outliers(coordinates)
    
    # Should remove the extreme outlier (500.0) but keep reasonable values
    assert 500.0 not in filtered
    assert all(coord in coordinates for coord in filtered)
    
    # Test with no outliers
    clean_coordinates = [10.0, 11.0, 12.0, 13.0]
    filtered_clean = detector.filter_coordinate_outliers(clean_coordinates)
    assert filtered_clean == clean_coordinates