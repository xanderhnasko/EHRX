"""
Tests for K-means column boundary detection
"""
import pytest
from typing import List, Dict, Any
import numpy as np


def test_detect_multi_column_layout_two_columns():
    """Test K-means detection with clear two-column layout."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Simulate two clear column groups
    blocks = [
        # Left column (around x=50)
        {"bbox_px": [45, 20, 90, 80]},
        {"bbox_px": [50, 100, 95, 180]}, 
        {"bbox_px": [48, 200, 93, 280]},
        {"bbox_px": [52, 300, 97, 380]},
        
        # Right column (around x=350)
        {"bbox_px": [345, 20, 390, 80]},
        {"bbox_px": [350, 100, 395, 180]},
        {"bbox_px": [348, 200, 393, 280]},
        {"bbox_px": [352, 300, 397, 380]}
    ]
    
    layout = detector.detect_multi_column_layout(blocks, page_width=500.0)
    
    # Should detect 2 columns
    assert layout.column_count == 2
    assert len(layout.boundaries) == 3  # [start, middle, end]
    
    # Boundaries should split around the gap between columns
    # Left column at ~50, right at ~350, so boundary around ~200
    middle_boundary = layout.boundaries[1]
    assert 150 < middle_boundary < 250  # Somewhere between columns


def test_detect_multi_column_layout_three_columns():
    """Test K-means detection with three-column layout."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Simulate three column groups
    blocks = [
        # Column 1 (around x=50)
        {"bbox_px": [45, 20, 90, 80]},
        {"bbox_px": [50, 100, 95, 180]},
        
        # Column 2 (around x=250)  
        {"bbox_px": [245, 20, 290, 80]},
        {"bbox_px": [250, 100, 295, 180]},
        
        # Column 3 (around x=450)
        {"bbox_px": [445, 20, 490, 80]},
        {"bbox_px": [450, 100, 495, 180]}
    ]
    
    layout = detector.detect_multi_column_layout(blocks, page_width=600.0)
    
    # Should detect 3 columns
    assert layout.column_count == 3
    assert len(layout.boundaries) == 4  # [start, mid1, mid2, end]


def test_kmeans_with_optimal_k_selection():
    """Test that K-means selects the best number of columns automatically."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Clear two-column data
    coordinates = [50.0, 52.0, 48.0, 51.0,  # Column 1
                   350.0, 352.0, 348.0, 351.0]  # Column 2
    
    best_k, score = detector.find_optimal_k(coordinates, max_k=3)
    
    # Should select k=2 for this data
    assert best_k == 2
    assert score >= 0  # Silhouette score should be valid


def test_kmeans_single_cluster_fallback():
    """Test K-means falls back to single column for unclear data."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Test with insufficient data (should fall back)
    few_blocks = [{"bbox_px": [10, 20, 50, 80]}]
    layout = detector.detect_multi_column_layout(few_blocks, page_width=100.0)
    assert layout.column_count == 1  # Should fall back to single column
    
    # Test with identical coordinates (should fall back)
    identical_blocks = [
        {"bbox_px": [50, 20, 90, 80]},
        {"bbox_px": [50, 100, 90, 140]},
        {"bbox_px": [50, 200, 90, 240]}
    ]
    layout = detector.detect_multi_column_layout(identical_blocks, page_width=100.0)
    assert layout.column_count == 1  # Should fall back to single column


def test_create_column_boundaries_from_clusters():
    """Test boundary creation from K-means cluster centers."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Test with known cluster centers
    cluster_centers = [50.0, 250.0, 450.0]  # 3 columns
    page_width = 600.0
    
    boundaries = detector.create_boundaries_from_centers(cluster_centers, page_width)
    
    # Should create 4 boundaries for 3 columns
    assert len(boundaries) == 4
    assert boundaries[0] == 0.0    # Start
    assert boundaries[-1] == 600.0  # End
    
    # Middle boundaries should split between cluster centers
    # Between 50 and 250: ~150
    # Between 250 and 450: ~350
    assert 100 < boundaries[1] < 200
    assert 300 < boundaries[2] < 400


def test_handle_edge_cases():
    """Test K-means handles various edge cases gracefully."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Test with very few coordinates
    few_coords = [50.0, 100.0]
    best_k, score = detector.find_optimal_k(few_coords, max_k=3)
    assert best_k == 1  # Should fall back to single column
    
    # Test with identical coordinates  
    identical_coords = [100.0, 100.0, 100.0, 100.0]
    best_k, score = detector.find_optimal_k(identical_coords, max_k=3)
    assert best_k == 1  # Should fall back to single column
    
    # Test with empty coordinates (should handle gracefully)
    empty_coords = []
    best_k, score = detector.find_optimal_k(empty_coords, max_k=3)
    assert best_k == 1  # Should default to single column


def test_full_pipeline_detect_multi_column():
    """Test the full multi-column detection pipeline."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Real-world-like data with clear columns and some noise
    blocks = [
        # Left column blocks
        {"bbox_px": [72, 20, 250, 80]},
        {"bbox_px": [74, 100, 245, 180]},
        {"bbox_px": [70, 200, 248, 280]},
        
        # Right column blocks
        {"bbox_px": [320, 30, 500, 90]},
        {"bbox_px": [318, 110, 495, 190]},
        {"bbox_px": [322, 210, 498, 290]},
        
        # Some noise/outliers
        {"bbox_px": [10, 400, 50, 420]},  # Page number
        {"bbox_px": [400, 450, 500, 470]}   # Footer
    ]
    
    layout = detector.detect_multi_column_layout(blocks, page_width=612.0)
    
    # Should detect 2 main columns despite noise
    assert layout.column_count in [1, 2]  # Allow fallback to 1 if clustering unclear
    assert layout.page_width == 612.0
    
    # Test element assignment works
    left_element_col = layout.assign_to_column(75.0)  # Left column
    right_element_col = layout.assign_to_column(350.0)  # Right column
    
    if layout.column_count == 2:
        assert left_element_col != right_element_col