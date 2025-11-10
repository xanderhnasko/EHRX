"""
Tests for ColumnLayout data structure
"""
import pytest
from typing import List, Dict, Any

def test_column_layout_creation():
    """Test basic ColumnLayout creation and properties."""
    # Import will fail initially - this is the RED phase
    from ehrx.layout.column_detection import ColumnLayout
    
    # Test creating a single column layout
    single_col = ColumnLayout(
        column_count=1,
        boundaries=[0.0, 100.0],
        page_width=100.0
    )
    
    assert single_col.column_count == 1
    assert single_col.boundaries == [0.0, 100.0]
    assert single_col.page_width == 100.0
    
    # Test creating a multi-column layout
    multi_col = ColumnLayout(
        column_count=2,
        boundaries=[0.0, 50.0, 100.0],
        page_width=100.0
    )
    
    assert multi_col.column_count == 2
    assert multi_col.boundaries == [0.0, 50.0, 100.0]
    assert len(multi_col.boundaries) == multi_col.column_count + 1


def test_column_layout_validation():
    """Test ColumnLayout validation logic."""
    from ehrx.layout.column_detection import ColumnLayout
    
    # Test invalid column count vs boundaries
    with pytest.raises(ValueError, match="boundaries must have column_count \\+ 1 elements"):
        ColumnLayout(
            column_count=2,
            boundaries=[0.0, 100.0],  # Only 2 boundaries for 2 columns
            page_width=100.0
        )
    
    # Test non-sorted boundaries
    with pytest.raises(ValueError, match="boundaries must be in ascending order"):
        ColumnLayout(
            column_count=2,
            boundaries=[0.0, 100.0, 50.0],
            page_width=100.0
        )


def test_column_assignment():
    """Test assigning x-coordinates to columns."""
    from ehrx.layout.column_detection import ColumnLayout
    
    layout = ColumnLayout(
        column_count=3,
        boundaries=[0.0, 33.0, 67.0, 100.0],
        page_width=100.0
    )
    
    # Test coordinate assignment
    assert layout.assign_to_column(10.0) == 0  # First column
    assert layout.assign_to_column(45.0) == 1  # Second column  
    assert layout.assign_to_column(80.0) == 2  # Third column
    
    # Test edge cases
    assert layout.assign_to_column(0.0) == 0   # Left edge
    assert layout.assign_to_column(33.0) == 1  # Boundary point
    assert layout.assign_to_column(100.0) == 2 # Right edge
    
    # Test out-of-bounds (should clamp)
    assert layout.assign_to_column(-10.0) == 0  # Below minimum
    assert layout.assign_to_column(110.0) == 2  # Above maximum


def test_column_layout_serialization():
    """Test ColumnLayout can be serialized to/from dict."""
    from ehrx.layout.column_detection import ColumnLayout
    
    original = ColumnLayout(
        column_count=2,
        boundaries=[0.0, 50.0, 100.0],
        page_width=100.0
    )
    
    # Test to_dict
    data = original.to_dict()
    expected_keys = {"column_count", "boundaries", "page_width"}
    assert set(data.keys()) == expected_keys
    
    # Test from_dict
    restored = ColumnLayout.from_dict(data)
    assert restored.column_count == original.column_count
    assert restored.boundaries == original.boundaries
    assert restored.page_width == original.page_width


def test_column_layout_equality():
    """Test ColumnLayout equality comparison."""
    from ehrx.layout.column_detection import ColumnLayout
    
    layout1 = ColumnLayout(
        column_count=2,
        boundaries=[0.0, 50.0, 100.0],
        page_width=100.0
    )
    
    layout2 = ColumnLayout(
        column_count=2,
        boundaries=[0.0, 50.0, 100.0],
        page_width=100.0
    )
    
    layout3 = ColumnLayout(
        column_count=3,
        boundaries=[0.0, 33.0, 67.0, 100.0],
        page_width=100.0
    )
    
    assert layout1 == layout2
    assert layout1 != layout3