"""
Tests for GlobalOrderingManager - focused on major functionality
"""
import pytest
from typing import List, Dict, Any


def test_global_z_order_counter():
    """Test global z-order counter persists across pages."""
    from ehrx.layout.global_ordering import GlobalOrderingManager
    from ehrx.layout.column_detection import ColumnLayout
    
    # Single column layout for simplicity
    layout = ColumnLayout(
        column_count=1,
        boundaries=[0.0, 100.0],
        page_width=100.0
    )
    
    manager = GlobalOrderingManager(layout)
    
    # Test initial state
    assert manager.get_next_z_order() == 0
    assert manager.get_next_z_order() == 1
    assert manager.get_next_z_order() == 2
    
    # Test counter persistence (doesn't reset)
    current_count = manager.get_current_z_order()
    assert current_count == 2  # Last assigned was 2
    
    # Continue counting
    assert manager.get_next_z_order() == 3
    assert manager.get_next_z_order() == 4


def test_column_aware_element_sorting():
    """Test elements are sorted by (column_index, y_coordinate)."""
    from ehrx.layout.global_ordering import GlobalOrderingManager
    from ehrx.layout.column_detection import ColumnLayout
    
    # Two-column layout
    layout = ColumnLayout(
        column_count=2,
        boundaries=[0.0, 50.0, 100.0],
        page_width=100.0
    )
    
    manager = GlobalOrderingManager(layout)
    
    # Elements in mixed order (should be sorted by column, then Y)
    elements = [
        {"bbox_px": [60, 100, 90, 120], "id": "right_bottom"},  # Col 1, Y=100
        {"bbox_px": [10, 50, 40, 70], "id": "left_middle"},     # Col 0, Y=50  
        {"bbox_px": [60, 20, 90, 40], "id": "right_top"},       # Col 1, Y=20
        {"bbox_px": [10, 10, 40, 30], "id": "left_top"}         # Col 0, Y=10
    ]
    
    sorted_elements = manager.sort_elements_reading_order(elements)
    
    # Expected order: left column first (by Y), then right column (by Y)
    expected_ids = ["left_top", "left_middle", "right_top", "right_bottom"]
    actual_ids = [elem["id"] for elem in sorted_elements]
    
    assert actual_ids == expected_ids


def test_heading_context_tracking():
    """Test heading context is tracked per column."""
    from ehrx.layout.global_ordering import GlobalOrderingManager
    from ehrx.layout.column_detection import ColumnLayout
    
    # Two-column layout
    layout = ColumnLayout(
        column_count=2,
        boundaries=[0.0, 50.0, 100.0],
        page_width=100.0
    )
    
    manager = GlobalOrderingManager(layout)
    
    # Add headings in different columns
    heading_left = {
        "id": "H1_001",
        "type": "text_block",
        "bbox_px": [10, 20, 40, 40],
        "payload": {"text": "MEDICATIONS"}
    }
    
    heading_right = {
        "id": "H2_001", 
        "type": "text_block",
        "bbox_px": [60, 20, 90, 40],
        "payload": {"text": "ALLERGIES"}
    }
    
    # Track headings
    manager.track_heading_context(heading_left)
    manager.track_heading_context(heading_right)
    
    # Test finding associated headings per column
    left_table = {"bbox_px": [10, 100, 40, 120]}  # Column 0
    right_figure = {"bbox_px": [60, 100, 90, 120]}  # Column 1
    
    left_heading = manager.find_associated_heading(left_table)
    right_heading = manager.find_associated_heading(right_figure)
    
    assert left_heading == "H1_001"  # Should associate with MEDICATIONS
    assert right_heading == "H2_001"  # Should associate with ALLERGIES


def test_cross_page_state_management():
    """Test state persists across page processing."""
    from ehrx.layout.global_ordering import GlobalOrderingManager  
    from ehrx.layout.column_detection import ColumnLayout
    
    layout = ColumnLayout(
        column_count=1,
        boundaries=[0.0, 100.0],
        page_width=100.0
    )
    
    manager = GlobalOrderingManager(layout)
    
    # Process elements from page 1
    page1_elements = [
        {"id": "E_001", "page": 1, "bbox_px": [10, 20, 40, 40]},
        {"id": "E_002", "page": 1, "bbox_px": [10, 60, 40, 80]}
    ]
    
    # Assign z-orders
    for element in page1_elements:
        element["z_order"] = manager.get_next_z_order()
    
    # Process elements from page 2 (z-order should continue)
    page2_elements = [
        {"id": "E_003", "page": 2, "bbox_px": [10, 20, 40, 40]},
        {"id": "E_004", "page": 2, "bbox_px": [10, 60, 40, 80]}
    ]
    
    # Assign z-orders (should continue from page 1)
    for element in page2_elements:
        element["z_order"] = manager.get_next_z_order()
    
    # Verify continuous ordering across pages
    assert page1_elements[0]["z_order"] == 0
    assert page1_elements[1]["z_order"] == 1  
    assert page2_elements[0]["z_order"] == 2  # Continues from page 1
    assert page2_elements[1]["z_order"] == 3


def test_element_column_assignment():
    """Test elements are correctly assigned to columns."""
    from ehrx.layout.global_ordering import GlobalOrderingManager
    from ehrx.layout.column_detection import ColumnLayout
    
    # Three-column layout
    layout = ColumnLayout(
        column_count=3,
        boundaries=[0.0, 33.0, 67.0, 100.0],
        page_width=100.0
    )
    
    manager = GlobalOrderingManager(layout)
    
    # Elements in different columns
    elements = [
        {"bbox_px": [10, 20, 30, 40]},   # Column 0 (left edge at 10)
        {"bbox_px": [45, 20, 65, 40]},   # Column 1 (left edge at 45)  
        {"bbox_px": [80, 20, 95, 40]}    # Column 2 (left edge at 80)
    ]
    
    # Test column assignment
    columns = [manager.assign_element_to_column(elem) for elem in elements]
    assert columns == [0, 1, 2]