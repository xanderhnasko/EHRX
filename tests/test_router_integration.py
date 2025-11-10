"""
Integration tests for ElementRouter with column detection and global ordering
"""
import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock
import numpy as np


def test_enhanced_router_with_column_detection():
    """Test ElementRouter integration with column detection and global ordering."""
    from ehrx.layout.column_detection import DocumentColumnDetector, ColumnLayout
    from ehrx.layout.global_ordering import GlobalOrderingManager
    from ehrx.layout.enhanced_router import EnhancedElementRouter
    
    # Mock configuration and dependencies
    mock_config = Mock()
    mock_config.ocr = Mock()
    
    # Create two-column layout
    layout = ColumnLayout(
        column_count=2,
        boundaries=[0.0, 300.0, 600.0],
        page_width=600.0
    )
    
    # Initialize enhanced router
    router = EnhancedElementRouter(
        config=mock_config,
        doc_id="test_doc",
        column_layout=layout
    )
    
    # Mock layout blocks (simulating LayoutParser output)
    mock_blocks = [
        Mock(block=Mock(x_1=50, y_1=100, x_2=200, y_2=150), type=1, score=0.9),   # Left text
        Mock(block=Mock(x_1=350, y_1=100, x_2=500, y_2=150), type=1, score=0.8),  # Right text  
        Mock(block=Mock(x_1=50, y_1=200, x_2=200, y_2=300), type=4, score=0.85),  # Left table
        Mock(block=Mock(x_1=350, y_1=50, x_2=500, y_2=90), type=1, score=0.9)     # Right heading
    ]
    
    # Mock page info and coordinate mapper
    mock_page_info = Mock(page_number=1)
    mock_mapper = Mock()
    mock_mapper.pixel_to_pdf.return_value = [0, 0, 100, 100]  # Simplified
    
    # Create test page image
    mock_page_image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Process blocks
    elements = router.process_layout_blocks_with_global_ordering(
        mock_blocks, mock_page_image, mock_page_info, mock_mapper
    )
    
    # Verify elements are processed and ordered correctly
    assert len(elements) == 4
    
    # Check global z-order assignment (should be continuous)
    z_orders = [elem["z_order"] for elem in elements]
    assert z_orders == [0, 1, 2, 3]  # Sequential global ordering
    
    # Check column assignments work correctly
    # Reading order should be: left column (by Y), then right column (by Y)
    # Elements: left_text(0,100), left_table(0,200), right_heading(1,50), right_text(1,100)
    
    # Verify column assignments
    columns = [elem.get("column_index") for elem in elements]
    assert columns == [0, 0, 1, 1]  # Left column first, then right column


def test_two_pass_document_processing():
    """Test complete two-pass processing pipeline."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    from ehrx.layout.global_ordering import GlobalOrderingManager
    from ehrx.layout.enhanced_router import DocumentProcessor
    
    # Mock document-wide blocks (simulating data from multiple pages)
    all_pages_blocks = [
        # Page 1: Clear two-column layout
        [
            {"bbox_px": [50, 100, 200, 150], "type": 1},   # Left column
            {"bbox_px": [350, 100, 500, 150], "type": 1},  # Right column
            {"bbox_px": [50, 200, 200, 300], "type": 4},   # Left table
            {"bbox_px": [350, 200, 500, 300], "type": 5}   # Right figure
        ],
        # Page 2: Continuing same layout 
        [
            {"bbox_px": [50, 50, 200, 100], "type": 1},    # Left column
            {"bbox_px": [350, 50, 500, 100], "type": 1},   # Right column  
            {"bbox_px": [50, 150, 200, 250], "type": 1},   # Left text
            {"bbox_px": [350, 150, 500, 250], "type": 4}   # Right table
        ]
    ]
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Pass 1: Analyze layout
    column_layout = processor.analyze_document_layout(all_pages_blocks, page_width=600.0)
    
    # Should detect 2 columns
    assert column_layout.column_count == 2
    assert len(column_layout.boundaries) == 3
    
    # Pass 2: Process with global state
    results = processor.process_document_with_global_ordering(
        all_pages_blocks, column_layout
    )
    
    # Verify continuous z-ordering across pages
    page1_elements, page2_elements = results
    
    # Check z-order continuity
    max_page1_z = max(elem["z_order"] for elem in page1_elements)
    min_page2_z = min(elem["z_order"] for elem in page2_elements)
    
    assert min_page2_z == max_page1_z + 1  # No gap between pages
    
    # Check column assignments
    for elements in results:
        for elem in elements:
            assert "column_index" in elem
            assert elem["column_index"] in [0, 1]


def test_heading_association_across_columns():
    """Test heading-element associations work correctly in multi-column layout."""
    from ehrx.layout.column_detection import ColumnLayout
    from ehrx.layout.global_ordering import GlobalOrderingManager
    
    # Two-column layout
    layout = ColumnLayout(
        column_count=2,
        boundaries=[0.0, 300.0, 600.0],
        page_width=600.0
    )
    
    manager = GlobalOrderingManager(layout)
    
    # Simulate document flow: headings followed by content
    elements = [
        # Left column heading
        {"id": "H1_001", "type": "text_block", "bbox_px": [50, 50, 200, 80], 
         "payload": {"text": "MEDICATIONS"}},
        
        # Right column heading  
        {"id": "H1_002", "type": "text_block", "bbox_px": [350, 50, 500, 80],
         "payload": {"text": "ALLERGIES"}},
        
        # Left column content
        {"id": "E_001", "type": "table", "bbox_px": [50, 100, 200, 200]},
        {"id": "E_002", "type": "text_block", "bbox_px": [50, 220, 200, 260]},
        
        # Right column content
        {"id": "E_003", "type": "figure", "bbox_px": [350, 100, 500, 200]},
        {"id": "E_004", "type": "text_block", "bbox_px": [350, 220, 500, 260]}
    ]
    
    # Process in reading order
    sorted_elements = manager.sort_elements_reading_order(elements)
    
    # Track headings as we encounter them
    for elem in sorted_elements:
        if elem["type"] == "text_block" and "MEDICATIONS" in elem.get("payload", {}).get("text", ""):
            manager.track_heading_context(elem)
        elif elem["type"] == "text_block" and "ALLERGIES" in elem.get("payload", {}).get("text", ""):
            manager.track_heading_context(elem)
    
    # Test associations
    left_table = {"bbox_px": [50, 100, 200, 200]}  # Should associate with MEDICATIONS
    right_figure = {"bbox_px": [350, 100, 500, 200]}  # Should associate with ALLERGIES
    
    left_heading = manager.find_associated_heading(left_table)
    right_heading = manager.find_associated_heading(right_figure)
    
    assert left_heading == "H1_001"  # MEDICATIONS heading
    assert right_heading == "H1_002"  # ALLERGIES heading


def test_fallback_behavior():
    """Test system falls back gracefully when column detection fails."""
    from ehrx.layout.column_detection import DocumentColumnDetector
    
    detector = DocumentColumnDetector()
    
    # Irregular blocks that should trigger fallback
    irregular_blocks = [
        {"bbox_px": [10, 20, 50, 80]},    # Too few blocks
        {"bbox_px": [100, 30, 150, 90]}   # for reliable clustering
    ]
    
    # Should fall back to single column
    layout = detector.detect_multi_column_layout(irregular_blocks, page_width=500.0)
    
    assert layout.column_count == 1
    assert layout.boundaries == [0.0, 500.0]
    
    # Global ordering should still work with single column
    from ehrx.layout.global_ordering import GlobalOrderingManager
    
    manager = GlobalOrderingManager(layout)
    
    # Elements should be assigned to column 0
    for block in irregular_blocks:
        column = manager.assign_element_to_column(block)
        assert column == 0