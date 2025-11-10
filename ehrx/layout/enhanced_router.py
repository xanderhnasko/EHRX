"""
Enhanced ElementRouter with column detection and global ordering integration
"""
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np
from datetime import datetime

if TYPE_CHECKING:
    import layoutparser as lp

from .column_detection import ColumnLayout, DocumentColumnDetector
from .global_ordering import GlobalOrderingManager


class EnhancedElementRouter:
    """Enhanced element router with column awareness and global ordering."""
    
    def __init__(self, config, doc_id: str, column_layout: ColumnLayout):
        """Initialize enhanced element router.
        
        Args:
            config: EHRX configuration
            doc_id: Document identifier
            column_layout: Column layout for the document
        """
        self.config = config
        self.doc_id = doc_id
        self.column_layout = column_layout
        self.logger = logging.getLogger(__name__)
        
        # Initialize global ordering manager
        self.ordering_manager = GlobalOrderingManager(column_layout)
    
    def process_layout_blocks_with_global_ordering(
        self, 
        blocks: List, 
        page_image: np.ndarray,
        page_info,
        mapper
    ) -> List[Dict[str, Any]]:
        """Process layout blocks with global ordering and column awareness.
        
        Args:
            blocks: Layout blocks from detection
            page_image: Page image for cropping
            page_info: Page metadata
            mapper: Coordinate mapping utilities
            
        Returns:
            List of processed elements with global ordering
        """
        elements = []
        
        # Convert blocks to standard format for ordering
        element_data = []
        for block in blocks:
            # Extract coordinates from block
            bbox_px = [block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2]
            
            element_data.append({
                "bbox_px": bbox_px,
                "block": block,
                "type": self._map_block_type(getattr(block, 'type', 'unknown'))
            })
        
        # Sort in column-aware reading order
        sorted_elements = self.ordering_manager.sort_elements_reading_order(element_data)
        
        # Process each element with global z-order
        for elem_data in sorted_elements:
            try:
                # Assign global z-order
                z_order = self.ordering_manager.get_next_z_order()
                
                # Assign column
                column_idx = self.ordering_manager.assign_element_to_column(elem_data)
                
                # Create processed element
                element = {
                    "id": f"E_{z_order:04d}",
                    "doc_id": self.doc_id,
                    "page": page_info.page_number,
                    "type": elem_data["type"],
                    "bbox_px": elem_data["bbox_px"],
                    "bbox_pdf": mapper.pixel_to_pdf(elem_data["bbox_px"]),
                    "z_order": z_order,
                    "column_index": column_idx,
                    "source": "ocr",
                    "created_at": datetime.now().isoformat(),
                    "detector_name": "layoutparser",
                    "detector_conf": getattr(elem_data["block"], 'score', None)
                }
                
                # Track heading context if this is a heading
                if self._is_heading_element(element):
                    self.ordering_manager.track_heading_context(element)
                
                elements.append(element)
                
            except Exception as e:
                self.logger.error(f"Failed to process element: {e}")
                continue
        
        return elements
    
    def _map_block_type(self, block_type) -> str:
        """Map block type to element type."""
        # Handle numeric types from layoutparser
        if isinstance(block_type, (int, float)):
            numeric_mapping = {
                1: "text_block",
                2: "text_block",  # Title -> text_block
                3: "text_block",  # List -> text_block
                4: "table",
                5: "figure"
            }
            return numeric_mapping.get(int(block_type), "text_block")
        
        # Handle string types
        type_mapping = {
            "text_block": "text_block",
            "text": "text_block",
            "Text": "text_block",
            "table": "table",
            "Table": "table",
            "figure": "figure",
            "Figure": "figure"
        }
        
        return type_mapping.get(str(block_type), "text_block")
    
    def _is_heading_element(self, element: Dict[str, Any]) -> bool:
        """Determine if an element is a heading (simplified for testing)."""
        # In real implementation, would check text content, formatting, etc.
        # For testing, assume text_blocks with specific patterns are headings
        return element["type"] == "text_block"


class DocumentProcessor:
    """Handles two-pass document processing pipeline."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_document_layout(
        self, 
        all_pages_blocks: List[List[Dict[str, Any]]], 
        page_width: float
    ) -> ColumnLayout:
        """Pass 1: Analyze document layout to detect columns.
        
        Args:
            all_pages_blocks: Blocks from all pages
            page_width: Page width
            
        Returns:
            Detected column layout
        """
        detector = DocumentColumnDetector()
        
        # Flatten all blocks across pages
        all_blocks = []
        for page_blocks in all_pages_blocks:
            all_blocks.extend(page_blocks)
        
        # Detect column layout
        if len(all_blocks) >= 3:
            layout = detector.detect_multi_column_layout(all_blocks, page_width)
        else:
            layout = detector.detect_single_column_layout(all_blocks, page_width)
        
        self.logger.info(f"Detected {layout.column_count} columns")
        return layout
    
    def process_document_with_global_ordering(
        self,
        all_pages_blocks: List[List[Dict[str, Any]]],
        column_layout: ColumnLayout
    ) -> List[List[Dict[str, Any]]]:
        """Pass 2: Process document with global ordering.
        
        Args:
            all_pages_blocks: Blocks from all pages
            column_layout: Column layout from pass 1
            
        Returns:
            Processed elements per page with global ordering
        """
        ordering_manager = GlobalOrderingManager(column_layout)
        results = []
        
        for page_idx, page_blocks in enumerate(all_pages_blocks):
            page_elements = []
            
            # Convert to standard format
            element_data = []
            for block in page_blocks:
                element_data.append({
                    "bbox_px": block["bbox_px"],
                    "type": self._map_block_type(block["type"])
                })
            
            # Sort in reading order
            sorted_elements = ordering_manager.sort_elements_reading_order(element_data)
            
            # Process each element
            for elem_data in sorted_elements:
                z_order = ordering_manager.get_next_z_order()
                column_idx = ordering_manager.assign_element_to_column(elem_data)
                
                element = {
                    "id": f"E_{z_order:04d}",
                    "page": page_idx + 1,
                    "type": elem_data["type"],
                    "bbox_px": elem_data["bbox_px"],
                    "z_order": z_order,
                    "column_index": column_idx
                }
                
                page_elements.append(element)
            
            results.append(page_elements)
        
        return results
    
    def _map_block_type(self, block_type) -> str:
        """Map block type to element type."""
        if isinstance(block_type, (int, float)):
            numeric_mapping = {
                1: "text_block",
                2: "text_block",
                3: "text_block",
                4: "table",
                5: "figure"
            }
            return numeric_mapping.get(int(block_type), "text_block")
        
        return str(block_type)