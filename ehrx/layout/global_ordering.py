"""
Global ordering and state management for document-wide element processing
"""
from typing import List, Dict, Any, Optional
from .column_detection import ColumnLayout


class GlobalOrderingManager:
    """Manages document-wide state for proper element sequencing and associations."""
    
    def __init__(self, column_layout: ColumnLayout):
        """Initialize global ordering manager.
        
        Args:
            column_layout: Column layout configuration for the document
        """
        self.column_layout = column_layout
        self._global_z_order = 0
        
        # Track active headings per column for associations
        # Key: column_index, Value: latest heading_id in that column
        self._active_headings_per_column: Dict[int, str] = {}
    
    def get_next_z_order(self) -> int:
        """Get next global z-order value and increment counter.
        
        Returns:
            Next z-order value for element assignment
        """
        current = self._global_z_order
        self._global_z_order += 1
        return current
    
    def get_current_z_order(self) -> int:
        """Get current z-order value without incrementing.
        
        Returns:
            Current z-order counter value (last assigned + 1)
        """
        return self._global_z_order - 1 if self._global_z_order > 0 else 0
    
    def assign_element_to_column(self, element: Dict[str, Any]) -> int:
        """Assign an element to a column based on its coordinates.
        
        Args:
            element: Element with bbox_px coordinates
            
        Returns:
            Column index (0-based)
        """
        bbox_px = element.get("bbox_px", [0, 0, 0, 0])
        if len(bbox_px) < 4:
            return 0  # Default to first column
        
        left_edge = float(bbox_px[0])
        return self.column_layout.assign_to_column(left_edge)
    
    def sort_elements_reading_order(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort elements in proper reading order: (column_index, y_coordinate).
        
        Args:
            elements: List of elements with bbox_px coordinates
            
        Returns:
            Elements sorted in reading order
        """
        def sort_key(element):
            # Assign to column
            column_idx = self.assign_element_to_column(element)
            
            # Get Y coordinate (top edge)
            bbox_px = element.get("bbox_px", [0, 0, 0, 0])
            y_coord = float(bbox_px[1]) if len(bbox_px) >= 4 else 0.0
            
            return (column_idx, y_coord)
        
        return sorted(elements, key=sort_key)
    
    def track_heading_context(self, heading_element: Dict[str, Any]) -> None:
        """Track heading context for column-aware associations.
        
        Args:
            heading_element: Heading element to track
        """
        # Determine which column this heading belongs to
        column_idx = self.assign_element_to_column(heading_element)
        
        # Update active heading for this column
        heading_id = heading_element.get("id")
        if heading_id:
            self._active_headings_per_column[column_idx] = heading_id
    
    def find_associated_heading(self, element: Dict[str, Any]) -> Optional[str]:
        """Find the associated heading for an element in the same column.
        
        Args:
            element: Element to find heading for (table, figure, etc.)
            
        Returns:
            Heading ID if found, None otherwise
        """
        # Determine which column this element belongs to
        column_idx = self.assign_element_to_column(element)
        
        # Return active heading for this column
        return self._active_headings_per_column.get(column_idx)
    
    def reset_heading_context(self, column_idx: Optional[int] = None) -> None:
        """Reset heading context for a specific column or all columns.
        
        Args:
            column_idx: Column to reset, or None for all columns
        """
        if column_idx is not None:
            self._active_headings_per_column.pop(column_idx, None)
        else:
            self._active_headings_per_column.clear()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get current document processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        return {
            "total_elements_processed": self._global_z_order,
            "active_headings_per_column": dict(self._active_headings_per_column),
            "column_count": self.column_layout.column_count,
            "page_width": self.column_layout.page_width
        }