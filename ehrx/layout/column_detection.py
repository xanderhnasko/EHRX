"""
Column detection and layout analysis
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ColumnLayout:
    """Represents column layout configuration for a document."""
    
    column_count: int
    boundaries: List[float]  # x-coordinates of column boundaries
    page_width: float
    
    def __post_init__(self):
        """Validate the column layout after initialization."""
        # Validate boundaries count
        expected_boundaries = self.column_count + 1
        if len(self.boundaries) != expected_boundaries:
            raise ValueError(
                f"boundaries must have column_count + 1 elements. "
                f"Expected {expected_boundaries}, got {len(self.boundaries)}"
            )
        
        # Validate boundaries are sorted
        if self.boundaries != sorted(self.boundaries):
            raise ValueError("boundaries must be in ascending order")
    
    def assign_to_column(self, x_coordinate: float) -> int:
        """Assign an x-coordinate to a column index.
        
        Args:
            x_coordinate: X position to assign
            
        Returns:
            Column index (0-based)
        """
        # Handle out-of-bounds coordinates
        if x_coordinate <= self.boundaries[0]:
            return 0
        if x_coordinate >= self.boundaries[-1]:
            return self.column_count - 1
        
        # Find the appropriate column
        for i in range(self.column_count):
            if self.boundaries[i] <= x_coordinate < self.boundaries[i + 1]:
                return i
        
        # Fallback (shouldn't reach here with valid input)
        return self.column_count - 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "column_count": self.column_count,
            "boundaries": self.boundaries,
            "page_width": self.page_width
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnLayout":
        """Create ColumnLayout from dictionary."""
        return cls(
            column_count=data["column_count"],
            boundaries=data["boundaries"],
            page_width=data["page_width"]
        )
    
    def __eq__(self, other) -> bool:
        """Check equality with another ColumnLayout."""
        if not isinstance(other, ColumnLayout):
            return False
        return (
            self.column_count == other.column_count and
            self.boundaries == other.boundaries and
            self.page_width == other.page_width
        )


class DocumentColumnDetector:
    """Detects column layout patterns in documents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the column detector.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
    
    def detect_single_column_layout(self, blocks: List[Dict[str, Any]], 
                                   page_width: float) -> ColumnLayout:
        """Detect a single-column layout (baseline/fallback).
        
        This method always returns a single-column layout that spans 
        the entire page width, regardless of block positions.
        
        Args:
            blocks: List of layout blocks (ignored for single-column)
            page_width: Width of the page
            
        Returns:
            ColumnLayout with single column spanning full width
            
        Raises:
            ValueError: If page_width is not positive
        """
        if page_width <= 0:
            raise ValueError("page_width must be positive")
        
        return ColumnLayout(
            column_count=1,
            boundaries=[0.0, page_width],
            page_width=page_width
        )
    
    def extract_left_edges(self, blocks: List[Dict[str, Any]]) -> List[float]:
        """Extract left-edge x-coordinates from layout blocks.
        
        Args:
            blocks: List of layout blocks with coordinate information
            
        Returns:
            List of left-edge x-coordinates (floats)
        """
        left_edges = []
        
        for block in blocks:
            try:
                x_coord = self._extract_x_coordinate(block)
                if x_coord is not None:
                    left_edges.append(float(x_coord))
            except (ValueError, TypeError, IndexError):
                # Skip malformed blocks
                continue
        
        return left_edges
    
    def _extract_x_coordinate(self, block: Dict[str, Any]) -> Optional[float]:
        """Extract x-coordinate from a single block.
        
        Supports multiple formats:
        - {"bbox_px": [x0, y0, x1, y1]}
        - {"block": {"x_1": x0, "y_1": y0, "x_2": x1, "y_2": y1}}
        
        Args:
            block: Single layout block
            
        Returns:
            X-coordinate (left edge) or None if not extractable
        """
        # Try standard bbox_px format
        if "bbox_px" in block:
            bbox = block["bbox_px"]
            if isinstance(bbox, list) and len(bbox) >= 4:
                try:
                    return float(bbox[0])  # x0 coordinate
                except (ValueError, TypeError):
                    pass
        
        # Try layoutparser block format
        if "block" in block and isinstance(block["block"], dict):
            block_data = block["block"]
            if "x_1" in block_data:
                try:
                    return float(block_data["x_1"])
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def filter_coordinate_outliers(self, coordinates: List[float]) -> List[float]:
        """Filter outliers from coordinate list using IQR method.
        
        Args:
            coordinates: List of x-coordinates
            
        Returns:
            Filtered list with outliers removed
        """
        if len(coordinates) <= 3:
            return coordinates  # Too few points to filter
        
        # Sort coordinates to calculate quartiles
        sorted_coords = sorted(coordinates)
        n = len(sorted_coords)
        
        # Calculate quartiles
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_coords[q1_idx]
        q3 = sorted_coords[q3_idx]
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers
        filtered = [coord for coord in coordinates 
                   if lower_bound <= coord <= upper_bound]
        
        return filtered
    
    def detect_multi_column_layout(self, blocks: List[Dict[str, Any]], 
                                  page_width: float) -> ColumnLayout:
        """Detect multi-column layout using K-means clustering.
        
        Args:
            blocks: List of layout blocks with coordinate information
            page_width: Width of the page
            
        Returns:
            ColumnLayout with detected columns
        """
        # Extract coordinates
        left_edges = self.extract_left_edges(blocks)
        
        if len(left_edges) < 3:
            # Not enough data for clustering, fall back to single column
            return self.detect_single_column_layout(blocks, page_width)
        
        # Filter outliers
        filtered_coords = self.filter_coordinate_outliers(left_edges)
        
        if len(filtered_coords) < 3:
            # Not enough clean data, fall back to single column
            return self.detect_single_column_layout(blocks, page_width)
        
        # Find optimal number of clusters
        max_k = min(3, len(filtered_coords) // 2)  # Limit to reasonable max
        if max_k < 2:
            return self.detect_single_column_layout(blocks, page_width)
            
        best_k, score = self.find_optimal_k(filtered_coords, max_k)
        
        if best_k == 1 or score < 0.3:  # Poor clustering score
            return self.detect_single_column_layout(blocks, page_width)
        
        # Perform K-means with optimal k
        cluster_centers = self._perform_kmeans(filtered_coords, best_k)
        
        # Create boundaries from cluster centers
        boundaries = self.create_boundaries_from_centers(cluster_centers, page_width)
        
        return ColumnLayout(
            column_count=best_k,
            boundaries=boundaries,
            page_width=page_width
        )
    
    def find_optimal_k(self, coordinates: List[float], max_k: int) -> Tuple[int, float]:
        """Find optimal number of clusters using silhouette analysis.
        
        Args:
            coordinates: List of x-coordinates
            max_k: Maximum number of clusters to try
            
        Returns:
            Tuple of (best_k, silhouette_score)
        """
        if len(coordinates) < 2:
            return 1, 0.0
        
        if len(set(coordinates)) == 1:  # All coordinates identical
            return 1, 0.0
        
        best_k = 1
        best_score = -1.0
        
        # Test k=2 to max_k
        coords_array = np.array(coordinates).reshape(-1, 1)
        
        for k in range(2, min(max_k + 1, len(coordinates))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(coords_array)
                
                # Calculate silhouette score
                if len(set(cluster_labels)) > 1:  # More than one cluster found
                    score = silhouette_score(coords_array, cluster_labels)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
            except Exception:
                # K-means failed for this k, continue
                continue
        
        return best_k, best_score
    
    def _perform_kmeans(self, coordinates: List[float], k: int) -> List[float]:
        """Perform K-means clustering and return sorted cluster centers.
        
        Args:
            coordinates: List of x-coordinates
            k: Number of clusters
            
        Returns:
            Sorted list of cluster centers
        """
        coords_array = np.array(coordinates).reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(coords_array)
        
        # Return sorted cluster centers
        centers = [float(center[0]) for center in kmeans.cluster_centers_]
        return sorted(centers)
    
    def create_boundaries_from_centers(self, cluster_centers: List[float], 
                                     page_width: float) -> List[float]:
        """Create column boundaries from cluster centers.
        
        Args:
            cluster_centers: Sorted list of cluster centers
            page_width: Width of the page
            
        Returns:
            List of column boundaries
        """
        if len(cluster_centers) == 1:
            return [0.0, page_width]
        
        boundaries = [0.0]  # Start with left edge
        
        # Add boundaries halfway between cluster centers
        for i in range(len(cluster_centers) - 1):
            mid_point = (cluster_centers[i] + cluster_centers[i + 1]) / 2
            boundaries.append(mid_point)
        
        boundaries.append(page_width)  # End with right edge
        
        return boundaries