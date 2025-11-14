"""
Write JSONL + index.json + assets
"""
import json
import logging
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import threading
import numpy as np
from PIL import Image

from .layout.column_detection import ColumnLayout


class SerializationError(Exception):
    """Custom exception for serialization errors."""
    pass


class JsonlWriter:
    """Streaming JSONL writer for document.elements.jsonl."""
    
    def __init__(self, output_path: Path):
        """Initialize JSONL writer.
        
        Args:
            output_path: Path to output JSONL file
        """
        self.output_path = Path(output_path)
        self.logger = logging.getLogger(__name__)
        self._file_handle = None
        self._lock = threading.Lock()
        self._element_count = 0
        
        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file for writing
        self._file_handle = open(self.output_path, 'w', encoding='utf-8')
        self.logger.info(f"Opened JSONL writer: {self.output_path}")
    
    def append(self, element: Dict[str, Any]) -> None:
        """Append element to JSONL file.
        
        Args:
            element: Element dictionary to write
            
        Raises:
            SerializationError: If writing fails
        """
        if not self._file_handle:
            raise SerializationError("JSONL writer is closed")
        
        try:
            with self._lock:
                # Write JSON on single line
                json_line = json.dumps(element, ensure_ascii=False, separators=(',', ':'))
                self._file_handle.write(json_line + '\n')
                self._file_handle.flush()
                self._element_count += 1
                
                if self._element_count % 100 == 0:
                    self.logger.debug(f"Written {self._element_count} elements")
                    
        except Exception as e:
            raise SerializationError(f"Failed to write element {element.get('id', 'unknown')}: {e}")
    
    def close(self) -> int:
        """Close the JSONL writer.
        
        Returns:
            Number of elements written
        """
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            self.logger.info(f"Closed JSONL writer. Wrote {self._element_count} elements")
        
        return self._element_count
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AssetManager:
    """Manages saving of images and CSV files."""
    
    def __init__(self, assets_dir: Path):
        """Initialize asset manager.
        
        Args:
            assets_dir: Directory for saving assets
        """
        self.assets_dir = Path(assets_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create assets directory
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Initialized asset manager: {self.assets_dir}")
    
    def save_table_image(self, element_id: str, crop_image: np.ndarray) -> str:
        """Save table crop image.
        
        Args:
            element_id: Element ID for naming
            crop_image: Cropped image array
            
        Returns:
            Relative path to saved image
        """
        filename = f"table_{element_id}.png"
        return self._save_image(crop_image, filename)
    
    def save_figure_image(self, element_id: str, crop_image: np.ndarray) -> str:
        """Save figure crop image.
        
        Args:
            element_id: Element ID for naming
            crop_image: Cropped image array
            
        Returns:
            Relative path to saved image
        """
        filename = f"figure_{element_id}.png"
        return self._save_image(crop_image, filename)
    
    def save_handwriting_image(self, element_id: str, crop_image: np.ndarray) -> str:
        """Save handwriting crop image.
        
        Args:
            element_id: Element ID for naming
            crop_image: Cropped image array
            
        Returns:
            Relative path to saved image
        """
        filename = f"hand_{element_id}.png"
        return self._save_image(crop_image, filename)
    
    def save_table_csv(self, element_id: str, rows: List[List[str]], 
                      headers: Optional[List[str]] = None) -> str:
        """Save table data as CSV.
        
        Args:
            element_id: Element ID for naming
            rows: Table rows data
            headers: Optional column headers
            
        Returns:
            Relative path to saved CSV
            
        Raises:
            SerializationError: If CSV writing fails
        """
        filename = f"table_{element_id}.csv"
        csv_path = self.assets_dir / filename
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers if provided
                if headers:
                    writer.writerow(headers)
                
                # Write data rows
                for row in rows:
                    writer.writerow(row)
            
            self.logger.debug(f"Saved table CSV: {csv_path}")
            return f"assets/{filename}"
            
        except Exception as e:
            raise SerializationError(f"Failed to save CSV for {element_id}: {e}")
    
    def _save_image(self, crop_image: np.ndarray, filename: str) -> str:
        """Save image array to file.
        
        Args:
            crop_image: Image array (H,W,C or H,W)
            filename: Output filename
            
        Returns:
            Relative path to saved image
            
        Raises:
            SerializationError: If image saving fails
        """
        try:
            # Convert numpy array to PIL Image
            if len(crop_image.shape) == 2:
                # Grayscale
                image = Image.fromarray(crop_image)
            elif len(crop_image.shape) == 3:
                # RGB or BGR
                if crop_image.shape[2] == 3:
                    # Assume BGR (OpenCV format) and convert to RGB
                    rgb_image = crop_image[:, :, ::-1]
                    image = Image.fromarray(rgb_image)
                else:
                    image = Image.fromarray(crop_image)
            else:
                raise SerializationError(f"Unsupported image shape: {crop_image.shape}")
            
            # Save image
            image_path = self.assets_dir / filename
            image.save(image_path, format='PNG', optimize=True)
            
            self.logger.debug(f"Saved image: {image_path}")
            return f"assets/{filename}"
            
        except Exception as e:
            raise SerializationError(f"Failed to save image {filename}: {e}")


class IndexBuilder:
    """Builds document index with enhanced metadata."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def build_manifest(
        self, 
        config: Dict[str, Any], 
        stats: Dict[str, Any], 
        column_layout: ColumnLayout
    ) -> Dict[str, Any]:
        """Build manifest section with column layout and stats.
        
        Args:
            config: Processing configuration
            stats: Processing statistics
            column_layout: Document column layout
            
        Returns:
            Manifest dictionary
        """
        manifest = {
            "pages": stats.get("total_pages", 0),
            "detector": "vlm",  # VLM-based detection
            "ocr": config.get("ocr", {}).get("engine", "tesseract"),
            "created_at": datetime.now().isoformat(),
            "column_layout": column_layout.to_dict(),
            "stats": {
                "total_elements": stats.get("total_elements", 0),
                "z_order_range": stats.get("z_order_range", [0, 0]),
                "elements_by_type": stats.get("elements_by_type", {}),
                "processing_time_seconds": stats.get("processing_time", 0.0)
            }
        }
        
        return manifest
    
    def build_index(
        self, 
        doc_id: str,
        hierarchy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build complete document index with only hierarchy.
        
        Args:
            doc_id: Document ID
            hierarchy: Document hierarchy (categories -> subcategories -> documents -> pages -> elements)
            
        Returns:
            Complete index dictionary with clean structure
        """
        # If hierarchy is empty, return minimal structure
        if not hierarchy or not hierarchy.get("categories"):
            return {
                "ehr_id": doc_id,
                "total_pages": 0,
                "total_documents": 0,
                "categories": {}
            }
        
        # Return clean structure with hierarchy
        index = {
            "ehr_id": doc_id,
            "total_pages": hierarchy.get("total_pages", 0),
            "total_documents": hierarchy.get("total_documents", 0),
            "categories": hierarchy.get("categories", {})
        }
        
        return index
    
    def write_index(self, output_path: Path, index_data: Dict[str, Any]) -> None:
        """Write index to JSON file.
        
        Args:
            output_path: Path to output index file
            index_data: Index data to write
            
        Raises:
            SerializationError: If writing fails
        """
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Wrote document index: {output_path}")
            
        except Exception as e:
            raise SerializationError(f"Failed to write index: {e}")


class DocumentSerializer:
    """High-level document serialization coordinator."""
    
    def __init__(self, output_dir: Path, doc_id: str, config: Dict[str, Any]):
        """Initialize document serializer.
        
        Args:
            output_dir: Output directory for all files
            doc_id: Document ID
            config: Processing configuration
        """
        self.output_dir = Path(output_dir)
        self.doc_id = doc_id
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.output_dir / "assets"
        
        # Initialize components
        self.jsonl_writer = JsonlWriter(self.output_dir / "document.elements.jsonl")
        self.asset_manager = AssetManager(self.assets_dir)
        self.index_builder = IndexBuilder()
        
        # Tracking
        self._element_count = 0
        self._elements_by_type = {}
        self._z_order_range = [float('inf'), float('-inf')]
        
        self.logger.info(f"Initialized serializer for {doc_id} in {output_dir}")
    
    def serialize_element(
        self, 
        element: Dict[str, Any], 
        crop_image: Optional[np.ndarray] = None,
        table_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Serialize an element with optional assets.
        
        Args:
            element: Element data to serialize
            crop_image: Optional cropped image for visual elements
            table_data: Optional table data with rows/headers
        """
        try:
            element_id = element.get("id", "unknown")
            element_type = element.get("type", "unknown")
            
            # Update tracking stats
            self._update_stats(element)
            
            # Handle asset saving based on element type
            if crop_image is not None:
                if element_type == "table":
                    image_ref = self.asset_manager.save_table_image(element_id, crop_image)
                    element["payload"] = element.get("payload", {})
                    element["payload"]["image_ref"] = image_ref
                    
                    # Save CSV if table data provided
                    if table_data and table_data.get("rows"):
                        csv_ref = self.asset_manager.save_table_csv(
                            element_id, 
                            table_data["rows"],
                            table_data.get("headers")
                        )
                        element["payload"]["csv_ref"] = csv_ref
                        element["payload"]["headers"] = table_data.get("headers")
                        element["payload"]["rows"] = table_data["rows"]
                
                elif element_type == "figure":
                    image_ref = self.asset_manager.save_figure_image(element_id, crop_image)
                    element["payload"] = element.get("payload", {})
                    element["payload"]["image_ref"] = image_ref
                
                elif element_type == "handwriting":
                    image_ref = self.asset_manager.save_handwriting_image(element_id, crop_image)
                    element["payload"] = element.get("payload", {})
                    element["payload"]["image_ref"] = image_ref
            
            # Write element to JSONL
            self.jsonl_writer.append(element)
            
        except Exception as e:
            self.logger.error(f"Failed to serialize element {element.get('id', 'unknown')}: {e}")
            raise SerializationError(f"Element serialization failed: {e}")
    
    def finalize(
        self, 
        hierarchy: Dict[str, Any],
        column_layout: ColumnLayout,
        processing_time: float = 0.0
    ) -> Dict[str, Any]:
        """Finalize serialization and write index.
        
        Args:
            hierarchy: Document hierarchy in new format
            column_layout: Document column layout (not included in output)
            processing_time: Total processing time
            
        Returns:
            Final statistics
        """
        try:
            # Close JSONL writer
            final_count = self.jsonl_writer.close()
            
            # Build statistics for internal use only
            stats = {
                "total_elements": final_count,
                "elements_by_type": self._elements_by_type.copy(),
                "processing_time": processing_time
            }
            
            # Build and write index (no manifest or column_layout in output)
            index_data = self.index_builder.build_index(self.doc_id, hierarchy)
            index_path = self.output_dir / "document.index.json"
            self.index_builder.write_index(index_path, index_data)
            
            self.logger.info(f"Finalized serialization: {final_count} elements")
            return stats
            
        except Exception as e:
            raise SerializationError(f"Serialization finalization failed: {e}")
    
    def _update_stats(self, element: Dict[str, Any]) -> None:
        """Update internal statistics tracking."""
        self._element_count += 1
        
        # Track element types
        element_type = element.get("type", "unknown")
        self._elements_by_type[element_type] = self._elements_by_type.get(element_type, 0) + 1
        
        # Track z_order range
        z_order = element.get("z_order", 0)
        self._z_order_range[0] = min(self._z_order_range[0], z_order)
        self._z_order_range[1] = max(self._z_order_range[1], z_order)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure cleanup
        try:
            self.jsonl_writer.close()
        except:
            pass
