"""
Element routing and processing pipeline
"""
import logging
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from datetime import datetime
from pathlib import Path

if TYPE_CHECKING:
    import layoutparser as lp

# Core imports
from .ocr import OCREngine, OCRError
from .core.config import EHRXConfig
from .core.utils import IDGenerator, BBox, pdf_to_pixel_coords, pixel_to_pdf_coords, Timer
from .pdf.pager import CoordinateMapper, PageInfo


class ElementRoutingError(Exception):
    """Custom exception for element routing errors."""
    pass


class ElementRouter:
    """Routes layout detection blocks to appropriate element processors."""
    
    def __init__(self, config: EHRXConfig, doc_id: str):
        """Initialize element router.
        
        Args:
            config: Complete EHRX configuration
            doc_id: Document identifier for element IDs
        """
        self.config = config
        self.doc_id = doc_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.id_generator = IDGenerator(doc_id)
        self.ocr_engine = OCREngine(config.ocr)
        
        # Element processors
        self._processors = {
            "text_block": self._process_text_block,
            "table": self._process_table, 
            "figure": self._process_figure,
            "handwriting": self._process_handwriting
        }
        
        self.logger.info(f"Initialized element router for document: {doc_id}")
    
    def process_layout_blocks(self, layout: "lp.Layout", page_image: np.ndarray, 
                             page_info: PageInfo, 
                             mapper: CoordinateMapper) -> List[Dict[str, Any]]:
        """Process all layout blocks on a page into elements.
        
        Args:
            layout: LayoutParser Layout with detected blocks
            page_image: RGB page image (for cropping)
            page_info: PageInfo dataclass with page metadata
            mapper: Coordinate mapping utilities
            
        Returns:
            List of processed element dictionaries
        """
        elements = []
        page_num = page_info.page_number
        
        self.logger.info(f"Processing {len(layout)} layout blocks on page {page_num}")
        
        # Sort blocks for reading order (left-to-right, top-to-bottom)
        sorted_blocks = self._sort_blocks_reading_order(layout)
        
        for z_order, block in enumerate(sorted_blocks):
            try:
                element = self._route_block(
                    block, page_image, page_info, mapper, z_order
                )
                if element:
                    elements.append(element)
                    
            except Exception as e:
                self.logger.error(f"Failed to process block {z_order} on page {page_num}: {e}")
                # Continue with other blocks
                continue
        
        self.logger.info(f"Successfully processed {len(elements)} elements on page {page_num}")
        return elements
    
    def _route_block(self, block, page_image: np.ndarray, page_info: PageInfo,
                    mapper: CoordinateMapper, z_order: int) -> Optional[Dict[str, Any]]:
        """Route a single layout block to appropriate processor.
        
        Args:
            block: LayoutParser block object
            page_image: RGB page image
            page_info: Page metadata
            mapper: Coordinate mapping utilities
            z_order: Reading order position
            
        Returns:
            Processed element dictionary or None if processing failed
        """
        # Get block type and coordinates
        block_type = getattr(block, 'type', 'unknown')
        bbox_px = [block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2]
        
        self.logger.debug(f"Block type: {block_type}, bbox_px: {bbox_px}")
        
        # Map block type to our element types
        element_type = self._map_block_type(block_type)
        if element_type not in self._processors:
            self.logger.warning(f"Unknown element type: {element_type}, skipping")
            return None
        
        # Convert coordinates
        bbox_pdf = mapper.pixel_to_pdf(bbox_px)
        
        # Crop image region
        cropped_image = self._crop_image_region(page_image, bbox_px)
        if cropped_image is None:
            self.logger.warning(f"Failed to crop image for {element_type} block")
            return None
        
        # Generate element ID
        element_id = self.id_generator.next_element_id()
        
        # Create base element structure
        element = self._create_base_element(
            element_id, element_type, page_info, bbox_pdf, bbox_px, 
            z_order, block
        )
        
        # Route to appropriate processor
        processor = self._processors[element_type]
        
        try:
            with Timer(f"process_{element_type}", self.logger):
                payload = processor(cropped_image, block, element)
                
            element["payload"] = payload
            return element
            
        except Exception as e:
            self.logger.error(f"Processor failed for {element_type}: {e}")
            return None
    
    def _map_block_type(self, block_type) -> str:
        """Map LayoutParser block type to our element type."""
        # Handle numeric types from layoutparser
        if isinstance(block_type, (int, float)):
            # PubLayNet numeric mapping: 1=Text, 2=Title, 3=List, 4=Table, 5=Figure
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
            "Title": "text_block",
            "List": "text_block",
            "table": "table",
            "Table": "table",
            "figure": "figure", 
            "Figure": "figure",
            "handwriting": "handwriting",
            "Handwriting": "handwriting"
        }
        
        return type_mapping.get(str(block_type), "text_block")  # Default to text_block
    
    def _crop_image_region(self, image: np.ndarray, bbox_px: List[int]) -> Optional[np.ndarray]:
        """Crop image region based on pixel coordinates.
        
        Args:
            image: Full page image
            bbox_px: Pixel coordinates [x0, y0, x1, y1]
            
        Returns:
            Cropped image region or None if invalid
        """
        try:
            x0, y0, x1, y1 = bbox_px
            
            # Ensure coordinates are integers
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            
            # Validate coordinates
            if x0 >= x1 or y0 >= y1:
                return None
            
            # Clamp to image bounds
            h, w = image.shape[:2]
            x0 = max(0, min(x0, w-1))
            y0 = max(0, min(y0, h-1)) 
            x1 = max(x0+1, min(x1, w))
            y1 = max(y0+1, min(y1, h))
            
            # Crop region
            cropped = image[y0:y1, x0:x1]
            
            # Ensure minimum size
            if cropped.shape[0] < 5 or cropped.shape[1] < 5:
                return None
                
            return cropped
            
        except Exception as e:
            self.logger.error(f"Image cropping failed: {e}")
            return None
    
    def _create_base_element(self, element_id: str, element_type: str,
                           page_info: PageInfo, bbox_pdf: List[float],
                           bbox_px: List[int], z_order: int, block) -> Dict[str, Any]:
        """Create base element structure with common fields."""
        
        # Get confidence score if available
        confidence = getattr(block, 'score', None)
        
        return {
            "id": element_id,
            "doc_id": self.doc_id,
            "page": page_info.page_number,
            "type": element_type,
            "bbox_pdf": bbox_pdf,
            "bbox_px": bbox_px,
            "rotation": 0,  # TODO: Detect rotation from block
            "z_order": z_order,
            "source": "ocr",  # Will be updated by processors if needed
            "created_at": datetime.now().isoformat(),
            "detector_name": "layoutparser",
            "detector_conf": confidence
        }
    
    def _sort_blocks_reading_order(self, layout: "lp.Layout") -> List:
        """Sort blocks in reading order (left-to-right, top-to-bottom)."""
        blocks = list(layout)
        
        # Sort by Y coordinate (top to bottom), then X coordinate (left to right)
        return sorted(blocks, key=lambda b: (b.block.y_1, b.block.x_1))
    
    # Element Processors
    
    def _process_text_block(self, cropped_image: np.ndarray, block, 
                          element: Dict[str, Any]) -> Dict[str, Any]:
        """Process text block element.
        
        Returns:
            Payload dict with text and metadata
        """
        # Try vector text extraction first (if configured)
        vector_text = None
        if hasattr(self.config, 'allow_vector') and self.config.allow_vector:
            # TODO: Implement vector text extraction
            pass
        
        # Use OCR for text extraction
        ocr_result = self.ocr_engine.extract_text(
            cropped_image, 
            element_type="text",
            apply_preprocessing=True
        )
        
        # Update element source
        element["source"] = "vector" if vector_text else "ocr"
        
        return {
            "text": vector_text or ocr_result["text"],
            "ocr_confidence": ocr_result.get("confidence"),
            "preprocessing_applied": ocr_result.get("preprocessing_applied", False)
        }
    
    def _process_table(self, cropped_image: np.ndarray, block,
                      element: Dict[str, Any]) -> Dict[str, Any]:
        """Process table element.
        
        Returns:
            Payload dict with table structure and OCR text
        """
        # Extract text using OCR
        ocr_result = self.ocr_engine.extract_text(
            cropped_image,
            element_type="table", 
            apply_preprocessing=True
        )
        
        # TODO: Implement table structure detection
        # For now, just return OCR lines
        ocr_lines = ocr_result["text"].split('\n') if ocr_result["text"] else []
        
        return {
            "headers": None,  # TODO: Detect headers
            "rows": None,     # TODO: Detect row structure  
            "csv_ref": None,  # TODO: Save CSV if structured
            "ocr_lines": ocr_lines,
            "ocr_confidence": ocr_result.get("confidence")
        }
    
    def _process_figure(self, cropped_image: np.ndarray, block,
                       element: Dict[str, Any]) -> Dict[str, Any]:
        """Process figure element.
        
        Returns:
            Payload dict with image reference and caption
        """
        # TODO: Save cropped image to assets
        image_ref = f"assets/figure_{element['id']}.png"
        
        # TODO: Detect nearby caption
        caption = None
        
        return {
            "image_ref": image_ref,
            "caption": caption
        }
    
    def _process_handwriting(self, cropped_image: np.ndarray, block,
                           element: Dict[str, Any]) -> Dict[str, Any]:
        """Process handwriting element.
        
        Returns:
            Payload dict with image reference and optional OCR
        """
        # TODO: Save cropped image to assets  
        image_ref = f"assets/hand_{element['id']}.png"
        
        # Attempt OCR on handwriting (often low confidence)
        try:
            ocr_result = self.ocr_engine.extract_text(
                cropped_image,
                element_type="handwriting",
                apply_preprocessing=True
            )
            ocr_text = ocr_result["text"]
            ocr_confidence = ocr_result.get("confidence")
        except Exception:
            ocr_text = None
            ocr_confidence = None
        
        return {
            "image_ref": image_ref,
            "ocr_text": ocr_text,
            "ocr_confidence": ocr_confidence
        }

