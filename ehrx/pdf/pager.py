"""
PDF → page raster + coordinate mapping
"""
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterator, Union
import numpy as np

# PDF backends (PyMuPDF preferred, pdf2image as fallback)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


@dataclass
class PageInfo:
    """Information about a PDF page and its rasterization."""
    page_number: int
    width_pdf: float  # PDF points
    height_pdf: float  # PDF points
    width_px: int  # Pixel width
    height_px: int  # Pixel height
    dpi: int
    rotation: int = 0
    
    @property
    def scale_x(self) -> float:
        """Scale factor from PDF points to pixels (X-axis)."""
        return self.width_px / self.width_pdf
    
    @property
    def scale_y(self) -> float:
        """Scale factor from PDF points to pixels (Y-axis)."""
        return self.height_px / self.height_pdf


class CoordinateMapper:
    """Handle coordinate transformations between PDF and pixel space."""
    
    def __init__(self, page_info: PageInfo):
        self.page_info = page_info
    
    def pdf_to_pixel(self, bbox_pdf: List[float]) -> List[int]:
        """Convert PDF coordinates to pixel coordinates.
        
        PDF coordinates have origin at bottom-left, pixels at top-left.
        """
        x0_pdf, y0_pdf, x1_pdf, y1_pdf = bbox_pdf
        
        # Convert to pixel coordinates with origin flip
        x0_px = int(x0_pdf * self.page_info.scale_x)
        y0_px = int((self.page_info.height_pdf - y1_pdf) * self.page_info.scale_y)  # Flip Y
        x1_px = int(x1_pdf * self.page_info.scale_x)
        y1_px = int((self.page_info.height_pdf - y0_pdf) * self.page_info.scale_y)  # Flip Y
        
        return [x0_px, y0_px, x1_px, y1_px]
    
    def pixel_to_pdf(self, bbox_px: List[int]) -> List[float]:
        """Convert pixel coordinates to PDF coordinates."""
        x0_px, y0_px, x1_px, y1_px = bbox_px
        
        # Convert to PDF coordinates with origin flip
        x0_pdf = x0_px / self.page_info.scale_x
        y1_pdf = self.page_info.height_pdf - (y0_px / self.page_info.scale_y)  # Flip Y
        x1_pdf = x1_px / self.page_info.scale_x
        y0_pdf = self.page_info.height_pdf - (y1_px / self.page_info.scale_y)  # Flip Y
        
        return [x0_pdf, y0_pdf, x1_pdf, y1_pdf]


class PDFRasterizer:
    """Handle PDF rasterization using available backends."""
    
    def __init__(self, pdf_path: Union[str, Path]):
        self.pdf_path = Path(pdf_path)
        self.logger = logging.getLogger(__name__)
        
        # Determine available backend
        if PYMUPDF_AVAILABLE:
            self.backend = "pymupdf"
            self._doc = fitz.open(str(self.pdf_path))
            self.page_count = self._doc.page_count
            self.logger.info(f"Using PyMuPDF backend for {self.page_count} pages")
            
        elif PDF2IMAGE_AVAILABLE:
            self.backend = "pdf2image"
            # Get page count by converting first page only
            self._test_pages = convert_from_path(str(self.pdf_path), first_page=1, last_page=1)
            # For page count, we need to load all pages once (cached)
            self._all_pages = convert_from_path(str(self.pdf_path))
            self.page_count = len(self._all_pages)
            self.logger.info(f"Using pdf2image backend for {self.page_count} pages")
            
        else:
            raise RuntimeError("No PDF backend available. Install PyMuPDF or pdf2image.")
    
    def rasterize_page(self, page_num: int, dpi: int = 150) -> Tuple[np.ndarray, PageInfo]:
        """Rasterize a single page to numpy array.
        
        Args:
            page_num: Page number (0-indexed)
            dpi: Target DPI for rasterization
            
        Returns:
            Tuple of (image_array, page_info)
        """
        if self.backend == "pymupdf":
            return self._rasterize_page_pymupdf(page_num, dpi)
        else:
            return self._rasterize_page_pdf2image(page_num, dpi)
    
    def _rasterize_page_pymupdf(self, page_num: int, dpi: int) -> Tuple[np.ndarray, PageInfo]:
        """Rasterize using PyMuPDF."""
        page = self._doc[page_num]
        
        # Get page dimensions
        rect = page.rect
        width_pdf = rect.width
        height_pdf = rect.height
        rotation = page.rotation
        
        # Calculate scale factor
        scale = dpi / 72.0  # PDF default is 72 DPI
        
        # Create pixmap
        mat = fitz.Matrix(scale, scale)
        pixmap = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        width_px = pixmap.width
        height_px = pixmap.height
        
        # Get image data
        img_data = np.frombuffer(pixmap.samples, dtype=np.uint8)
        img_data = img_data.reshape((height_px, width_px, pixmap.n))
        
        # Convert to RGB if needed
        if pixmap.n == 4:  # RGBA
            img_data = img_data[:, :, :3]  # Drop alpha
        elif pixmap.n == 1:  # Grayscale
            img_data = np.stack([img_data] * 3, axis=-1)
        
        # Create page info
        page_info = PageInfo(
            page_number=page_num,
            width_pdf=width_pdf,
            height_pdf=height_pdf,
            width_px=width_px,
            height_px=height_px,
            dpi=dpi,
            rotation=rotation
        )
        
        return img_data, page_info
    
    def _rasterize_page_pdf2image(self, page_num: int, dpi: int) -> Tuple[np.ndarray, PageInfo]:
        """Rasterize using pdf2image."""
        # Convert specific page
        pages = convert_from_path(
            str(self.pdf_path),
            dpi=dpi,
            first_page=page_num + 1,  # pdf2image uses 1-based indexing
            last_page=page_num + 1
        )
        
        if not pages:
            raise ValueError(f"Could not rasterize page {page_num}")
        
        pil_image = pages[0]
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        img_data = np.array(pil_image)
        width_px, height_px = pil_image.size
        
        # For pdf2image, we don't have direct access to PDF dimensions
        # Estimate based on standard page size and DPI
        width_pdf = (width_px / dpi) * 72  # Convert back to points
        height_pdf = (height_px / dpi) * 72
        
        page_info = PageInfo(
            page_number=page_num,
            width_pdf=width_pdf,
            height_pdf=height_pdf,
            width_px=width_px,
            height_px=height_px,
            dpi=dpi,
            rotation=0  # pdf2image doesn't provide rotation info easily
        )
        
        return img_data, page_info
    
    def extract_vector_text(self, page_num: int) -> List[Dict[str, Any]]:
        """Extract vector text from page if available.
        
        Only works with PyMuPDF backend.
        """
        if self.backend != "pymupdf":
            return []
        
        page = self._doc[page_num]
        text_blocks = page.get_text("dict")["blocks"]
        
        extracted_blocks = []
        for block in text_blocks:
            if "lines" in block:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        extracted_blocks.append({
                            "bbox": span["bbox"],
                            "text": span["text"],
                            "block_no": block["number"],
                            "line_no": len(extracted_blocks)
                        })
        
        return extracted_blocks
    
    def close(self):
        """Close the PDF document."""
        if hasattr(self, '_doc'):
            self._doc.close()


def parse_page_range(page_range: str, total_pages: int) -> List[int]:
    """Parse page range string into list of 0-indexed page numbers.
    
    Args:
        page_range: String like "all", "1-10", "1-5,8-10", "3"
        total_pages: Total number of pages in document
        
    Returns:
        List of 0-indexed page numbers
    """
    if page_range.lower() == "all":
        return list(range(total_pages))
    
    pages = []
    
    # Split by commas for multiple ranges
    ranges = page_range.split(",")
    
    for range_str in ranges:
        range_str = range_str.strip()
        
        if "-" in range_str:
            # Range like "3-7"
            start_str, end_str = range_str.split("-", 1)
            start = max(1, int(start_str.strip()))  # 1-indexed input
            end = min(total_pages, int(end_str.strip()))  # 1-indexed input
            
            # Convert to 0-indexed and add to list
            pages.extend(range(start - 1, end))
        else:
            # Single page like "5"
            page_num = max(1, min(total_pages, int(range_str)))  # 1-indexed input
            pages.append(page_num - 1)  # Convert to 0-indexed
    
    # Remove duplicates and sort
    return sorted(list(set(pages)))


class Pager:
    """Main interface for PDF page iteration and rasterization."""
    
    def __init__(self, pdf_path: Union[str, Path]):
        self.pdf_path = Path(pdf_path)
        self.rasterizer = PDFRasterizer(pdf_path)
        self.page_count = self.rasterizer.page_count
        self.logger = logging.getLogger(__name__)
    
    def pages(self, page_range: str = "all", dpi: int = 150) -> Iterator[Tuple[np.ndarray, PageInfo, CoordinateMapper]]:
        """Iterate over pages with rasterization.
        
        Args:
            page_range: Range specification ("all", "1-10", "1-5,8-10")
            dpi: DPI for rasterization
            
        Yields:
            Tuple of (image_array, page_info, coordinate_mapper)
        """
        page_numbers = parse_page_range(page_range, self.page_count)
        
        self.logger.info(f"Processing {len(page_numbers)} pages at {dpi} DPI")
        
        for page_num in page_numbers:
            self.logger.debug(f"Rasterizing page {page_num + 1}/{self.page_count}")
            
            # Rasterize page
            image, page_info = self.rasterizer.rasterize_page(page_num, dpi)
            
            # Create coordinate mapper
            mapper = CoordinateMapper(page_info)
            
            yield image, page_info, mapper
    
    def get_page_vector_text(self, page_num: int) -> List[Dict[str, Any]]:
        """Get vector text from a specific page if available."""
        return self.rasterizer.extract_vector_text(page_num)
    
    def close(self):
        """Close the pager and release resources."""
        self.rasterizer.close()

