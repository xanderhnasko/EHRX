#!/usr/bin/env python3
"""
Test script for LayoutParser integration with real EHR PDFs.

Tests layout detection, vector text extraction, and OCR on sample documents.
Outputs annotated PDF and JSON results.
"""

import sys
from pathlib import Path
import json
import argparse
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ehrx.pager import Pager, PageInfo, CoordinateMapper
from ehrx.detect import LayoutDetector
from ehrx.ocr import OCREngine
from ehrx.config import load_config, get_default_config_path
from ehrx.utils import setup_logger, Timer, BBox, generate_element_id

try:
    import fitz  # PyMuPDF for PDF annotation
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False


def draw_boxes_on_page(
    image: np.ndarray,
    blocks: list,
    _unused
) -> np.ndarray:
    """
    Draw bounding boxes on image with labels.
    
    Args:
        image: RGB image
        blocks: List of DetectedBlock objects
        page_data: PageData for coordinate conversion
    
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    # Color map for different types
    color_map = {
        "text_block": (0, 255, 0),      # Green
        "table": (255, 0, 0),            # Red
        "figure": (0, 0, 255),           # Blue
        "handwriting": (255, 255, 0),   # Yellow
    }
    
    for block in blocks:
        bbox = block.bbox_px
        label = block.label_mapped
        conf = block.confidence
        
        color = color_map.get(label, (128, 128, 128))
        
        # Draw rectangle
        x0, y0 = int(bbox.x0), int(bbox.y0)
        x1, y1 = int(bbox.x1), int(bbox.y1)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
        
        # Draw label
        label_text = f"{label} ({conf:.2f})"
        cv2.putText(
            annotated,
            label_text,
            (x0, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    return annotated


def create_annotated_pdf(
    pdf_path: Path,
    results: list,
    output_path: Path,
    logger
):
    """
    Create annotated PDF with bounding boxes.
    
    Args:
        pdf_path: Original PDF path
        results: List of page results
        output_path: Output PDF path
        logger: Logger
    """
    if not FITZ_AVAILABLE:
        logger.warning("PyMuPDF not available, skipping annotated PDF")
        return
    
    doc = fitz.open(str(pdf_path))
    
    # Color map
    color_map = {
        "text_block": (0, 1, 0),      # Green
        "table": (1, 0, 0),            # Red
        "figure": (0, 0, 1),           # Blue
        "handwriting": (1, 1, 0),      # Yellow
    }
    
    for result in results:
        page_num = result["page_num"]
        page = doc[page_num]
        
        for element in result["elements"]:
            bbox_pdf = element["bbox_pdf"]
            label = element["label_mapped"]
            conf = element["confidence"]
            
            color = color_map.get(label, (0.5, 0.5, 0.5))
            
            # Draw rectangle
            rect = fitz.Rect(bbox_pdf[0], bbox_pdf[1], bbox_pdf[2], bbox_pdf[3])
            page.draw_rect(rect, color=color, width=2)
            
            # Add label
            label_text = f"{label} ({conf:.2f})"
            page.insert_text(
                (bbox_pdf[0], bbox_pdf[1] - 2),
                label_text,
                fontsize=8,
                color=color
            )
    
    doc.save(str(output_path))
    doc.close()
    
    logger.info(f"Annotated PDF saved: {output_path}")


def test_layoutparser(
    pdf_path: Path,
    output_dir: Path,
    max_pages: int = 20,
    config_path: Path = None,
    allow_vector: bool = True
):
    """
    Test LayoutParser on EHR PDF.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory for results
        max_pages: Maximum pages to process
        config_path: Path to config file (optional)
        allow_vector: Try vector text extraction before OCR
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("test_layoutparser", level="INFO")
    logger.info(f"Testing LayoutParser on: {pdf_path}")
    logger.info(f"Processing first {max_pages} pages")
    
    # Load config
    if config_path is None:
        config_path = get_default_config_path()
    
    config = load_config(config_path)
    logger.info(f"Loaded config from: {config_path}")
    
    # Initialize components
    detector = LayoutDetector(config.detector, logger)
    # Initialize OCR if available
    ocr_engine = None
    try:
        ocr_engine = OCREngine(config.ocr, logger)
    except Exception as e:
        logger.warning(f"OCR engine unavailable, proceeding without OCR: {e}")
    
    # Results storage
    all_results = []
    element_id_counter = 0
    
    # Process PDF
    with Timer("Total processing", logger):
        pager = Pager(pdf_path)
        try:
            logger.info(f"PDF has {pager.page_count} pages")
            page_range = f"1-{max_pages}"
            for image, page_info, mapper in pager.pages(page_range=page_range, dpi=200):
                page_num = page_info.page_number
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing page {page_num + 1}/{min(max_pages, pager.page_count)}")
                
                with Timer(f"Page {page_num}", logger):
                    # Detect layout
                    with Timer("Layout detection", logger):
                        blocks = detector.detect(image)
                    
                    logger.info(f"Detected {len(blocks)} blocks")
                    
                    # Process each block
                    elements = []
                    for block in blocks:
                        element_id_counter += 1
                        element_id = generate_element_id(block.label_mapped, element_id_counter)
                        
                        # Convert bbox to PDF coordinates
                        bbox_px_list = [int(block.bbox_px.x0), int(block.bbox_px.y0), int(block.bbox_px.x1), int(block.bbox_px.y1)]
                        bbox_pdf_list = mapper.pixel_to_pdf(bbox_px_list)
                        
                        # Try vector text extraction first
                        text = ""
                        source = "ocr"
                        ocr_confidence = 0.0
                        
                        if allow_vector and block.label_mapped == "text_block":
                            with Timer("Vector text extraction", logger):
                                vec_text = pager.extract_vector_text_in_bbox(page_num, bbox_pdf_list)
                                if vec_text:
                                    text = vec_text
                                    source = "vector"
                                    ocr_confidence = 1.0
                                    logger.info(f"  {element_id}: Vector text extracted ({len(text)} chars)")
                        
                        # Fall back to OCR if no vector text
                        if not text and block.label_mapped in ["text_block", "table"] and ocr_engine is not None:
                            with Timer("OCR", logger):
                                ocr_img, ocr_info = pager.rasterize_page_detached(page_num, dpi=300)
                                ocr_mapper = CoordinateMapper(ocr_info)
                                bbox_px_ocr = ocr_mapper.pdf_to_pixel(bbox_pdf_list)
                                bbox_obj = BBox(x0=bbox_px_ocr[0], y0=bbox_px_ocr[1], x1=bbox_px_ocr[2], y1=bbox_px_ocr[3])
                                ocr_result = ocr_engine.ocr_region(ocr_img, bbox_px=bbox_obj, psm=config.ocr.psm_text if block.label_mapped == "text_block" else config.ocr.psm_table)
                                text = ocr_result.text
                                source = "ocr"
                                ocr_confidence = ocr_result.confidence
                                logger.info(f"  {element_id}: OCR text extracted ({len(text)} chars, conf={ocr_confidence:.2f})")
                        
                        # Create element
                        element = {
                            "id": element_id,
                            "page": page_num,
                            "type": block.label_mapped,
                            "bbox_pdf": bbox_pdf_list,
                            "bbox_px": block.bbox_px.to_list(),
                            "label": block.label,
                            "label_mapped": block.label_mapped,
                            "confidence": block.confidence,
                            "source": source,
                            "text": text[:200] if text else "",  # Truncate for privacy
                            "text_length": len(text),
                            "ocr_confidence": ocr_confidence
                        }
                        
                        elements.append(element)
                    
                    # Save annotated page image
                    annotated_img = draw_boxes_on_page(image, blocks, None)
                    
                    img_output_path = output_dir / f"page_{page_num:04d}_annotated.png"
                    cv2.imwrite(str(img_output_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                    logger.info(f"Saved annotated image: {img_output_path}")
                    
                    # Store results
                    page_result = {
                        "page_num": page_num,
                        "page_width_pt": page_info.width_pdf,
                        "page_height_pt": page_info.height_pdf,
                        "page_width_px": page_info.width_px,
                        "page_height_px": page_info.height_px,
                        "dpi": page_info.dpi,
                        "elements": elements
                    }
                    
                    all_results.append(page_result)
    
        finally:
            pager.close()

    # Save JSON results
    json_output_path = output_dir / "detection_results.json"
    with open(json_output_path, 'w') as f:
        json.dump({
            "pdf_path": str(pdf_path),
            "pages_processed": len(all_results),
            "total_elements": element_id_counter,
            "detector": detector.get_detector_info(),
            "ocr": (ocr_engine.get_ocr_info() if ocr_engine else {"engine": "none"}),
            "pages": all_results
        }, f, indent=2)
    
    logger.info(f"\nJSON results saved: {json_output_path}")
    
    # Create annotated PDF
    annotated_pdf_path = output_dir / "annotated.pdf"
    create_annotated_pdf(pdf_path, all_results, annotated_pdf_path, logger)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Pages processed: {len(all_results)}")
    logger.info(f"Total elements detected: {element_id_counter}")
    
    # Count by type
    type_counts = {}
    vector_count = 0
    ocr_count = 0
    
    for page_result in all_results:
        for element in page_result["elements"]:
            elem_type = element["type"]
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
            
            if element["source"] == "vector":
                vector_count += 1
            elif element["source"] == "ocr":
                ocr_count += 1
    
    logger.info(f"\nElements by type:")
    for elem_type, count in sorted(type_counts.items()):
        logger.info(f"  {elem_type}: {count}")
    
    logger.info(f"\nText extraction:")
    logger.info(f"  Vector text: {vector_count}")
    logger.info(f"  OCR text: {ocr_count}")
    
    logger.info(f"\nOutput files:")
    logger.info(f"  JSON: {json_output_path}")
    logger.info(f"  Annotated PDF: {annotated_pdf_path}")
    logger.info(f"  Page images: {output_dir}/page_*_annotated.png")


def main():
    parser = argparse.ArgumentParser(
        description="Test LayoutParser on EHR PDFs"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to PDF file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_output"),
        help="Output directory (default: test_output)"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=20,
        help="Number of pages to process (default: 20)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Disable vector text extraction"
    )
    
    args = parser.parse_args()
    
    if not args.pdf.exists():
        print(f"Error: PDF not found: {args.pdf}")
        sys.exit(1)
    
    test_layoutparser(
        pdf_path=args.pdf,
        output_dir=args.output,
        max_pages=args.pages,
        config_path=args.config,
        allow_vector=not args.no_vector
    )


if __name__ == "__main__":
    main()

