"""
Standalone test script for hierarchy building with visual debug output.

Usage:
    python test_hierarchy_standalone.py --pdf <path> --output <dir> --pages <num>
"""
import argparse
import json
import logging
from pathlib import Path

from ehrx.core.config import EHRXConfig
from ehrx.pdf.pager import PDFRasterizer
from ehrx.detect import LayoutDetector
from ehrx.ocr import OCREngine
from ehrx.hierarchy import HierarchyBuilder
from ehrx.visualize import HierarchyVisualizer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test hierarchy building with visual debug output")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--output", default="hierarchy_test_output", help="Output directory")
    parser.add_argument("--pages", type=int, default=5, help="Number of pages to process")
    parser.add_argument("--config", help="Config file path")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    
    config = EHRXConfig.from_yaml(config_path)
    
    # Setup paths
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing PDF: {pdf_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max pages: {args.pages}")
    
    # Initialize components
    logger.info("Initializing components...")
    rasterizer = PDFRasterizer(pdf_path)
    detector = LayoutDetector(config.detector)
    ocr_engine = OCREngine(config.ocr)
    hierarchy_builder = HierarchyBuilder(config.model_dump())
    visualizer = HierarchyVisualizer(output_dir)
    
    # Process pages
    pages_data = []
    page_images = {}
    
    max_pages = min(args.pages, rasterizer.page_count)
    logger.info(f"Processing {max_pages} pages...")
    
    for page_num in range(max_pages):
        logger.info(f"Processing page {page_num}...")
        
        # Rasterize page
        page_image, page_info = rasterizer.rasterize_page(page_num, dpi=200)
        page_images[page_num] = page_image
        
        # Detect layout
        layout = detector.detect_layout(page_image)
        logger.info(f"  Detected {len(layout)} elements")
        
        # Extract text for each block
        elements = []
        for block in layout:
            bbox_px = [block.block.x_1, block.block.y_1, 
                      block.block.x_2, block.block.y_2]
            
            elem = {
                "id": f"E_{page_num:04d}_{len(elements):04d}",
                "page": page_num,
                "type": getattr(block, 'type', 'text_block'),
                "bbox_px": bbox_px,
                "payload": {"text": "", "confidence": 0.0}
            }
            
            # OCR text blocks
            if elem["type"] == "text_block":
                try:
                    x1, y1, x2, y2 = [int(c) for c in bbox_px]
                    
                    # Ensure coordinates are within image bounds
                    h, w = page_image.shape[:2]
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        cropped = page_image[y1:y2, x1:x2]
                        
                        if cropped.size > 0:
                            ocr_result = ocr_engine.extract_text(cropped, "text")
                            confidence = ocr_result.get("confidence", 0.0)
                            if confidence is None:
                                confidence = 0.0
                            
                            elem["payload"] = {
                                "text": ocr_result["text"],
                                "confidence": float(confidence)
                            }
                except Exception as e:
                    logger.warning(f"  OCR failed for element: {e}")
            
            elements.append(elem)
        
        logger.info(f"  Extracted text from {sum(1 for e in elements if e['payload']['text'])} elements")
        
        # Store page data
        page_data = {
            "page_num": page_num,
            "elements": elements,
            "page_info": {
                "width_px": page_info.width_px,
                "height_px": page_info.height_px,
                "dpi": page_info.dpi
            }
        }
        pages_data.append(page_data)
        
        # Visualize label detection
        logger.info(f"  Detecting document label...")
        label = hierarchy_builder.label_detector.detect_label(
            elements, 
            page_data["page_info"]
        )
        detection_region = hierarchy_builder.label_detector.get_detection_region(
            page_data["page_info"]
        )
        
        # Find label element
        label_elem = None
        for elem in elements:
            if elem.get("payload", {}).get("text") == label:
                label_elem = elem
                break
        
        logger.info(f"  Label detected: '{label}'")
        
        visualizer.visualize_label_detection(
            page_image,
            page_num,
            elements,
            detection_region,
            label,
            label_elem
        )
    
    # Build hierarchy
    logger.info("Building document hierarchy...")
    hierarchy = hierarchy_builder.build_hierarchy(pages_data)
    
    # Save hierarchy to JSON
    hierarchy_file = output_dir / "hierarchy_output.json"
    with open(hierarchy_file, "w") as f:
        json.dump(hierarchy, f, indent=2)
    
    logger.info(f"Saved hierarchy to {hierarchy_file}")
    
    # Generate visual outputs
    logger.info("Generating visual debug outputs...")
    visualizer.create_document_summary(hierarchy, pdf_path.stem)
    visualizer.visualize_document_overview(hierarchy, page_images, pdf_path.stem)
    
    # Visualize sections for each page
    for page_data in pages_data:
        page_num = page_data["page_num"]
        elements = page_data["elements"]
        
        # Detect headings for this page
        page_headings = []
        for i, elem in enumerate(elements):
            if elem.get("type") != "text_block":
                continue
            
            text = elem.get("payload", {}).get("text", "").strip()
            if not text:
                continue
            
            # Check if it's a heading
            is_heading, level, scores = hierarchy_builder.section_detector._classify_heading(
                elem, elements, i
            )
            
            if is_heading:
                page_headings.append({
                    "element": elem,
                    "text": text,
                    "level": level,
                    "scores": scores
                })
        
        # Find document type for this page
        doc_type = None
        for doc in hierarchy["documents"]:
            if page_num in doc["pages"]:
                doc_type = doc["document_type"]
                break
        
        visualizer.visualize_sections(
            page_images[page_num],
            page_num,
            elements,
            page_headings,
            doc_type
        )
    
    logger.info("Visual outputs generated!")
    
    # Print summary
    print("\n" + "=" * 80)
    print("HIERARCHY TEST RESULTS")
    print("=" * 80)
    print(f"PDF: {pdf_path.name}")
    print(f"Total documents: {hierarchy['total_documents']}")
    print(f"Total pages: {hierarchy['total_pages']}")
    print(f"Categories: {', '.join(hierarchy['categories'])}")
    print("\nDocuments:")
    for doc in hierarchy["documents"]:
        print(f"  - {doc['document_type']} ({doc['category']})")
        print(f"    Pages: {doc['page_range'][0]}-{doc['page_range'][1]}")
        print(f"    Sections: {len(doc['sections'])}")
        for section in doc['sections'][:3]:  # Show first 3 sections
            print(f"      • [L{section['level']}] {section['heading']}")
    print("\nOutput files:")
    print(f"  - Hierarchy JSON: {hierarchy_file}")
    print(f"  - Debug visualizations: {output_dir / 'debug'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

