#!/usr/bin/env python3
"""
Test script for visual verification of layout detection
"""
import sys
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add ehrx to path
sys.path.insert(0, str(Path(__file__).parent))

from ehrx.detect import LayoutDetector, LayoutDetectionError
from ehrx.pdf.pager import Pager
from ehrx.route import ElementRouter, ElementRoutingError
from ehrx.core.config import load_default_config, setup_logging_from_config


def test_detection_on_pdf(pdf_path: str, page_range: str = "1", output_dir: str = "test_output", enable_ocr: bool = False):
    """Test layout detection on PDF pages with visualization.
    
    Args:
        pdf_path: Path to PDF file
        page_range: Page range to test (e.g., "1", "1-3", "1,3,5", or "all")
        output_dir: Directory to save output images
        enable_ocr: Whether to run OCR on detected elements
    """
    try:
        # Load configuration
        config = load_default_config()
        logger = setup_logging_from_config(config, level="INFO")
        
        logger.info(f"Testing layout detection on: {pdf_path}")
        logger.info(f"Page range: {page_range}")
        
        # Initialize detector
        detector = LayoutDetector(config.detector)
        
        # Initialize element router for OCR processing
        element_router = None
        if enable_ocr:
            try:
                doc_id = f"test_{Path(pdf_path).stem}"
                element_router = ElementRouter(config, doc_id)
                logger.info("OCR processing enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize OCR: {e}")
                enable_ocr = False
        
        # Initialize pager
        pager = Pager(pdf_path)
        logger.info(f"PDF has {pager.page_count} pages")
        
        # Parse page range
        if page_range.lower() == "all":
            pages_to_process = list(range(pager.page_count))
        else:
            pages_to_process = []
            for part in page_range.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    pages_to_process.extend(range(start-1, end))  # Convert to 0-indexed
                else:
                    pages_to_process.append(int(part) - 1)  # Convert to 0-indexed
        
        # Validate page numbers
        invalid_pages = [p for p in pages_to_process if p >= pager.page_count or p < 0]
        if invalid_pages:
            raise ValueError(f"Invalid pages: {[p+1 for p in invalid_pages]}. PDF has {pager.page_count} pages")
        
        logger.info(f"Processing {len(pages_to_process)} pages: {[p+1 for p in pages_to_process]}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Track statistics across all pages
        all_page_stats = []
        total_elements = 0
        
        # Process each page
        for page_idx, page_num in enumerate(pages_to_process):
            logger.info(f"Processing page {page_num + 1} ({page_idx + 1}/{len(pages_to_process)})")
            
            # Process single page  
            for i, (image, page_info, mapper) in enumerate(pager.pages(f"{page_num + 1}", dpi=200)):
                if i > 0:  # Only process first (and only) page from range
                    break
                
                logger.info(f"Image shape: {image.shape}")
                
                # Run layout detection
                layout = detector.detect_layout(image)
                
                # Run OCR processing if enabled
                elements = []
                if enable_ocr and element_router:
                    try:
                        elements = element_router.process_layout_blocks(layout, image, page_info, mapper)
                        logger.info(f"OCR processed {len(elements)} elements")
                    except Exception as e:
                        logger.error(f"OCR processing failed: {e}")
                
                # Get detection stats
                stats = detector.get_detection_stats(layout)
                stats['page_number'] = page_num + 1
                all_page_stats.append(stats)
                total_elements += stats['total_elements']
                
                logger.info(f"Page {page_num + 1} detection stats: {stats['total_elements']} elements")
                
                # Visualize with LayoutParser
                try:
                    annotated_image = detector.visualize_detection(image, layout, box_width=3)
                    
                    # Save LayoutParser visualization
                    lp_output_path = output_path / f"page_{page_num + 1}_layoutparser_detection.png"
                    plt.figure(figsize=(12, 16))
                    plt.imshow(annotated_image)
                    plt.axis('off')
                    plt.title(f"LayoutParser Detection - Page {page_num + 1}\n{stats['total_elements']} elements detected")
                    plt.tight_layout()
                    plt.savefig(lp_output_path, dpi=150, bbox_inches='tight')
                    plt.show()
                    logger.info(f"Saved LayoutParser visualization: {lp_output_path}")
                    
                except Exception as e:
                    logger.warning(f"LayoutParser visualization failed: {e}")
                    logger.info("Falling back to manual visualization")
                    
                    # Manual visualization as fallback
                    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
                    ax.imshow(image)
                    
                    # Draw bounding boxes manually
                    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
                    element_type_colors = {}
                    
                    for i, block in enumerate(layout):
                        # Get block coordinates
                        x1, y1, x2, y2 = block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2
                        
                        # Assign color by type
                        block_type = block.type if hasattr(block, 'type') else 'unknown'
                        if block_type not in element_type_colors:
                            color_idx = len(element_type_colors) % len(colors)
                            element_type_colors[block_type] = colors[color_idx]
                        
                        color = element_type_colors[block_type]
                        
                        # Create rectangle
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=3, edgecolor=color, facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # Improved label positioning and formatting
                        confidence = f" ({block.score:.2f})" if hasattr(block, 'score') else ""
                        element_num = f"[{i+1}] "
                        label_text = f"{element_num}{block_type}{confidence}"
                        
                        # Calculate adaptive font size based on box size
                        box_width = x2 - x1
                        box_height = y2 - y1
                        font_size = min(max(int(box_width / 50), 8), 16)  # Adaptive font size between 8-16
                        
                        # Position label outside the box when possible
                        label_y = y1 - 10 if y1 > 20 else y2 + 15  # Above or below box
                        label_x = x1
                        
                        # Ensure label doesn't go outside image bounds
                        if label_y < 0:
                            label_y = y1 + 15
                        if label_y > image.shape[0]:
                            label_y = y2 - 15
                        
                        ax.text(label_x, label_y, label_text, 
                               color=color, fontsize=font_size, weight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=color))
                    
                    ax.set_xlim(0, image.shape[1])
                    ax.set_ylim(image.shape[0], 0)  # Flip Y axis
                    ax.axis('off')
                    ax.set_title(f"Layout Detection - Page {page_num + 1}\n{stats['total_elements']} elements detected")
                    
                    # Add legend
                    legend_elements = [patches.Patch(color=color, label=f"{type_name} ({stats['element_types'].get(type_name, 0)})")
                                     for type_name, color in element_type_colors.items()]
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    # Save manual visualization
                    manual_output_path = output_path / f"page_{page_num + 1}_manual_detection.png"
                    plt.tight_layout()
                    plt.savefig(manual_output_path, dpi=150, bbox_inches='tight')
                    plt.show()
                    logger.info(f"Saved manual visualization: {manual_output_path}")
            
            # Print detailed detection results
            print(f"\n=== Detection Results for Page {page_num + 1} ===")
            print(f"Total elements: {stats['total_elements']}")
            print(f"Element types: {stats['element_types']}")
            if stats['confidence_stats']:
                print(f"Confidence stats: {stats['confidence_stats']}")
            
            print(f"\nDetailed elements:")
            for i, block in enumerate(layout):
                block_type = block.type if hasattr(block, 'type') else 'unknown'
                confidence = f" (conf: {block.score:.3f})" if hasattr(block, 'score') else ""
                coords = f"[{block.block.x_1:.0f}, {block.block.y_1:.0f}, {block.block.x_2:.0f}, {block.block.y_2:.0f}]"
                print(f"  {i+1}. {block_type}{confidence} at {coords}")
                
                # Show OCR results if available
                if enable_ocr and i < len(elements):
                    element = elements[i]
                    payload = element.get("payload", {})
                    
                    if "text" in payload:
                        text = payload["text"][:100] + "..." if len(payload["text"]) > 100 else payload["text"]
                        print(f"      OCR: \"{text}\"")
                        if "ocr_confidence" in payload and payload["ocr_confidence"]:
                            print(f"      OCR Confidence: {payload['ocr_confidence']:.2f}")
                    
                    elif "ocr_lines" in payload:
                        ocr_lines = payload["ocr_lines"][:3]  # Show first 3 lines
                        for line_idx, line in enumerate(ocr_lines):
                            if line.strip():
                                line_text = line[:80] + "..." if len(line) > 80 else line
                                print(f"      Line {line_idx+1}: \"{line_text}\"")
                        if len(payload["ocr_lines"]) > 3:
                            print(f"      ... and {len(payload['ocr_lines'])-3} more lines")
                    
                    elif "image_ref" in payload:
                        print(f"      Image: {payload['image_ref']}")
                        if "ocr_text" in payload and payload["ocr_text"]:
                            text = payload["ocr_text"][:50] + "..." if len(payload["ocr_text"]) > 50 else payload["ocr_text"]
                            print(f"      OCR: \"{text}\"")
                print()  # Empty line between elements
        
        # Print summary statistics
        if len(all_page_stats) > 1:
            print(f"\n=== SUMMARY ACROSS {len(all_page_stats)} PAGES ===")
            print(f"Total elements detected: {total_elements}")
            
            # Aggregate element types across all pages
            all_element_types = {}
            for page_stats in all_page_stats:
                for elem_type, count in page_stats['element_types'].items():
                    all_element_types[elem_type] = all_element_types.get(elem_type, 0) + count
            print(f"Element types across all pages: {all_element_types}")
            
            # Per-page breakdown
            print(f"\nPer-page breakdown:")
            for page_stats in all_page_stats:
                print(f"  Page {page_stats['page_number']}: {page_stats['total_elements']} elements - {page_stats['element_types']}")
        
        pager.close()
        logger.info("Detection test completed successfully")
        
    except Exception as e:
        logger.error(f"Detection test failed: {e}")
        raise


def main():
    """Main entry point for detection testing."""
    parser = argparse.ArgumentParser(description="Test EHR layout detection with visual output")
    parser.add_argument("pdf_path", help="Path to PDF file to test")
    parser.add_argument("--pages", default="1", help="Pages to test (e.g., '1', '1-3', '1,3,5', or 'all', default: '1')")
    parser.add_argument("--output", default="test_output", help="Output directory for visualizations")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR processing on detected elements")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    if not Path(args.pdf_path).exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    try:
        test_detection_on_pdf(args.pdf_path, args.pages, args.output, args.ocr)
        print(f"\nVisualizations saved to: {Path(args.output).absolute()}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()