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
from ehrx.core.config import load_default_config, setup_logging_from_config


def test_detection_on_pdf(pdf_path: str, page_num: int = 0, output_dir: str = "test_output"):
    """Test layout detection on a specific PDF page with visualization.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number to test (0-indexed)
        output_dir: Directory to save output images
    """
    try:
        # Load configuration
        config = load_default_config()
        logger = setup_logging_from_config(config, level="INFO")
        
        logger.info(f"Testing layout detection on: {pdf_path}")
        logger.info(f"Page: {page_num + 1}")
        
        # Initialize detector
        detector = LayoutDetector(config.detector)
        
        # Initialize pager
        pager = Pager(pdf_path)
        logger.info(f"PDF has {pager.page_count} pages")
        
        if page_num >= pager.page_count:
            raise ValueError(f"Page {page_num + 1} exceeds PDF page count ({pager.page_count})")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Process single page
        for i, (image, page_info, mapper) in enumerate(pager.pages(f"{page_num + 1}", dpi=200)):
            if i > 0:  # Only process first (and only) page from range
                break
                
            logger.info(f"Processing page {page_info.page_number + 1}")
            logger.info(f"Image shape: {image.shape}")
            
            # Run layout detection
            layout = detector.detect_layout(image)
            
            # Get detection stats
            stats = detector.get_detection_stats(layout)
            logger.info(f"Detection stats: {stats}")
            
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
                colors = ['red', 'blue', 'green', 'orange', 'purple']
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
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    confidence = f" ({block.score:.2f})" if hasattr(block, 'score') else ""
                    ax.text(x1, y1 - 5, f"{block_type}{confidence}", 
                           color=color, fontsize=8, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
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
        
        pager.close()
        logger.info("Detection test completed successfully")
        
    except Exception as e:
        logger.error(f"Detection test failed: {e}")
        raise


def main():
    """Main entry point for detection testing."""
    parser = argparse.ArgumentParser(description="Test EHR layout detection with visual output")
    parser.add_argument("pdf_path", help="Path to PDF file to test")
    parser.add_argument("--page", type=int, default=1, help="Page number to test (1-indexed, default: 1)")
    parser.add_argument("--output", default="test_output", help="Output directory for visualizations")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Convert to 0-indexed
    page_num = args.page - 1
    
    if not Path(args.pdf_path).exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    try:
        test_detection_on_pdf(args.pdf_path, page_num, args.output)
        print(f"\nVisualizations saved to: {Path(args.output).absolute()}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()