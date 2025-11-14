#!/usr/bin/env python
"""
Test VLM processing using the existing Pager pipeline.

This demonstrates how VLM integrates with the existing PDF processing pipeline.

Usage:
    python scripts/test_vlm_with_pager.py SENSITIVE_ehr1_copy.pdf --page 1
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  Warning: python-dotenv not installed, using system environment variables only")

from ehrx.pdf.pager import Pager
from ehrx.vlm.client import VLMClient
from ehrx.vlm.config import VLMConfig
from ehrx.vlm.models import VLMRequest, DocumentContext


def main():
    parser = argparse.ArgumentParser(
        description="Test VLM processing with existing Pager pipeline"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to PDF file"
    )
    parser.add_argument(
        "--page",
        type=str,
        default="1",
        help="Page number (1-indexed) or range (e.g., '1-3' or 'all')"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for rasterization (default: 200, good for VLM)"
    )
    parser.add_argument(
        "--document-type",
        type=str,
        default="Clinical Notes",
        help="Document type hint"
    )
    parser.add_argument(
        "--save-output",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    print("=" * 70)
    print("VLM Test with Existing Pager Pipeline")
    print("=" * 70)
    print()

    # Initialize VLM client
    print("📋 Initializing VLM client...")
    try:
        config = VLMConfig.from_env()
        print(f"   Project: {config.project_id}")
        print(f"   Model: {config.model_name}")
        client = VLMClient(config)
        print("✅ VLM client ready")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize VLM: {e}")
        print("   Check your .env file and GCP setup")
        sys.exit(1)

    # Initialize Pager
    print(f"📄 Opening PDF: {pdf_path.name}")
    try:
        pager = Pager(pdf_path)
        print(f"   Total pages: {pager.page_count}")
        print(f"   Backend: {pager.rasterizer.backend}")
        print()
    except Exception as e:
        print(f"❌ Failed to open PDF: {e}")
        sys.exit(1)

    # Process pages
    all_results = []

    try:
        for img_array, page_info, mapper in pager.pages(page_range=args.page, dpi=args.dpi):
            page_num_display = page_info.page_number + 1

            print("─" * 70)
            print(f"Processing Page {page_num_display}/{pager.page_count}")
            print("─" * 70)
            print(f"   PDF dimensions: {page_info.width_pdf:.1f} x {page_info.height_pdf:.1f} points")
            print(f"   Image size: {page_info.width_px} x {page_info.height_px} pixels")
            print(f"   DPI: {page_info.dpi}")
            print()

            # Convert numpy array to PIL Image
            image = Image.fromarray(img_array)

            # Create document context
            context = DocumentContext(
                document_type=args.document_type,
                page_number=page_info.page_number,
                total_pages=pager.page_count,
            )

            # Create VLM request
            request = VLMRequest(
                image_path=str(pdf_path),  # Source PDF path
                context=context,
            )

            # Process with VLM
            print("🔍 Processing with VLM...")
            try:
                response = client.detect_elements(image, request)
                print("✅ VLM processing complete")
                print()
            except Exception as e:
                print(f"❌ VLM processing failed: {e}")
                continue

            # Print results
            print("📊 Results:")
            print(f"   Overall confidence: {response.overall_confidence:.2%}")
            print(f"   Human review needed: {response.requires_human_review}")
            print(f"   Elements detected: {len(response.elements)}")
            print()

            # Show element type distribution
            element_types = {}
            for elem in response.elements:
                elem_type = elem.semantic_type.value
                element_types[elem_type] = element_types.get(elem_type, 0) + 1

            print("   Element distribution:")
            for elem_type, count in sorted(element_types.items()):
                print(f"     {elem_type}: {count}")
            print()

            # Show first 5 elements with coordinate conversion
            print("   First 5 elements:")
            for i, elem in enumerate(response.elements[:5], 1):
                # Convert pixel coordinates back to PDF coordinates
                bbox_pixel = [elem.bbox.x0, elem.bbox.y0, elem.bbox.x1, elem.bbox.y1]
                bbox_pdf = mapper.pixel_to_pdf(bbox_pixel)

                content_preview = elem.content[:60] + "..." if len(elem.content) > 60 else elem.content

                print(f"\n   {i}. {elem.semantic_type.value}")
                print(f"      Confidence: {elem.confidence_scores.overall():.2%}")
                print(f"      BBox (pixel): [{elem.bbox.x0:.0f}, {elem.bbox.y0:.0f}, {elem.bbox.x1:.0f}, {elem.bbox.y1:.0f}]")
                print(f"      BBox (PDF):   [{bbox_pdf[0]:.1f}, {bbox_pdf[1]:.1f}, {bbox_pdf[2]:.1f}, {bbox_pdf[3]:.1f}]")
                print(f"      Content: {content_preview}")

            print()

            # Store results
            all_results.append({
                "page_number": page_info.page_number,
                "page_info": {
                    "width_pdf": page_info.width_pdf,
                    "height_pdf": page_info.height_pdf,
                    "width_px": page_info.width_px,
                    "height_px": page_info.height_px,
                    "dpi": page_info.dpi,
                },
                "response": {
                    "overall_confidence": response.overall_confidence,
                    "requires_human_review": response.requires_human_review,
                    "review_reasons": response.review_reasons,
                    "element_count": len(response.elements),
                    "elements": [
                        {
                            "id": elem.element_id,
                            "type": elem.semantic_type.value,
                            "content": elem.content,
                            "confidence": {
                                "overall": elem.confidence_scores.overall(),
                                "extraction": elem.confidence_scores.extraction,
                                "classification": elem.confidence_scores.classification,
                            },
                            "bbox_pixel": [elem.bbox.x0, elem.bbox.y0, elem.bbox.x1, elem.bbox.y1],
                            "bbox_pdf": mapper.pixel_to_pdf([elem.bbox.x0, elem.bbox.y0, elem.bbox.x1, elem.bbox.y1]),
                        }
                        for elem in response.elements
                    ],
                },
            })

    finally:
        pager.close()

    # Print overall statistics
    print("=" * 70)
    print("Overall Statistics")
    print("=" * 70)

    stats = client.get_stats()
    print(f"Pages processed: {len(all_results)}")
    print(f"Total elements: {sum(r['response']['element_count'] for r in all_results)}")
    print(f"Total cost: ${stats['total_cost_usd']:.4f}")
    print(f"Avg cost/page: ${stats['average_cost_per_request']:.4f}")
    print()

    # Save output if requested
    if args.save_output:
        import json
        output_path = Path(args.save_output)

        output_data = {
            "pdf_path": str(pdf_path),
            "document_type": args.document_type,
            "dpi": args.dpi,
            "pages": all_results,
            "statistics": stats,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Results saved to: {output_path}")
        print()

    print("✅ Test complete!")


if __name__ == "__main__":
    main()
