#!/usr/bin/env python
"""
Simple test script for VLM client validation.

Tests VLM processing on a single page image and prints results.
Useful for manual validation and debugging.

Usage:
    python scripts/test_vlm.py /path/to/page.png
    python scripts/test_vlm.py /path/to/page.png --config configs/default.yaml
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ehrx.vlm import VLMClient, VLMConfig, VLMRequest, DocumentContext


def main():
    parser = argparse.ArgumentParser(
        description="Test VLM processing on a single page image"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to page image (PNG, JPG, etc.)"
    )
    parser.add_argument(
        "--document-type",
        type=str,
        default="Clinical Notes",
        help="Document type hint (default: Clinical Notes)"
    )
    parser.add_argument(
        "--page-number",
        type=int,
        default=0,
        help="Page number (0-indexed, default: 0)"
    )
    parser.add_argument(
        "--total-pages",
        type=int,
        default=1,
        help="Total pages in document (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="Gemini model to use (default: gemini-1.5-flash)"
    )
    parser.add_argument(
        "--save-output",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("VLM Client Test Script")
    print("=" * 60)
    print()

    # Initialize VLM client from environment
    print("📋 Initializing VLM client...")
    try:
        config = VLMConfig.from_env()
        # Override model if specified
        if args.model != "gemini-1.5-flash":
            config.model_name = args.model

        print(f"   Project: {config.project_id}")
        print(f"   Location: {config.location}")
        print(f"   Model: {config.model_name}")
        print()

        client = VLMClient(config)
        print("✅ VLM client initialized successfully")
        print()

    except Exception as e:
        print(f"❌ Failed to initialize VLM client: {e}")
        print()
        print("Make sure you have:")
        print("  1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        print("  2. Set GCP_PROJECT_ID environment variable")
        print("  3. Enabled Vertex AI API in your GCP project")
        print()
        print("See docs/GCP_SETUP.md for detailed setup instructions")
        sys.exit(1)

    # Load image
    print(f"🖼️  Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"   Size: {image.size[0]}x{image.size[1]} pixels")
        print(f"   Mode: {image.mode}")
        print()
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        sys.exit(1)

    # Build document context
    context = DocumentContext(
        document_type=args.document_type,
        page_number=args.page_number,
        total_pages=args.total_pages,
    )

    # Create request
    request = VLMRequest(
        image_path=str(image_path),
        context=context,
    )

    # Process with VLM
    print("🔍 Processing with VLM...")
    print(f"   Document type: {args.document_type}")
    print(f"   Page: {args.page_number + 1} of {args.total_pages}")
    print()

    try:
        response = client.detect_elements(image, request)
        print("✅ VLM processing complete")
        print()

    except Exception as e:
        print(f"❌ VLM processing failed: {e}")
        sys.exit(1)

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    print(f"📊 Overall Confidence: {response.overall_confidence:.2%}")
    print(f"👁️  Human Review Required: {response.requires_human_review}")
    if response.review_reasons:
        print(f"   Reasons: {', '.join(response.review_reasons)}")
    print()

    print(f"🔖 Detected Elements: {len(response.elements)}")
    print()

    # Print each element
    for i, element in enumerate(response.elements, 1):
        print(f"Element {i}:")
        print(f"  ID: {element.element_id}")
        print(f"  Type: {element.semantic_type.value}")
        print(f"  Confidence: {element.confidence_scores.overall():.2%}")
        print(f"    - Extraction: {element.confidence_scores.extraction:.2%}")
        print(f"    - Classification: {element.confidence_scores.classification:.2%}")
        print(f"    - Clinical Context: {element.confidence_scores.clinical_context:.2%}")
        print(f"  BBox: [{element.bbox.x0:.0f}, {element.bbox.y0:.0f}, {element.bbox.x1:.0f}, {element.bbox.y1:.0f}]")

        # Print content (truncate if long)
        content = element.content
        if len(content) > 100 and not args.verbose:
            content = content[:100] + "..."
        print(f"  Content: {content}")

        # Print clinical metadata if present
        if element.clinical_metadata:
            cm = element.clinical_metadata
            if cm.temporal_qualifier:
                print(f"  Temporal: {cm.temporal_qualifier}")
            if cm.clinical_domain:
                print(f"  Domain: {cm.clinical_domain}")
            if cm.requires_validation:
                print(f"  ⚠️  Requires Validation: {cm.validation_reason or 'Yes'}")

        print()

    # Print statistics
    stats = client.get_stats()
    print("=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print()
    print(f"Total Requests: {stats['request_count']}")
    print(f"Total Cost: ${stats['total_cost_usd']:.4f}")
    print(f"Avg Cost/Request: ${stats['average_cost_per_request']:.4f}")
    print()

    # Save output if requested
    if args.save_output:
        import json
        output_path = Path(args.save_output)
        output_data = {
            "image_path": str(image_path),
            "context": context.model_dump(),
            "response": {
                "overall_confidence": response.overall_confidence,
                "requires_human_review": response.requires_human_review,
                "review_reasons": response.review_reasons,
                "elements": [elem.to_dict() for elem in response.elements],
            },
            "statistics": stats,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Results saved to: {output_path}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()

    element_types = {}
    for elem in response.elements:
        elem_type = elem.semantic_type.value
        element_types[elem_type] = element_types.get(elem_type, 0) + 1

    print("Element Type Distribution:")
    for elem_type, count in sorted(element_types.items()):
        print(f"  {elem_type}: {count}")
    print()

    low_confidence_count = len(response.low_confidence_elements(threshold=0.85))
    print(f"Low Confidence Elements (<85%): {low_confidence_count}")
    print()

    print("✅ Test complete!")


if __name__ == "__main__":
    main()
