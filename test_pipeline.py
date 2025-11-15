"""
Test script for multi-page processing pipeline.

Run this to validate the pipeline on a small sample before processing 650 pages.
"""

import logging
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from ehrx.vlm.pipeline import DocumentPipeline
from ehrx.vlm.config import VLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Test pipeline on sample pages."""

    # Find the 650-page PDF (update this path as needed)
    pdf_path = input("Enter path to PDF file: ").strip()

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    # Output directory
    output_dir = Path("output/test_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline with default config (from environment)
    pipeline = DocumentPipeline(
        vlm_config=VLMConfig.from_env(),
        checkpoint_interval=10,  # Checkpoint every 10 pages for testing
        dpi=200
    )

    # Test on first 5 pages
    print("\n=== Testing on first 5 pages ===\n")
    result = pipeline.process_document(
        pdf_path=pdf_path,
        output_dir=str(output_dir),
        page_range="1-5",  # Just first 5 pages for testing
        document_context={"document_type": "Clinical EHR"}
    )

    print("\n=== Processing Complete! ===")
    print(f"Document ID: {result['document_id']}")
    print(f"Pages processed: {result['processing_stats']['processed_pages']}")
    print(f"Total elements: {result['processing_stats']['total_elements']}")
    print(f"Total cost: ${result['processing_stats']['total_cost_usd']:.4f}")
    print(f"Processing time: {result['processing_stats']['processing_time_seconds']:.1f}s")
    print(f"\nOutput saved to: {output_dir}")

    # Show sample elements from first page
    if result['pages'] and result['pages'][0]['elements']:
        print("\n=== Sample Elements from Page 1 ===")
        for i, element in enumerate(result['pages'][0]['elements'][:3]):
            print(f"\nElement {i+1}:")
            print(f"  Type: {element['type']}")
            print(f"  Content: {element['content'][:100]}...")  # First 100 chars
            print(f"  Confidence: {element['confidence']['overall']:.2f}")
            print(f"  Bbox (pixel): {element['bbox_pixel']}")

    return result


if __name__ == "__main__":
    main()
