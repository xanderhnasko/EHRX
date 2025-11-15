"""
Complete MVP Pipeline: PDF → Schema → Query Interface

End-to-end demonstration of:
1. Multi-page processing (650 pages)
2. Sub-document grouping
3. Hybrid query agent
"""

import json
import logging
from pathlib import Path
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from ehrx.vlm.pipeline import DocumentPipeline
from ehrx.vlm.grouping import SubDocumentGrouper, generate_hierarchical_index
from ehrx.vlm.config import VLMConfig
from ehrx.agent.query import HybridQueryAgent, run_example_queries


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mvp_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run complete MVP pipeline."""

    print("\n" + "=" * 80)
    print("PDF2EHR MVP PIPELINE")
    print("=" * 80)

    # Configuration
    pdf_path = input("\nEnter path to PDF file (or press Enter for test mode): ").strip()

    if not pdf_path:
        print("\nTest mode: Using sample 5-page extraction")
        test_mode = True
        pdf_path = input("Enter path to test PDF: ").strip()
        page_range = "1-5"
    else:
        test_mode = False
        page_range = input("Enter page range (default 'all'): ").strip() or "all"

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    output_dir = Path("output/mvp_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STAGE 1: Multi-Page Processing
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: MULTI-PAGE VLM EXTRACTION")
    print("=" * 80)

    pipeline = DocumentPipeline(
        vlm_config=VLMConfig.from_env(),
        checkpoint_interval=50,
        dpi=200
    )

    logger.info(f"Processing PDF: {pdf_path}")
    logger.info(f"Page range: {page_range}")

    document = pipeline.process_document(
        pdf_path=pdf_path,
        output_dir=str(output_dir),
        page_range=page_range,
        document_context={"document_type": "Clinical EHR"}
    )

    stats = document['processing_stats']
    print(f"\n✓ Extraction Complete!")
    print(f"  Pages processed: {stats['processed_pages']}/{stats['total_pages']}")
    print(f"  Total elements: {stats['total_elements']}")
    print(f"  Processing time: {stats['processing_time_seconds']:.1f}s")
    print(f"  Total cost: ${stats['total_cost_usd']:.4f}")

    if stats['failed_pages']:
        print(f"  ⚠ Failed pages: {stats['failed_pages']}")

    # ========================================================================
    # STAGE 2: Sub-Document Grouping
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: SUB-DOCUMENT GROUPING")
    print("=" * 80)

    grouper = SubDocumentGrouper(confidence_threshold=0.80)

    logger.info("Grouping pages into sub-documents")
    enhanced_doc = grouper.group_document(document)

    # Generate hierarchical index
    index = generate_hierarchical_index(enhanced_doc)

    # Save enhanced document and index
    enhanced_path = output_dir / f"{document['document_id']}_enhanced.json"
    index_path = output_dir / f"{document['document_id']}_index.json"

    with open(enhanced_path, 'w') as f:
        json.dump(enhanced_doc, f, indent=2)

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\n✓ Grouping Complete!")
    print(f"  Sub-documents detected: {len(enhanced_doc.get('sub_documents', []))}")

    if enhanced_doc.get('sub_documents'):
        print(f"\n  Sub-Documents:")
        for subdoc in enhanced_doc['sub_documents']:
            print(f"    - {subdoc['type']}: {subdoc['title']}")
            print(f"      Pages {subdoc['page_range'][0]}-{subdoc['page_range'][1]} ({subdoc['page_count']} pages)")
            print(f"      {subdoc['element_count']} elements")

    print(f"\n  Files saved:")
    print(f"    - Full schema: {enhanced_path}")
    print(f"    - Index: {index_path}")

    # ========================================================================
    # STAGE 3: Query Agent Setup
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 3: HYBRID QUERY AGENT")
    print("=" * 80)

    logger.info("Initializing query agent")
    agent = HybridQueryAgent(
        schema_path=str(enhanced_path),
        vlm_config=VLMConfig.from_env()
    )

    print(f"\n✓ Agent initialized with schema")
    print(f"  Total elements available: {agent._count_total_elements()}")

    # ========================================================================
    # STAGE 4: Interactive Queries
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 4: INTERACTIVE QUERY MODE")
    print("=" * 80)

    print("\nYou can now query the EHR schema using natural language.")
    print("Examples:")
    print("  - What were the patient's blood test results?")
    print("  - What medications is the patient taking?")
    print("  - Show me all vital signs")
    print("\nType 'examples' to run example queries")
    print("Type 'quit' to exit")

    while True:
        print("\n" + "-" * 80)
        query = input("\nQuery: ").strip()

        if not query:
            continue

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if query.lower() == 'examples':
            run_example_queries(agent)
            continue

        try:
            print("\nProcessing query...")
            result = agent.query(query)

            print(f"\n✓ Query Results")
            print(f"  Matched elements: {len(result['matched_elements'])}")
            print(f"  Filter efficiency: {result['filter_stats']['reduction_ratio']}")

            if 'answer_summary' in result:
                print(f"\nAnswer: {result.get('answer_summary', '')}")

            print(f"\nReasoning: {result.get('reasoning', '')}")

            if result['matched_elements']:
                print(f"\nMatched Elements:")
                for i, elem in enumerate(result['matched_elements'][:5], 1):  # Show top 5
                    print(f"\n  [{i}] {elem.get('type', 'unknown')}")
                    content = elem.get('content', '')
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"      Content: {content}")
                    print(f"      Page: {elem.get('page_number', 'N/A')}")
                    print(f"      Bbox (pixel): {elem.get('bbox_pixel', 'N/A')}")

                if len(result['matched_elements']) > 5:
                    print(f"\n  ... and {len(result['matched_elements']) - 5} more elements")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            logger.exception("Query failed")

    print("\n" + "=" * 80)
    print("MVP PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nFiles generated:")
    print(f"  1. {document['document_id']}_full.json - Complete extraction")
    print(f"  2. {document['document_id']}_enhanced.json - With sub-documents")
    print(f"  3. {document['document_id']}_index.json - Hierarchical index")
    print(f"  4. mvp_pipeline.log - Processing log")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"\n✗ Pipeline failed: {e}")
        sys.exit(1)
