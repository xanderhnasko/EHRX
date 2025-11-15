"""
Test query agent using existing extraction results.

Uses pre-existing _full.json to avoid re-processing.
"""

import json
import logging
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from ehrx.vlm.grouping import SubDocumentGrouper, generate_hierarchical_index
from ehrx.vlm.config import VLMConfig
from ehrx.agent.query import HybridQueryAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_queries_on_existing_extraction():
    """Test query agent on pre-existing extraction results."""

    print("\n" + "=" * 80)
    print("  QUERY AGENT TEST (Using Existing Extraction)")
    print("=" * 80)

    # Ask user for file path
    default_path = "output/test_20_pages/SENSITIVE_ehr1_copy_1763164390_full.json"

    print(f"\nDefault: {default_path}")
    user_input = input("Enter path to _full.json (or press Enter for default): ").strip()

    if user_input:
        full_json_path = Path(user_input)
    else:
        full_json_path = Path(default_path)

    if not full_json_path.exists():
        print(f"\n❌ File not found: {full_json_path}")

        # Show available files
        output_dir = Path("output/test_20_pages")
        if output_dir.exists():
            available = list(output_dir.glob("*_full.json"))
            if available:
                print(f"\nAvailable files in {output_dir}:")
                for f in available:
                    print(f"  • {f.name}")
        return

    print(f"\n📄 Using extraction: {full_json_path.name}")

    # Load the full extraction
    with open(full_json_path, 'r') as f:
        document = json.load(f)

    print(f"   Pages: {document['total_pages']}")
    print(f"   Elements: {document['processing_stats']['total_elements']}")

    # Check if we have enhanced version with sub-documents
    enhanced_json_path = full_json_path.parent / full_json_path.name.replace("_full.json", "_enhanced.json")

    if enhanced_json_path.exists():
        print(f"\n✓ Found existing enhanced.json, using that")
        with open(enhanced_json_path, 'r') as f:
            enhanced_doc = json.load(f)
    else:
        # Group sub-documents if not already done
        print(f"\n📚 Grouping sub-documents...")
        grouper = SubDocumentGrouper(confidence_threshold=0.80)
        enhanced_doc = grouper.group_document(document)

        # Save enhanced version
        with open(enhanced_json_path, 'w') as f:
            json.dump(enhanced_doc, f, indent=2)
        print(f"   Saved: {enhanced_json_path.name}")

    sub_docs = enhanced_doc.get('sub_documents', [])
    print(f"   Sub-documents: {len(sub_docs)}")

    # Initialize query agent
    print("\n" + "=" * 80)
    print("  INITIALIZING QUERY AGENT")
    print("=" * 80)

    agent = HybridQueryAgent(
        schema_path=str(enhanced_json_path),
        vlm_config=VLMConfig.from_env()
    )

    print(f"\n✓ Agent initialized")
    print(f"   Total elements: {agent._count_total_elements()}")

    # Test queries
    print("\n" + "=" * 80)
    print("  RUNNING TEST QUERIES")
    print("=" * 80)

    test_queries = [
        "What medications is the patient taking?",
        "What were the patient's blood test results?",
        "Show me the patient's vital signs",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print("=" * 80)

        try:
            result = agent.query(query)

            # Show answer first (most important!)
            if result.get('answer_summary'):
                print(f"\n{'=' * 80}")
                print(f"📝 ANSWER")
                print(f"{'=' * 80}")
                print(f"\n{result['answer_summary']}\n")

            print(f"\n📊 Query Statistics:")
            print(f"   Matched elements: {len(result['matched_elements'])}")
            print(f"   Filter efficiency: {result['filter_stats']['reduction_ratio']}")
            print(f"   Elements filtered: {result['filter_stats']['filtered_elements']} / {result['filter_stats']['original_elements']}")

            if result.get('reasoning'):
                print(f"\n💭 How the answer was found:")
                print(f"   {result['reasoning']}")

            if result['matched_elements']:
                print(f"\n📋 Matched Elements (showing first 3):")
                for j, elem in enumerate(result['matched_elements'][:3], 1):
                    print(f"\n   [{j}] {elem.get('type', 'unknown')}")
                    content = elem.get('content', '')
                    if len(content) > 150:
                        content = content[:150] + "..."
                    print(f"       Content: {content}")
                    print(f"       Page: {elem.get('page_number', 'N/A')}")
                    print(f"       Bbox (pixel): {elem.get('bbox_pixel', 'N/A')}")
                    if elem.get('relevance'):
                        print(f"       Relevance: {elem['relevance']}")

                if len(result['matched_elements']) > 3:
                    print(f"\n   ... and {len(result['matched_elements']) - 3} more elements")

        except Exception as e:
            print(f"\n❌ Query failed: {e}")
            logger.exception("Query error")

    # Interactive mode
    print("\n" + "=" * 80)
    print("  INTERACTIVE MODE")
    print("=" * 80)
    print("\nEnter your own queries (or 'quit' to exit):")

    while True:
        print("\n" + "-" * 80)
        query = input("\nQuery: ").strip()

        if not query:
            continue

        if query.lower() in ['quit', 'exit', 'q']:
            break

        try:
            result = agent.query(query)

            # Show answer prominently
            if result.get('answer_summary'):
                print(f"\n{'=' * 80}")
                print(f"ANSWER:")
                print(f"{'=' * 80}")
                print(f"{result['answer_summary']}")
                print(f"{'=' * 80}\n")

            print(f"📊 Stats: {len(result['matched_elements'])} elements matched, {result['filter_stats']['reduction_ratio']} filter efficiency")

            if result['matched_elements']:
                print(f"\n📍 Source Provenance:")
                for i, elem in enumerate(result['matched_elements'][:3], 1):
                    print(f"  {i}. [{elem.get('type')}] Page {elem.get('page_number')}, Element {elem.get('element_id')}")
                    print(f"     Bbox: {elem.get('bbox_pixel')}")
                if len(result['matched_elements']) > 3:
                    print(f"  ... and {len(result['matched_elements']) - 3} more sources")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 80)
    print("  TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_queries_on_existing_extraction()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.exception("Test failed")
        print(f"\n❌ Test failed: {e}")
