#!/usr/bin/env python3
"""
Run MVP Pipeline on SENSITIVE_ehr2_copy.pdf
Processes the full document and enables interactive querying.
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
from ehrx.agent.query import HybridQueryAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ehr2_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run complete MVP pipeline on SENSITIVE_ehr2_copy.pdf."""

    print("\n" + "=" * 80)
    print("PDF2EHR MVP PIPELINE - SENSITIVE_ehr2_copy.pdf")
    print("=" * 80)

    # Configuration
    pdf_path = Path("SENSITIVE_ehr2_copy.pdf")
    
    if not pdf_path.exists():
        print(f"❌ Error: PDF not found at {pdf_path}")
        print("   Please ensure SENSITIVE_ehr2_copy.pdf is in the current directory")
        return

    # Ask for page range
    print(f"\n📄 Found PDF: {pdf_path}")
    page_range_input = input("\nEnter page range (e.g., '1-20', 'all' for full document): ").strip()
    page_range = page_range_input if page_range_input else "all"

    output_dir = Path("output/ehr2_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STAGE 1: Multi-Page Processing
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: MULTI-PAGE VLM EXTRACTION")
    print("=" * 80)
    print(f"\n⏳ Processing {page_range} pages from {pdf_path.name}...")
    print("   This may take several minutes for large documents...")

    try:
        pipeline = DocumentPipeline(
            vlm_config=VLMConfig.from_env(),
            checkpoint_interval=50,
            dpi=200
        )

        logger.info(f"Processing PDF: {pdf_path}")
        logger.info(f"Page range: {page_range}")

        document = pipeline.process_document(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir),
            page_range=page_range,
            document_context={"document_type": "Clinical EHR"}
        )

        stats = document['processing_stats']
        print(f"\n✅ Extraction Complete!")
        print(f"   Pages processed: {stats['processed_pages']}/{stats['total_pages']}")
        print(f"   Total elements: {stats['total_elements']}")
        print(f"   Processing time: {stats['processing_time_seconds']:.1f}s")
        print(f"   Total cost: ${stats['total_cost_usd']:.4f}")

        if stats['failed_pages']:
            print(f"   ⚠️  Failed pages: {stats['failed_pages']}")

    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        logger.exception("Stage 1 failed")
        return

    # ========================================================================
    # STAGE 2: Sub-Document Grouping
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: SUB-DOCUMENT GROUPING")
    print("=" * 80)

    try:
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

        print(f"\n✅ Grouping Complete!")
        print(f"   Sub-documents detected: {len(enhanced_doc.get('sub_documents', []))}")

        if enhanced_doc.get('sub_documents'):
            print(f"\n   📑 Sub-Documents:")
            for subdoc in enhanced_doc['sub_documents']:
                print(f"      • {subdoc['type']}: {subdoc['title']}")
                print(f"        Pages {subdoc['page_range'][0]}-{subdoc['page_range'][1]} ({subdoc['page_count']} pages)")
                print(f"        {subdoc['element_count']} elements")

        print(f"\n   💾 Files saved:")
        print(f"      • Full schema: {enhanced_path}")
        print(f"      • Index: {index_path}")

    except Exception as e:
        print(f"\n❌ Grouping failed: {e}")
        logger.exception("Stage 2 failed")
        return

    # ========================================================================
    # STAGE 3: Query Agent Setup
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 3: HYBRID QUERY AGENT")
    print("=" * 80)

    try:
        logger.info("Initializing query agent")
        agent = HybridQueryAgent(
            schema_path=str(enhanced_path),
            vlm_config=VLMConfig.from_env()
        )

        print(f"\n✅ Agent initialized with schema")
        print(f"   Total elements available: {agent._count_total_elements()}")

    except Exception as e:
        print(f"\n❌ Agent initialization failed: {e}")
        logger.exception("Stage 3 failed")
        return

    # ========================================================================
    # STAGE 4: Interactive Queries
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 4: INTERACTIVE QUERY MODE")
    print("=" * 80)

    print("\n🔍 You can now query the EHR using natural language!")
    print("\n   Example queries:")
    print("   • What were the patient's blood test results?")
    print("   • What medications is the patient taking?")
    print("   • Show me all vital signs")
    print("   • What is the patient's diagnosis?")
    print("   • When was the patient admitted?")
    print("\n   Commands:")
    print("   • Type 'examples' to run pre-defined example queries")
    print("   • Type 'stats' to see document statistics")
    print("   • Type 'quit' or 'exit' to exit")

    while True:
        print("\n" + "-" * 80)
        query = input("\n💬 Query: ").strip()

        if not query:
            continue

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting query mode...")
            break

        if query.lower() == 'stats':
            print(f"\n📊 Document Statistics:")
            print(f"   • Total pages: {stats['total_pages']}")
            print(f"   • Processed pages: {stats['processed_pages']}")
            print(f"   • Total elements: {stats['total_elements']}")
            print(f"   • Sub-documents: {len(enhanced_doc.get('sub_documents', []))}")
            continue

        if query.lower() == 'examples':
            print("\n🔄 Running example queries...")
            from ehrx.agent.query import run_example_queries
            run_example_queries(agent)
            continue

        try:
            print("\n⏳ Processing query...")
            result = agent.query(query)

            print(f"\n✅ Query Results")
            print(f"   • Matched elements: {len(result['matched_elements'])}")
            print(f"   • Filter efficiency: {result['filter_stats']['reduction_ratio']:.2%}")

            if 'answer_summary' in result and result['answer_summary']:
                print(f"\n💡 Answer:")
                print(f"   {result.get('answer_summary', '')}")

            if result.get('reasoning'):
                print(f"\n🧠 Reasoning:")
                print(f"   {result.get('reasoning', '')}")

            if result['matched_elements']:
                print(f"\n📄 Matched Elements (showing top 5):")
                for i, elem in enumerate(result['matched_elements'][:5], 1):
                    print(f"\n   [{i}] {elem.get('type', 'unknown').upper()}")
                    content = elem.get('content', '')
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"       Content: {content}")
                    print(f"       Page: {elem.get('page_number', 'N/A')}")
                    
                    bbox = elem.get('bbox_pixel')
                    if bbox:
                        print(f"       Location: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

                if len(result['matched_elements']) > 5:
                    print(f"\n   ... and {len(result['matched_elements']) - 5} more elements")
            else:
                print(f"\n⚠️  No matching elements found for this query")

        except Exception as e:
            print(f"\n❌ Query failed: {e}")
            logger.exception("Query failed")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n📁 All outputs saved to: {output_dir}")
    print("\n📋 Files generated:")
    print(f"   1. {document['document_id']}_full.json - Complete extraction")
    print(f"   2. {document['document_id']}_enhanced.json - With sub-documents")
    print(f"   3. {document['document_id']}_index.json - Hierarchical index")
    print(f"   4. ehr2_pipeline.log - Processing log")
    print("\n✨ Thank you for using PDF2EHR!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)




