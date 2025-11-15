"""
Test pipeline on 20-page document with detailed output inspection.

Shows exactly what's extracted without needing a DB or frontend.
"""

import json
import logging
from pathlib import Path

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
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def inspect_extraction_results(document):
    """Show what was extracted from the PDF."""
    print_section("EXTRACTION RESULTS")

    stats = document['processing_stats']
    print(f"📄 Document: {document['source_pdf']}")
    print(f"📊 Pages processed: {stats['processed_pages']}/{stats['total_pages']}")
    print(f"📦 Total elements extracted: {stats['total_elements']}")
    print(f"⏱️  Processing time: {stats['processing_time_seconds']:.1f}s")
    print(f"💰 Total cost: ${stats['total_cost_usd']:.4f}")

    if stats['failed_pages']:
        print(f"⚠️  Failed pages: {stats['failed_pages']}")

    # Show breakdown by element type
    print("\n📋 Elements by Type:")
    type_counts = {}
    for page in document['pages']:
        for element in page.get('elements', []):
            elem_type = element.get('type', 'unknown')
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

    for elem_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {elem_type}: {count}")

    # Show sample elements from first page
    print("\n📄 Sample from Page 1:")
    if document['pages'] and document['pages'][0]['elements']:
        for i, element in enumerate(document['pages'][0]['elements'][:3]):
            print(f"\n  [{i+1}] {element['type']}")
            content = element['content']
            if len(content) > 150:
                content = content[:150] + "..."
            print(f"      Content: {content}")
            print(f"      Confidence: {element['confidence']['overall']:.2f}")
            print(f"      Bbox (pixel): {element['bbox_pixel']}")
            print(f"      Bbox (PDF): {element['bbox_pdf']}")
            print(f"      Needs review: {element['needs_review']}")


def inspect_grouping_results(enhanced_doc):
    """Show how pages were grouped into sub-documents."""
    print_section("SUB-DOCUMENT GROUPING")

    sub_documents = enhanced_doc.get('sub_documents', [])
    print(f"📚 Sub-documents detected: {len(sub_documents)}")

    if enhanced_doc.get('patient_demographics'):
        demo = enhanced_doc['patient_demographics']
        print(f"\n👤 Patient Demographics (Page {demo['page_number']}):")
        print(f"   Confidence: {demo['confidence']:.2f}")
        content = demo['content'][:200] + "..." if len(demo['content']) > 200 else demo['content']
        print(f"   Content: {content}")

    print("\n📑 Sub-Documents:")
    for i, subdoc in enumerate(sub_documents, 1):
        print(f"\n  [{i}] {subdoc['type'].upper()}")
        print(f"      Title: {subdoc['title']}")
        print(f"      Pages: {subdoc['page_range'][0]} - {subdoc['page_range'][1]} ({subdoc['page_count']} pages)")
        print(f"      Elements: {subdoc['element_count']}")
        print(f"      Confidence: {subdoc['confidence']:.2f}")


def inspect_output_files(output_dir, doc_id):
    """Show what files were created and their structure."""
    print_section("OUTPUT FILES")

    output_dir = Path(output_dir)

    files = {
        "Full Extraction": f"{doc_id}_full.json",
        "Enhanced (with sub-docs)": f"{doc_id}_enhanced.json",
        "Hierarchical Index": f"{doc_id}_index.json"
    }

    print("📁 Files created in output/test_20_pages/:\n")

    for name, filename in files.items():
        filepath = output_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename}")
            print(f"    {name}")
            print(f"    Size: {size_kb:.1f} KB")
            print(f"    Path: {filepath}\n")


def run_sample_queries(agent):
    """Run a few sample queries to show the agent works."""
    print_section("SAMPLE QUERIES")

    sample_queries = [
        "What medications is the patient taking?",
        "What were the patient's lab results?",
        "Show me vital signs",
    ]

    for i, query in enumerate(sample_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 80)

        try:
            result = agent.query(query)

            print(f"✓ Matched elements: {len(result['matched_elements'])}")
            print(f"  Filter efficiency: {result['filter_stats']['reduction_ratio']}")

            if result.get('answer_summary'):
                print(f"\n  Answer: {result['answer_summary']}")

            if result['matched_elements']:
                print(f"\n  Sample matches:")
                for elem in result['matched_elements'][:2]:
                    content = elem.get('content', '')[:100]
                    print(f"    • {elem.get('type')}: {content}...")
                    print(f"      Page {elem.get('page_number')}, Bbox: {elem.get('bbox_pixel')}")

        except Exception as e:
            print(f"✗ Error: {e}")


def main():
    """Run 20-page test pipeline."""

    print_section("PDF2EHR 20-PAGE TEST")

    # Configuration
    pdf_path = "SENSITIVE_ehr1_copy.pdf"
    output_dir = Path("output/test_20_pages")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("\nAvailable PDFs in repo:")
        for pdf in Path(".").rglob("*.pdf"):
            print(f"  • {pdf}")
        return

    print(f"📄 Testing with: {pdf_path}")
    print(f"📁 Output directory: {output_dir}")

    # ========================================================================
    # STEP 1: Extract with VLM (first 20 pages)
    # ========================================================================
    print_section("STEP 1: VLM EXTRACTION (20 pages)")

    pipeline = DocumentPipeline(
        vlm_config=VLMConfig.from_env(),
        checkpoint_interval=10,
        dpi=200
    )

    document = pipeline.process_document(
        pdf_path=pdf_path,
        output_dir=str(output_dir),
        page_range="1-20",  # Just first 20 pages
        document_context={"document_type": "Clinical EHR"}
    )

    inspect_extraction_results(document)

    # ========================================================================
    # STEP 2: Group into sub-documents
    # ========================================================================
    print_section("STEP 2: SUB-DOCUMENT GROUPING")

    grouper = SubDocumentGrouper(confidence_threshold=0.80)
    enhanced_doc = grouper.group_document(document)

    # Save enhanced document
    enhanced_path = output_dir / f"{document['document_id']}_enhanced.json"
    with open(enhanced_path, 'w') as f:
        json.dump(enhanced_doc, f, indent=2)

    # Generate and save index
    index = generate_hierarchical_index(enhanced_doc)
    index_path = output_dir / f"{document['document_id']}_index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    inspect_grouping_results(enhanced_doc)

    # ========================================================================
    # STEP 3: Show output files
    # ========================================================================
    inspect_output_files(output_dir, document['document_id'])

    # ========================================================================
    # STEP 4: Test query agent
    # ========================================================================
    print_section("STEP 3: QUERY AGENT TEST")

    print("Initializing query agent...")
    agent = HybridQueryAgent(
        schema_path=str(enhanced_path),
        vlm_config=VLMConfig.from_env()
    )
    print(f"✓ Agent loaded with {agent._count_total_elements()} elements\n")

    run_sample_queries(agent)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("TEST COMPLETE")

    print("✅ All components working!")
    print(f"\n📊 Summary:")
    print(f"  • Extracted {document['processing_stats']['total_elements']} elements from 20 pages")
    print(f"  • Grouped into {len(enhanced_doc.get('sub_documents', []))} sub-documents")
    print(f"  • Query agent ready for natural language queries")
    print(f"  • Total cost: ${document['processing_stats']['total_cost_usd']:.4f}")

    print(f"\n📁 Output files:")
    print(f"  {enhanced_path}")
    print(f"  {index_path}")

    print("\n💡 To inspect JSON outputs:")
    print(f"  cat {enhanced_path} | jq '.sub_documents[0]'")
    print(f"  cat {index_path} | jq '.sub_documents'")

    print("\n🎯 Next: Run on full 650 pages with: python run_mvp_pipeline.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Test failed")
        print(f"\n❌ Test failed: {e}")
