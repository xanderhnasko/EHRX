"""
Simple JSON result viewer for inspecting pipeline outputs without frontend.

Usage:
    python view_results.py output/test_20_pages/<doc_id>_enhanced.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def view_document_overview(data: Dict[str, Any]):
    """Show high-level document overview."""
    print_header("DOCUMENT OVERVIEW")

    print(f"Document ID: {data.get('document_id')}")
    print(f"Source PDF: {data.get('source_pdf')}")
    print(f"Total Pages: {data.get('total_pages')}")

    # Demographics
    if data.get('patient_demographics'):
        demo = data['patient_demographics']
        print(f"\n👤 Patient Demographics:")
        print(f"   Page: {demo['page_number']}")
        print(f"   Confidence: {demo['confidence']:.2%}")
        content = demo['content'][:200] + "..." if len(demo['content']) > 200 else demo['content']
        print(f"   {content}")

    # Processing stats
    if data.get('processing_stats'):
        stats = data['processing_stats']
        print(f"\n📊 Processing Stats:")
        print(f"   Processed: {stats['processed_pages']}/{stats['total_pages']} pages")
        print(f"   Total elements: {stats['total_elements']}")
        print(f"   Processing time: {stats['processing_time_seconds']:.1f}s")
        print(f"   Cost: ${stats['total_cost_usd']:.4f}")


def view_subdocuments(data: Dict[str, Any]):
    """Show sub-document structure."""
    print_header("SUB-DOCUMENTS")

    sub_documents = data.get('sub_documents', [])
    print(f"Total sub-documents: {len(sub_documents)}\n")

    for i, subdoc in enumerate(sub_documents, 1):
        print(f"[{i}] {subdoc['type'].upper()}")
        print(f"    Title: {subdoc['title']}")
        print(f"    Pages: {subdoc['page_range'][0]} - {subdoc['page_range'][1]} ({subdoc['page_count']} pages)")
        print(f"    Elements: {subdoc['element_count']}")
        print(f"    Confidence: {subdoc['confidence']:.2%}")
        print()


def view_element_breakdown(data: Dict[str, Any]):
    """Show breakdown of element types."""
    print_header("ELEMENT TYPE BREAKDOWN")

    # Count elements by type
    type_counts = {}
    total_elements = 0

    if 'sub_documents' in data:
        for subdoc in data['sub_documents']:
            for page in subdoc.get('pages', []):
                for element in page.get('elements', []):
                    elem_type = element.get('type', 'unknown')
                    type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
                    total_elements += 1
    else:
        for page in data.get('pages', []):
            for element in page.get('elements', []):
                elem_type = element.get('type', 'unknown')
                type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
                total_elements += 1

    print(f"Total elements: {total_elements}\n")

    for elem_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_elements * 100) if total_elements > 0 else 0
        bar_length = int(percentage / 2)  # Scale to fit
        bar = "█" * bar_length
        print(f"  {elem_type:30s} {count:4d} {bar} {percentage:.1f}%")


def view_sample_elements(data: Dict[str, Any], subdoc_idx: int = 0, count: int = 5):
    """Show sample elements from a sub-document."""
    print_header(f"SAMPLE ELEMENTS (Sub-document {subdoc_idx + 1})")

    if 'sub_documents' in data:
        sub_documents = data.get('sub_documents', [])
        if subdoc_idx >= len(sub_documents):
            print(f"Sub-document {subdoc_idx} not found")
            return

        subdoc = sub_documents[subdoc_idx]
        print(f"From: {subdoc['type']} - {subdoc['title']}\n")

        # Get first few elements
        shown = 0
        for page in subdoc.get('pages', []):
            for element in page.get('elements', []):
                if shown >= count:
                    break

                print(f"[{shown + 1}] {element['type']}")
                content = element['content']
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"    Content: {content}")
                print(f"    Page: {page.get('page_number')}")
                print(f"    Confidence: {element['confidence']['overall']:.2%}")
                print(f"    Bbox (pixel): {element.get('bbox_pixel')}")
                print(f"    Bbox (PDF): {element.get('bbox_pdf')}")
                print(f"    Needs review: {element.get('needs_review')}")
                print()

                shown += 1

            if shown >= count:
                break
    else:
        # Flat page structure
        print("Showing first page elements:\n")
        pages = data.get('pages', [])
        if pages:
            for i, element in enumerate(pages[0].get('elements', [])[:count]):
                print(f"[{i + 1}] {element['type']}")
                content = element['content']
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"    Content: {content}")
                print(f"    Confidence: {element['confidence']['overall']:.2%}")
                print(f"    Bbox (pixel): {element.get('bbox_pixel')}")
                print()


def search_elements(data: Dict[str, Any], query: str):
    """Search for elements containing specific text."""
    print_header(f"SEARCH RESULTS: '{query}'")

    matches = []

    if 'sub_documents' in data:
        for subdoc in data['sub_documents']:
            for page in subdoc.get('pages', []):
                for element in page.get('elements', []):
                    if query.lower() in element.get('content', '').lower():
                        matches.append({
                            'element': element,
                            'page': page.get('page_number'),
                            'subdoc': subdoc.get('title')
                        })
    else:
        for page in data.get('pages', []):
            for element in page.get('elements', []):
                if query.lower() in element.get('content', '').lower():
                    matches.append({
                        'element': element,
                        'page': page.get('page_number')
                    })

    print(f"Found {len(matches)} matches\n")

    for i, match in enumerate(matches[:10], 1):  # Show first 10
        element = match['element']
        print(f"[{i}] {element['type']} (Page {match['page']})")
        if 'subdoc' in match:
            print(f"    Sub-document: {match['subdoc']}")
        content = element['content']
        if len(content) > 150:
            content = content[:150] + "..."
        print(f"    Content: {content}")
        print(f"    Bbox: {element.get('bbox_pixel')}")
        print()

    if len(matches) > 10:
        print(f"... and {len(matches) - 10} more matches")


def interactive_viewer(data: Dict[str, Any]):
    """Interactive viewer with commands."""
    print_header("INTERACTIVE VIEWER")

    print("Commands:")
    print("  overview    - Show document overview")
    print("  subdocs     - List sub-documents")
    print("  elements    - Show element type breakdown")
    print("  sample [N]  - Show sample elements from sub-document N (default 0)")
    print("  search TEXT - Search for text in elements")
    print("  quit        - Exit")

    while True:
        print("\n" + "-" * 80)
        cmd = input("\nCommand: ").strip().lower()

        if not cmd:
            continue

        if cmd in ['quit', 'exit', 'q']:
            break

        elif cmd == 'overview':
            view_document_overview(data)

        elif cmd == 'subdocs':
            view_subdocuments(data)

        elif cmd == 'elements':
            view_element_breakdown(data)

        elif cmd.startswith('sample'):
            parts = cmd.split()
            idx = int(parts[1]) if len(parts) > 1 else 0
            view_sample_elements(data, subdoc_idx=idx)

        elif cmd.startswith('search'):
            query = ' '.join(cmd.split()[1:])
            if query:
                search_elements(data, query)
            else:
                print("Usage: search <text>")

        else:
            print(f"Unknown command: {cmd}")


def main():
    """View JSON results."""

    if len(sys.argv) < 2:
        print("Usage: python view_results.py <path_to_json>")
        print("\nExample:")
        print("  python view_results.py output/test_20_pages/<doc_id>_enhanced.json")
        return

    json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return

    print(f"Loading: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Show automatic views
    view_document_overview(data)
    view_subdocuments(data)
    view_element_breakdown(data)
    view_sample_elements(data, subdoc_idx=0, count=3)

    # Enter interactive mode
    interactive_viewer(data)


if __name__ == "__main__":
    main()
