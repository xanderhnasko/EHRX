"""
Minimal end-to-end demo using an existing extraction JSON.

Usage:
    python run_sample_e2e.py \
        --full-json output/test_20_pages/SENSITIVE_ehr1_copy_1763164390_full.json

If --full-json is omitted, the default above is attempted.
Requires: GCP creds + env vars for VLMConfig (same as other scripts).
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from ehrx.vlm.grouping import SubDocumentGrouper, generate_hierarchical_index
from ehrx.vlm.config import VLMConfig
from ehrx.agent.query import HybridQueryAgent


def load_or_group(full_json_path: Path) -> Path:
    """Load enhanced schema if present, else group and save one."""
    enhanced_path = full_json_path.with_name(
        full_json_path.name.replace("_full.json", "_enhanced.json")
    )

    if enhanced_path.exists():
        print(f"Found existing enhanced schema: {enhanced_path}")
        return enhanced_path

    print("No enhanced schema found; grouping sub-documents...")
    with open(full_json_path, "r") as f:
        document = json.load(f)

    grouper = SubDocumentGrouper(confidence_threshold=0.80)
    enhanced_doc = grouper.group_document(document)

    with open(enhanced_path, "w") as f:
        json.dump(enhanced_doc, f, indent=2)

    index_path = full_json_path.with_name(
        full_json_path.name.replace("_full.json", "_index.json")
    )
    with open(index_path, "w") as f:
        json.dump(generate_hierarchical_index(enhanced_doc), f, indent=2)

    print(f"Saved enhanced schema: {enhanced_path}")
    print(f"Saved index: {index_path}")
    return enhanced_path


def run_queries(schema_path: Path) -> None:
    """Run a few canned queries and show provenance for top hits."""
    agent = HybridQueryAgent(schema_path=str(schema_path), vlm_config=VLMConfig.from_env())

    sample_queries = [
        "What medications is the patient taking?",
        "What were the patient's blood test results?",
    ]

    for query in sample_queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("=" * 80)
        result = agent.query(query)
        print(f"Answer:\n{result.get('answer_summary', 'No answer')}\n")
        print(f"Reasoning: {result.get('reasoning', '')}\n")
        print(f"Matched elements: {len(result['matched_elements'])}")
        for elem in result["matched_elements"][:3]:
            print(
                f"- {elem.get('type')} | Page {elem.get('page_number')} "
                f"| bbox {elem.get('bbox_pixel')} | id {elem.get('element_id')}"
            )
            content = elem.get("content", "")
            if len(content) > 120:
                content = content[:120] + "..."
            print(f"  Content: {content}")


def main():
    # Ensure .env is loaded so VLMConfig.from_env sees GCP_PROJECT_ID, etc.
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run sample E2E query on existing extraction.")
    parser.add_argument(
        "--full-json",
        default="output/test_20_pages/SENSITIVE_ehr1_copy_1763164390_full.json",
        help="Path to *_full.json extraction (default: %(default)s)",
    )
    args = parser.parse_args()

    full_json_path = Path(args.full_json)
    if not full_json_path.exists():
        print(f"Full extraction not found: {full_json_path}")
        print("Provide --full-json pointing to an existing *_full.json file.")
        return

    enhanced_path = load_or_group(full_json_path)
    try:
        run_queries(enhanced_path)
    except Exception as exc:
        print(f"Failed to run queries: {exc}")
        print("Ensure GCP credentials and project ID are available (GCP_PROJECT_ID / GOOGLE_CLOUD_PROJECT).")


if __name__ == "__main__":
    main()
