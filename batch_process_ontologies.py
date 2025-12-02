#!/usr/bin/env python3
"""
Batch process PDFs to generate consolidated JSON ontologies using the MVP pipeline.

Usage:
  python3 batch_process_ontologies.py --in SampleEHR_docs --out SampleEHR_ontologies

Behavior:
- Processes each PDF as a single batch (all pages) to build a single ontology JSON
- Writes exactly one file per input PDF: <pdf_stem>.json in the output directory
- Reads VLM settings from .env via VLMConfig.from_env()
"""
import argparse
import json
import logging
from pathlib import Path
import sys

from dotenv import load_dotenv
load_dotenv()

from ehrx.vlm.pipeline import DocumentPipeline
from ehrx.vlm.grouping import SubDocumentGrouper, generate_hierarchical_index
from ehrx.vlm.config import VLMConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_ontologies.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def process_pdf(pdf_path: Path, out_dir: Path, dpi: int = 200) -> None:
    pdf_stem = pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing ALL pages: {pdf_path} → {out_dir}")

    # Build pipeline (reads .env)
    pipeline = DocumentPipeline(
        vlm_config=VLMConfig.from_env(),
        checkpoint_interval=50,
        dpi=dpi,
    )

    # Always process entire document (single batch)
    document = pipeline.process_document(
        pdf_path=str(pdf_path),
        output_dir=str(out_dir / f"{pdf_stem}_artifacts"),  # keep artifacts separate if needed
        page_range="all",
        document_context={"document_type": "Clinical EHR"}
    )

    # Group pages into sub-documents and build hierarchical index (ontology)
    grouper = SubDocumentGrouper(confidence_threshold=0.80)
    enhanced_doc = grouper.group_document(document)
    hierarchical_index = generate_hierarchical_index(enhanced_doc)

    # Consolidate into a single ontology JSON
    ontology = {
        "document_id": document.get("document_id"),
        "source_pdf": str(pdf_path),
        "processing_stats": document.get("processing_stats", {}),
        # Schema-level enriched content (elements + groups)
        "schema": enhanced_doc,
        # Navigable ontology view
        "hierarchical_index": hierarchical_index,
    }

    # Write single-file ontology per document
    single_path = out_dir / f"{pdf_stem}.json"
    with open(single_path, 'w') as f:
        json.dump(ontology, f, indent=2)

    stats = document.get('processing_stats', {})
    logger.info(
        f"Done: {pdf_path.name} → {single_path.name} | pages {stats.get('processed_pages','?')}/{stats.get('total_pages','?')} | "
        f"elements {stats.get('total_elements','?')}"
    )


def main():
    parser = argparse.ArgumentParser(description="Batch process PDFs into single-file JSON ontologies")
    parser.add_argument("--in", dest="input_dir", required=True, help="Input directory with PDFs")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output directory for ontologies")
    parser.add_argument("--dpi", dest="dpi", type=int, default=200, help="Rasterization DPI for VLM pipeline")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find PDFs
    pdfs = sorted(p for p in input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"⚠️  No PDFs found in {input_dir}")
        sys.exit(0)

    logger.info(f"Found {len(pdfs)} PDFs in {input_dir}")

    for pdf in pdfs:
        try:
            process_pdf(pdf, output_dir, dpi=args.dpi)
        except Exception as e:
            logger.exception(f"Failed to process {pdf.name}: {e}")
            continue

    print(f"\n✅ Completed. Ontologies written under: {output_dir}")


if __name__ == "__main__":
    main()
