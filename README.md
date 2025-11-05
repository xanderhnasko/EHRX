# ehrx - EHR Extraction Tool

A local-first pipeline for extracting structured, hierarchical data from scanned EHR PDFs.

## Overview

**ehrx** ingests scanned healthcare documents (PDFs) and outputs:
- Structured JSONL with detected elements (text blocks, tables, figures, handwriting)
- Hierarchical index with sections/subsections
- Extracted assets (cropped images, table CSVs)

All processing is done locally—no PHI leaves your machine.

## Features

- **Layout Detection**: Uses LayoutParser with Detectron2 or PaddleDetection
- **OCR**: Tesseract for text and table extraction
- **Hierarchical Structure**: Deterministic section detection (PROBLEMS, MEDICATIONS, LABS, etc.)
- **Table Extraction**: Heuristic grid detection and CSV export
- **Privacy-First**: No cloud calls; no PHI in logs
- **Scalable**: Handles 600+ page documents

## Installation

### Requirements

- Python 3.11
- System dependencies: `tesseract-ocr` (≥ 5.0)

### Setup

```bash
# Install system dependencies (macOS)
brew install tesseract

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr

# Install Python dependencies (Detectron2 CPU backend - default)
pip install 'torch==2.3.*' 'torchvision==0.18.*' --index-url https://download.pytorch.org/whl/cpu
pip install layoutparser detectron2==0.6
pip install pytesseract opencv-python pymupdf pandas pyyaml pydantic typer

# Alternative: PaddleDetection backend
pip install layoutparser[paddledetection] paddlepaddle
```

## Usage

```bash
ehrx extract \
  --in data/input/scan.pdf \
  --out runs/scan-001 \
  --detector pubLayNet \
  --min-conf 0.5 \
  --ocr tesseract \
  --allow-vector true \
  --assets on \
  --pages "all" \
  --threads 4 \
  --log-level INFO
```

### CLI Options

- `--in`: Input PDF file path
- `--out`: Output directory for results
- `--detector`: Layout detection model (`pubLayNet` or `paddle_ppstructure`)
- `--min-conf`: Minimum confidence threshold (default: 0.5)
- `--ocr`: OCR engine (currently only `tesseract`)
- `--allow-vector`: Try vector text extraction before OCR
- `--assets`: Save cropped images/CSVs (`on`/`off`)
- `--pages`: Page range to process (`"all"` or `"1-50,120-130"`)
- `--threads`: Number of OCR threads (default: 4)
- `--log-level`: Logging level (INFO, DEBUG, WARNING, ERROR)

## Output Structure

After running extraction, the output directory will contain:

```
runs/scan-001/
  document.elements.jsonl    # Flat list of detected elements
  document.index.json        # Hierarchical structure + manifest
  assets/
    table_E_0045.png
    table_E_0045.csv
    figure_E_0078.png
    hand_E_0123.png
```

### Element Types

Each element in `document.elements.jsonl` has:
- Common fields: `id`, `doc_id`, `page`, `type`, `bbox_pdf`, `bbox_px`, `rotation`, `z_order`, `source`, `detector_name`, `detector_conf`
- Type-specific payload:
  - `text_block`: `{text, tokens?}`
  - `table`: `{headers?, rows?, csv_ref?, ocr_lines}`
  - `figure`: `{image_ref, caption?}`
  - `handwriting`: `{image_ref, ocr_text?, ocr_confidence?}`

## Configuration

Default configuration is in `configs/default.yaml`. You can customize:
- Detector models and parameters
- OCR settings and preprocessing
- Table extraction heuristics
- Hierarchy heading patterns
- Privacy/logging settings

## Development

### Project Structure

```
ehrx/
  ehrx/                      # Main package
    __init__.py
    cli.py                   # CLI entrypoint
    config.py                # Config loading/validation
    pager.py                 # PDF rasterization
    detect.py                # Layout detection
    route.py                 # Element routing
    ocr.py                   # OCR wrappers
    tables.py                # Table extraction
    hierarchy.py             # Hierarchy builder
    serialize.py             # Output serialization
    utils.py                 # Utilities
  configs/default.yaml       # Default configuration
  tests/smoke_test.py        # Tests
  README.md
```

### Running Tests

```bash
pytest tests/
```

## Performance Notes

- Processes 600+ page documents
- Rasters at 150-200 DPI for detection, 300 DPI for OCR
- Streams output (doesn't hold entire document in memory)
- Multi-threaded OCR processing

## Privacy & PHI

- **Local-only processing**: No external API calls
- **No PHI in logs**: Text snippets are never logged (configurable)
- All processing happens on your machine
- Suitable for HIPAA-compliant workflows

## Future Extensions (Not in MVP)

- LLM-based summarization/normalization
- Advanced handwriting recognition
- FHIR/OMOP mapping
- Interactive web viewer
- Database integration

## License

[To be determined]

## References

See `SPECS.md` for detailed technical specifications.

