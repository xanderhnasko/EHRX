# PDF2EHR (ehrx)

Extract structured data from scanned EHR PDFs using vision language models with semantic ontology and agentic search.

## Overview

PDF2EHR converts scanned medical records into structured, queryable data with full provenance tracking. The pipeline:

1. **PDF Processing** - Converts PDF pages to images
2. **VLM Extraction** - Uses Gemini models to identify and classify medical elements
3. **Semantic Grouping** - Groups elements into logical sub-documents (medications, labs, notes, etc.)
4. **Agentic Query** - Natural language search with filter-then-reason architecture

## Quick Start

### Prerequisites

System dependencies:
```bash
brew install poppler  # macOS
```

### Installation

```bash
pip install -e .
pip install -r requirements.txt
```

### Environment Setup

Create `.env` file:
```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

Verify environment:
```bash
python scripts/verify_env.py
```

### Usage

**Process a PDF:**
```bash
python scripts/run_mvp_pipeline.py
```

**Query existing extraction:**
```bash
python scripts/test_query_only.py
```

## Project Structure

```
ehrx/
├── agent/          # Query interface (filter → reason pipeline)
├── vlm/            # Vision Language Model integration (Gemini)
├── core/           # Configuration and utilities
├── layout/         # Column detection and reading order
├── pdf/            # PDF page conversion
├── hierarchy.py    # Document structure modeling
└── serialize.py    # JSON serialization with provenance

scripts/            # Demo and testing scripts
tests/              # Unit tests
configs/            # YAML configuration
output/             # Extraction results (gitignored)
docs/               # Technical documentation
```

## Query Agent

The `HybridQueryAgent` provides natural language search over extracted EHR data:

```python
from ehrx.agent.query import HybridQueryAgent
from ehrx.vlm.config import VLMConfig

agent = HybridQueryAgent(
    schema_path="output/extraction_enhanced.json",
    vlm_config=VLMConfig.from_env()
)

result = agent.query("What medications is the patient taking?")

# result contains:
# - answer_summary: Human-readable answer
# - matched_elements: Source elements with bounding boxes
# - reasoning: How the answer was derived
# - filter_stats: Query efficiency metrics
```

## Output Format

Extractions produce JSON with:
- Page-level elements (tables, paragraphs, forms)
- Semantic types (medication_table, lab_result, clinical_paragraph)
- Bounding box coordinates (pixel and PDF space)
- Sub-document groupings (medications, labs, progress_notes)
- Full provenance chain for audit trails

## Development

Run tests:
```bash
pytest tests/
```

## License

Proprietary - PDF2EHR Team
