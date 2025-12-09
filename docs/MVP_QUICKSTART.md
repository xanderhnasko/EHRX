# PDF2EHR MVP Quick Start Guide

**Version**: 1.0 (MVP)
**Date**: November 14, 2025

## Overview

This MVP enables end-to-end processing of massive EHR PDFs (650+ pages) with natural language query capabilities.

**Pipeline**:
1. **Multi-Page Extraction**: VLM processes all pages → structured schema with bounding boxes
2. **Sub-Document Grouping**: Organizes pages into clinical sections (Labs, Meds, Imaging, etc.)
3. **Hybrid Query Agent**: Natural language → filtered context → Pro reasoning → answers with provenance

## Quick Start (5 Minutes)

### 1. Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GCP_PROJECT_ID="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### 2. Run Test Pipeline (5 pages)

```bash
python scripts/test_pipeline.py
```

When prompted:
- Enter path to PDF
- Pipeline extracts first 5 pages
- View results in `output/test_pipeline/`

### 3. Run Full MVP Pipeline (650 pages)

```bash
python scripts/run_mvp_pipeline.py
```

When prompted:
- Enter path to 650-page PDF
- Enter page range: `all` (or `1-650`)
- Wait 30-40 minutes for extraction
- Pipeline groups sub-documents
- Interactive query mode activates

### 4. Query Your EHR

Once in interactive mode:

```
Query: What were the patient's blood test results?

✓ Query Results
  Matched elements: 12
  Filter efficiency: 5.2x

Answer: Found 3 lab result tables with CBC, metabolic panel, and lipid panel.

Matched Elements:

  [1] lab_results_table
      Content: CBC: WBC 7.2, RBC 4.5, Hemoglobin 14.2...
      Page: 56
      Bbox (pixel): [68, 95, 424, 110]
```

## Architecture

### Component 1: Multi-Page Pipeline (`ehrx/vlm/pipeline.py`)

**What it does**:
- Rasterizes PDF pages at 200 DPI
- Calls Gemini 2.5 Flash for each page
- Extracts 19 semantic element types
- Tracks bounding boxes (pixel + PDF coordinates)
- Handles errors gracefully
- Checkpoints every 50 pages

**Performance**:
- ~3-5 seconds per page
- $0.0012 per page
- 650 pages = 30-40 minutes, $0.78

### Component 2: Sub-Document Grouping (`ehrx/vlm/grouping.py`)

**What it does**:
- Detects section headers via VLM classifications
- Matches keywords to sub-document types:
  - Labs: "LABORATORY", "BLOOD", "URINALYSIS"
  - Meds: "MEDICATION", "PHARMACY", "RX"
  - Imaging: "RADIOLOGY", "CT", "MRI"
  - etc.
- Groups consecutive pages under same heading
- Builds hierarchical index

**Output**:
- Enhanced schema with `sub_documents` array
- Lightweight index for navigation

### Component 3: Hybrid Query Agent (`ehrx/agent/query.py`)

**How it works**:

**Stage 1 - Liberal Analysis (Flash, $0.001)**:
```python
User: "What were blood results?"
  ↓
Flash analyzes query
  ↓
Returns: {
  "relevant_types": ["lab_results_table", "vital_signs_table",
                    "clinical_paragraph", "section_header"],
  "relevant_subdocs": ["laboratory_results"],
  "reasoning": "Labs primary, vitals might provide context"
}
```

**Stage 2 - Deterministic Filter (Python)**:
```python
Filter schema by relevant types + subdocs
  ↓
Always include: section_header, patient_demographics
  ↓
From 5,432 elements → 847 elements (6.4x reduction)
```

**Stage 3 - Pro Reasoning (Pro, $0.01-0.02)**:
```python
Pro reasons over filtered 847 elements (not full 5,432)
  ↓
Finds relevant elements
  ↓
Returns with bounding box provenance
```

**Why this beats full context dump**:
- 5-10x cheaper Pro queries
- Faster response times
- Better signal-to-noise ratio
- Liberal = safe, won't miss context

**Why this beats traditional RAG**:
- No embeddings needed
- Deterministic (get ALL matches, not top-k)
- Simpler to debug
- Leverages structured schema

## File Structure

```
PDF2EHR/
├── ehrx/
│   ├── vlm/
│   │   ├── pipeline.py        # Multi-page processor
│   │   ├── grouping.py        # Sub-document detection
│   │   ├── client.py          # VLM API client
│   │   ├── config.py          # Configuration
│   │   └── models.py          # Data models
│   ├── agent/
│   │   └── query.py           # Hybrid query agent
│   └── pdf/
│       └── pager.py           # PDF rasterization
├── docs/
│   ├── MVP_PLAN.md            # Detailed implementation plan
│   └── MVP_QUICKSTART.md      # This file
├── scripts/
│   ├── test_pipeline.py       # Test on 5 pages
│   └── run_mvp_pipeline.py    # Full pipeline + query mode
└── ...
```

## Output Files

After running the pipeline:

```
output/mvp_demo/
├── <doc_id>_full.json         # Complete extraction
│   └── pages: [{elements: [...], page_info: {...}]
├── <doc_id>_enhanced.json     # With sub-documents
│   ├── patient_demographics: {...}
│   └── sub_documents: [{type, title, pages: [...]}]
├── <doc_id>_index.json        # Lightweight index
│   └── sub_documents: [{type, page_range, page_summaries}]
└── <doc_id>_checkpoint_*.json # Intermediate saves
```

## Example Queries

### Blood Results
```
Query: What were the patient's blood test results?
→ Finds: lab_results_table elements
→ Returns: CBC, metabolic panels with page/bbox
```

### Medications
```
Query: What medications is the patient taking?
→ Finds: medication_table elements
→ Returns: Med lists with dosages, page/bbox
```

### Vitals
```
Query: Show me vital signs from pages 100-150
→ Finds: vital_signs_table in page range
→ Returns: BP, temp, pulse readings with page/bbox
```

### Temporal
```
Query: What were the most recent lab values?
→ Flash identifies temporal context: "recent"
→ Filters labs by recency
→ Returns latest results
```

## Cost Analysis

**650-Page Document**:
- Extraction (Flash): 650 × $0.0012 = $0.78
- Query (Flash + Pro): $0.011-0.021 per query
- Total for 10 queries: ~$0.80-0.90

**Comparison**:
- Full context dump (Pro): $0.05/query
- Hybrid approach (Flash + Pro): $0.011-0.021/query
- **Savings**: 2.4-4.5x cheaper

## Performance

**Processing Time**:
- 650 pages sequential: 30-40 minutes
- Future parallel (5 workers): 6-8 minutes

**Query Response Time**:
- Flash analysis: 1-2 seconds
- Python filtering: <0.1 seconds
- Pro reasoning: 3-5 seconds
- **Total**: 4-7 seconds per query

## Troubleshooting

### Authentication Error
```
Error: Failed to initialize Vertex AI
```
**Fix**:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GCP_PROJECT_ID="your-project-id"
```

### Rate Limit Error
```
Error: Quota exceeded
```
**Fix**:
- Free tier: 1,500 req/day
- Wait or use smaller batches
- Checkpoints allow resuming

### Out of Memory
```
Error: Memory error loading schema
```
**Fix**:
- Use hierarchical index instead of full schema
- Query sub-documents individually
- Increase system RAM

## Next Steps

### Ready for Production
After successful MVP demo, consider:

1. **Parallel Processing**: 5-10x speedup
2. **Clinical Relationship Extraction**: Link meds ↔ labs
3. **Human Review Interface**: Accept/reject/edit workflow
4. **Frontend Integration**: Bounding box visualization
5. **Advanced Caching**: Repeated pattern optimization

### Phase 2 Enhancements
See `docs/VLM_REFACTOR.md` for full roadmap:
- Cross-page context
- Gemini Pro cascade for complex elements
- Continuous learning from feedback
- Production monitoring

## Support

**Issues**: Check logs in `mvp_pipeline.log`

**Questions**: See `docs/MVP_PLAN.md` for detailed architecture

**Testing**: Run `test_pipeline.py` on small samples first

---

**Ready to process your 650-page EHR?**

```bash
python scripts/run_mvp_pipeline.py
```
