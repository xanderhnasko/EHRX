# PDF2EHR MVP Plan: 24-Hour Implementation Guide

**Target Date**: November 15, 2025
**Scope**: 650-page EHR processing with schema generation and natural language query agent
**Status**: Active Development

---

## Executive Summary

Transform PDF2EHR from single-page processing to full-document capability with natural language query interface. Leverage existing solid VLM foundation (Gemini 2.5 Flash, 19 semantic types, bounding box provenance) to process massive EHRs and enable semantic search.

**What's Working Now**:
- ✅ Single-page VLM extraction (production-ready)
- ✅ 19-type semantic classification
- ✅ Bounding box provenance (pixel + PDF coordinates)
- ✅ Confidence scoring with human review flags
- ✅ Cost: ~$0.0012/page ($0.78 for 650 pages)

**What We're Building**:
- 🔨 Multi-page processing pipeline (650+ pages)
- 🔨 Sub-document grouping (Labs, Meds, Imaging, etc.)
- 🔨 Natural language query agent over schema

---

## Critical Path: 3 Core Components

### Component 1: Multi-Page Processing Pipeline (4-6 hours)

**Goal**: Process entire 650-page PDF into structured schema with full provenance

**New File**: `ehrx/vlm/pipeline.py`

**Core Functionality**:
```python
class DocumentPipeline:
    """End-to-end PDF to structured schema processing."""

    def process_document(self, pdf_path: str, output_dir: str) -> dict:
        """
        Process all pages of a PDF through VLM extraction.

        Returns:
        {
            "document_id": "generated_id",
            "total_pages": 650,
            "pages": [
                {
                    "page_number": 1,
                    "elements": [...],  # VLM detections
                    "processing_metadata": {...}
                }
            ],
            "processing_summary": {
                "total_elements": 5432,
                "total_cost": 0.78,
                "errors": []
            }
        }
        """
```

**Implementation Steps**:
1. Iterate through pages using existing `Pager.iter_pages(pdf_path)`
2. Call `VLMClient.detect_elements()` for each page
3. Aggregate all results into single document structure
4. Add progress logging: "Processing page 245/650..."
5. Error handling: skip failed pages, log errors, continue processing
6. Save intermediate results every 50 pages (checkpointing)
7. Final output: `document_full.json` with all elements + bounding boxes

**Key Design Decisions**:
- **Sequential processing**: No parallelization for MVP (simpler, avoids rate limits)
- **No cross-page context**: Process each page independently
- **Flash-only**: Skip Gemini Pro cascade (working well with Flash)
- **Checkpoint on errors**: Save progress, can resume if API fails

**Expected Performance**:
- Time: 30-40 minutes for 650 pages (3-5 sec/page)
- Cost: ~$0.78 total (validated)
- API calls: 650 (well under 1,500/day free tier limit)

---

### Component 2: Sub-Document Grouping & Hierarchy (2-3 hours)

**Goal**: Organize flat page list into hierarchical sub-documents (Labs, Medications, Imaging, etc.)

**New File**: `ehrx/vlm/grouping.py`
**Modified**: `ehrx/hierarchy.py` (adapt for VLM element types)

**Core Functionality**:
```python
class SubDocumentGrouper:
    """Group pages into clinical sub-documents."""

    def group_pages(self, document_data: dict) -> dict:
        """
        Detect sub-document boundaries using section headers.

        Strategy:
        1. Scan all pages for high-confidence section_header elements
        2. Identify document type keywords:
           - "LABORATORY RESULTS" → Labs sub-document
           - "MEDICATIONS" / "PHARMACY" → Medications
           - "RADIOLOGY" / "IMAGING" → Imaging
           - "PROGRESS NOTES" → Clinical Notes
        3. Group consecutive pages under same heading
        4. Create hierarchical index

        Returns:
        {
            "document_id": "...",
            "sub_documents": [
                {
                    "type": "laboratory_results",
                    "title": "LABORATORY RESULTS",
                    "page_range": [45, 67],
                    "pages": [...],
                    "element_count": 234
                }
            ]
        }
        """
```

**Detection Logic**:
- Scan for `section_header` elements with high confidence (>0.85)
- Use keyword matching for common EHR section types:
  - Labs: "LAB", "LABORATORY", "BLOOD", "URINALYSIS"
  - Meds: "MEDICATION", "PHARMACY", "PRESCRIPTION", "RX"
  - Imaging: "RADIOLOGY", "IMAGING", "X-RAY", "CT", "MRI"
  - Vitals: "VITAL SIGNS", "VITALS"
  - Notes: "PROGRESS NOTE", "CLINICAL NOTE", "H&P"
  - Orders: "ORDERS", "PHYSICIAN ORDERS"
- Default: Group by page until new section detected
- Multi-page sub-documents: All pages under same heading grouped together

**Output Format**:
```json
{
  "document_id": "ehr_650_pages",
  "patient_demographics": {
    "extracted_from_page": 1,
    "elements": [...]
  },
  "sub_documents": [
    {
      "id": "subdoc_001",
      "type": "laboratory_results",
      "title": "LABORATORY RESULTS",
      "page_range": [45, 67],
      "page_count": 23,
      "pages": [
        {
          "page_number": 45,
          "elements": [
            {
              "element_id": "E_0001",
              "type": "lab_results_table",
              "content": "...",
              "bbox_pixel": [68, 95, 424, 110],
              "bbox_pdf": [24.48, 752.4, 152.64, 757.8],
              "confidence": 0.94
            }
          ]
        }
      ]
    }
  ]
}
```

**Integration with Existing `hierarchy.py`**:
- Preserve existing category mapping logic
- Adapt to use VLM semantic types instead of LayoutParser types
- Map VLM types to categories:
  - `lab_results_table` → Labs category
  - `medication_table` → Meds category
  - `vital_signs_table` → Vitals category
  - etc.

---

### Component 3: Natural Language Query Agent (3-4 hours)

**Goal**: Answer questions like "What were the patient's blood results?" by reasoning over schema

**New File**: `ehrx/agent/query.py`
**New File**: `ehrx/agent/__init__.py`

**Architecture: Hybrid Liberal Filtering + Pro Reasoning**

**Why Hybrid Over Full Dump**:
- 🎯 10-25x context reduction (500K → 20-100K tokens)
- 💰 Cheaper Pro queries (smaller context)
- ⚡ Faster response times
- ✅ Liberal filtering = no risk of missing relevant content
- ✅ No embeddings/vector search needed

**Two-Stage Process**:

#### Stage 1: Liberal Query Analysis (Flash - Cheap & Fast)

**Purpose**: Identify ALL potentially relevant schema fields (be generous!)

```python
def analyze_query(question: str) -> dict:
    """
    Use Flash to identify potentially relevant schema fields.

    Prompt to Flash:
    ---
    Analyze this user question about an EHR:
    "{question}"

    Available semantic types:
    - document_header, patient_demographics, section_header
    - clinical_paragraph, medication_table, lab_results_table
    - vital_signs_table, problem_list, assessment_plan
    - handwritten_annotation, medical_figure, form_field_group
    - etc. (all 19 types)

    Return ALL semantic types that MIGHT be relevant.
    Be LIBERAL - include anything potentially useful.
    Better to include extra than to miss important context.
    ---

    Returns:
    {
        "relevant_types": ["lab_results_table", "vital_signs_table",
                          "clinical_paragraph"],  # Liberal inclusion
        "relevant_subdocs": ["laboratory_results", "vitals"],
        "temporal_context": "recent",  # If mentioned
        "reasoning": "Labs are primary, vitals might show context..."
    }

    Cost: ~$0.001 per query
    """
```

**Key Principle**: **Better to include too much than risk missing relevant info**

#### Stage 2: Deterministic Filter + Pro Reasoning

**Purpose**: Extract matching elements and reason over filtered context

```python
class HybridQueryAgent:
    """Query agent with liberal filtering + Pro reasoning."""

    def __init__(self, schema_path: str):
        self.schema = self._load_schema(schema_path)
        self.flash_client = VLMClient(model_name="gemini-2.5-flash")
        self.pro_client = VLMClient(model_name="gemini-2.5-pro")

    def query(self, question: str) -> dict:
        """
        Two-stage hybrid query.

        Stage 1: Analyze query with Flash (liberal field identification)
        Stage 2: Filter schema + reason with Pro
        """

        # Stage 1: Liberal field identification (Flash)
        relevant_fields = self._analyze_query_liberal(question)
        # Returns: {"types": [...], "subdocs": [...]}

        # Stage 2: Deterministic filtering (no AI, just code)
        filtered_schema = self._filter_schema(relevant_fields)
        # From 5000 elements → maybe 500-1000 (liberal)
        # Still saves 5-10x tokens vs full dump

        # Stage 3: Reason with Pro over filtered context
        answer = self._reason_with_pro(question, filtered_schema)

        return {
            "question": question,
            "matched_elements": answer["elements"],
            "reasoning": answer["reasoning"],
            "filter_stats": {
                "original_elements": len(self.schema["elements"]),
                "filtered_elements": len(filtered_schema["elements"]),
                "reduction_ratio": "10x"  # Example
            }
        }

    def _filter_schema(self, relevant_fields: dict) -> dict:
        """
        Deterministically extract ALL elements matching criteria.
        Liberal = include anything that might be relevant.
        """
        filtered = []

        for element in self.schema["elements"]:
            # Include if type matches
            if element["type"] in relevant_fields["types"]:
                filtered.append(element)

            # Include if sub-document matches
            elif element.get("subdoc_type") in relevant_fields["subdocs"]:
                filtered.append(element)

            # Liberal: Include section headers for context
            elif element["type"] == "section_header":
                filtered.append(element)

            # Liberal: Always include patient demographics
            elif element["type"] == "patient_demographics":
                filtered.append(element)

        return {"elements": filtered}
```

**Example Query Flow**:

```
User: "What were the patient's blood results from last month?"

Stage 1 (Flash Analysis):
{
    "relevant_types": [
        "lab_results_table",      # Primary target
        "vital_signs_table",      # Liberal: might be relevant
        "clinical_paragraph",     # Liberal: might reference labs
        "section_header"          # Always include for context
    ],
    "relevant_subdocs": ["laboratory_results", "vitals"],
    "temporal_context": "last_month"
}

Stage 2 (Deterministic Filter):
- Original: 5,432 elements
- Filtered: 847 elements (still generous!)
- Tokens: 500K → 78K (6.4x reduction)

Stage 3 (Pro Reasoning):
- Context: 78K tokens (filtered schema)
- Query: "What were the patient's blood results from last month?"
- Pro finds: 3 lab tables with temporal context matching
- Returns: Elements with bounding boxes

Total Cost: $0.001 (Flash) + $0.01 (Pro on smaller context) = $0.011/query
```

**Why This Beats Full Dump AND Traditional RAG**:

**vs. Full Context Dump**:
- ✅ 5-10x token reduction (still liberal)
- ✅ Faster Pro queries
- ✅ Cheaper Pro queries
- ✅ Better signal-to-noise for reasoning

**vs. Traditional RAG/Embeddings**:
- ✅ No vector embeddings needed
- ✅ Deterministic extraction (get ALL matching elements, not top-k)
- ✅ Leverages existing structured schema
- ✅ No risk of missing relevant elements due to similarity scores
- ✅ Simpler to implement and debug
- ✅ Liberal filtering = safety margin

**vs. Strict Schema Filtering**:
- ✅ Liberal approach reduces risk of missing context
- ✅ Cheap Flash call enables generous inclusion
- ✅ Still saves massive tokens on obviously irrelevant content

---

## Simplified MVP Scope: What We're Cutting

To hit the 24-hour timeline, we're ruthlessly cutting from the full PRD:

### Excluded from MVP (Phase 2+)
- ❌ **Gemini Pro cascade**: Flash-only (working well, no low-confidence cases yet)
- ❌ **Cross-page context injection**: Process pages independently
- ❌ **Clinical relationship extraction**: No temporal/causal links
- ❌ **Human review interface**: Flag elements only, no UI
- ❌ **Parallel processing**: Sequential fine for 30-40min runtime
- ❌ **Advanced chunking**: Page-based, not semantic boundaries
- ❌ **Continuous learning**: No feedback loop
- ❌ **Frontend build**: Delay until separate repo integrated

### Retained for MVP
- ✅ **19-type semantic classification**: Full ontology
- ✅ **Bounding box provenance**: Pixel + PDF coordinates
- ✅ **Confidence scores**: With review flags
- ✅ **Sub-document grouping**: Multi-page clinical sections
- ✅ **Full 650-page processing**: End-to-end capability
- ✅ **Natural language queries**: Direct reasoning over schema

---

## Hour-by-Hour Implementation Timeline

### Hours 1-2: Multi-Page Pipeline Foundation
**Tasks**:
- Create `ehrx/vlm/pipeline.py`
- Implement `DocumentPipeline` class
- Page iteration with `Pager`
- VLM extraction per page
- JSON aggregation logic
- Progress logging

**Milestone**: Process 10-page sample successfully

### Hours 3-4: Error Handling & Testing
**Tasks**:
- Add error handling and retry logic
- Implement checkpointing (save every 50 pages)
- Test on 20-page sample
- Start full 650-page run (let it cook in background)

**Milestone**: 650-page processing running, no crashes on sample

### Hours 5-7: Sub-Document Grouping
**Tasks**:
- Create `ehrx/vlm/grouping.py`
- Implement `SubDocumentGrouper` class
- Section header detection logic
- Keyword matching for document types
- Build hierarchical index
- Test on 650-page output (should be done by now)

**Milestone**: Hierarchical structure with sub-documents identified

### Hours 8-11: Query Agent
**Tasks**:
- Create `ehrx/agent/query.py`
- Implement `DirectReasoningAgent` class
- Schema loader
- Prompt engineering for direct reasoning
- Result formatter with bounding boxes
- Test queries:
  - "What were the patient's blood results?"
  - "What medications is the patient currently taking?"
  - "Show me all vital signs from page 100-150"

**Milestone**: Agent answers 3+ test queries correctly

### Hours 12-14: Integration & Demo Prep
**Tasks**:
- Run 10+ diverse test queries
- Verify bounding box provenance traces correctly
- Create example query script for demo
- Document known limitations
- Buffer time for debugging

**Milestone**: Ready for demo

---

## Success Criteria (Definition of Done)

### Primary Goals
- ✅ **Process 650-page PDF**: Complete run without crashes
- ✅ **Hierarchical schema**: Sub-documents correctly grouped
- ✅ **Bounding box provenance**: All elements traceable to source
- ✅ **Query accuracy**: Agent answers 80%+ of test queries correctly
- ✅ **Cost efficiency**: Total cost under $1.00

### Demo Capabilities
- ✅ Show full document processing (650 pages → structured JSON)
- ✅ Display hierarchical structure (sub-documents)
- ✅ Query: "Patient's blood results" → Returns lab tables with page/bbox
- ✅ Query: "Current medications" → Returns medication elements
- ✅ Query: "Vital signs on page X" → Returns vitals with location
- ✅ Trace any element back to source PDF location

### Output Artifacts
- ✅ `document_full.json`: Complete extraction with all elements
- ✅ `document_index.json`: Hierarchical sub-document index
- ✅ `query_examples.json`: Sample queries and results
- ✅ Processing log with stats (time, cost, errors)

---

## Files to Create/Modify

### New Files
```
ehrx/vlm/pipeline.py          # Multi-page processor
ehrx/vlm/grouping.py          # Sub-document detection
ehrx/agent/__init__.py        # Agent module init
ehrx/agent/query.py           # Query agent
tests/vlm/test_pipeline.py    # Pipeline tests
tests/agent/test_query.py     # Agent tests
docs/MVP_PLAN.md              # This document
```

### Modified Files
```
ehrx/hierarchy.py             # Adapt for VLM element types (minimal changes)
```

### Output Files (Generated)
```
output/document_full.json     # Complete extraction
output/document_index.json    # Hierarchical index
output/query_examples.json    # Demo queries
output/processing_log.txt     # Stats and errors
```

---

## Risk Mitigation

### API Rate Limits
- **Risk**: 650 calls might hit daily limits
- **Mitigation**: Already have retry logic; free tier supports 1,500 req/day
- **Fallback**: Split processing across 2 days if needed

### Processing Time
- **Risk**: 30-40 min runtime delays iteration
- **Mitigation**: Test on 20-page sample first; run full batch overnight
- **Acceptable**: MVP demo doesn't require real-time processing

### Query Accuracy
- **Risk**: Direct reasoning might miss relevant elements
- **Mitigation**: Start with simple exact-match queries; expand gradually
- **Fallback**: Pre-program specific demo queries that we know work

### Sub-Document Detection
- **Risk**: Section header detection might miss boundaries
- **Mitigation**: Use high-confidence VLM classifications; flag uncertain boundaries
- **Fallback**: Manual verification for demo dataset

### Context Window Limits
- **Risk**: Schema might exceed 1M tokens
- **Mitigation**: Estimate shows ~500K tokens; monitor actual size
- **Fallback**: Query over sub-documents individually if needed

---

## Technical Dependencies

### Existing (Already Working)
- ✅ Google Cloud Vertex AI client with authentication
- ✅ Gemini 2.5 Flash API integration
- ✅ VLM structured output with response schemas
- ✅ PDF rasterization (Pager)
- ✅ Coordinate mapping (pixel ↔ PDF)
- ✅ 19-type semantic element taxonomy
- ✅ Confidence scoring

### New (To Build)
- 🔨 Multi-page iteration orchestration
- 🔨 Result aggregation logic
- 🔨 Sub-document boundary detection
- 🔨 Query agent prompt engineering
- 🔨 Response formatting with provenance

### External
- ✅ Google Cloud project with Vertex AI enabled
- ✅ 650-page test PDF available
- ⏳ Frontend (separate repo, defer for now)

---

## Cost Analysis

### Current Validated Costs
- **Extraction (Flash)**: $0.0012 per page × 650 pages = $0.78
- **Query analyzer (Flash)**: ~$0.001 per query (liberal field identification)
- **Query reasoning (Pro)**: ~$0.01-0.02 per query (filtered context, 5-10x smaller)
- **Total per query**: ~$0.011-0.021 (vs $0.05 with full dump)
- **Total MVP**: ~$0.80-$0.85 (extraction + 10 test queries)

### Model Selection Strategy
- **Flash for extraction**: Fast, cost-effective, excellent for structured output
- **Flash for query analysis**: Cheap liberal filtering to identify relevant fields
- **Pro for query reasoning**: Superior reasoning over filtered context (5-10x cheaper than full dump)

### Optimization Opportunities (Phase 2)
- Caching repeated patterns
- Batch API calls
- Model selection based on complexity

---

## Phase 2 Enhancements (Post-MVP)

After successful MVP demo, consider:

1. **Gemini Pro Cascade**: For complex tables/figures with low confidence
2. **Cross-Page Context**: Maintain clinical narratives across pages
3. **Clinical Relationships**: Link medications ↔ labs, temporal references
4. **Parallel Processing**: 5-10x throughput improvement
5. **Advanced Query Parsing**: Schema-based RAG (SLIDERS approach)
6. **Human Review Interface**: Accept/reject/edit workflow
7. **Continuous Learning**: Feedback loop for prompt refinement
8. **Frontend Integration**: Bounding box visualization
9. **Production Deployment**: Scaling, monitoring, security

---

## Open Questions

- [ ] Where is the 650-page PDF located?
- [ ] What format should demo queries be in? (CLI, Jupyter notebook, script?)
- [ ] Do we need specific patient demographics extraction?
- [ ] Should we prioritize specific sub-document types for demo?

---

## Notes & Observations

**Single-Page Quality**: Current VLM extraction is production-ready with excellent semantic understanding. The confidence scores are high (>0.95 typical), and we haven't encountered cases requiring Gemini Pro escalation yet.

**Context Window Advantage**: The 1M token context window is a game-changer. Instead of complex RAG infrastructure, we can dump the entire schema and let the model reason directly. This is both simpler AND more accurate for the MVP.

**Sub-Document Importance**: EHRs contain 10+ distinct clinical documents. Proper grouping is critical for user experience and query accuracy. Multi-page lab reports (20-30 pages) must stay together under a single heading.

**Bounding Box Provenance**: This is our killer feature. Every piece of extracted data traces back to the exact pixel location in the source PDF. Essential for medical accuracy and auditability.

---

**Document Version**: 1.0
**Last Updated**: November 14, 2025
**Status**: Active Implementation Guide
