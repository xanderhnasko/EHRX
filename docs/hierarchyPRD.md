# PRD: Hierarchical Structuring of EHR PDFs

## 1. Objective / Goal

The goal of this system is to process multi-page patient EHR PDFs and produce a **hierarchical JSON representation** reflecting:

1. **Document-level grouping**: Pages belonging to the same document (e.g., “Clinical Notes”) are grouped together.  
2. **Hierarchical document organization**: Documents are organized according to predefined semantic categories (Demographics, Vitals, Orders, Meds, Notes, Labs).  
3. **Sub-document structure**: Within multi-page documents, content is structured hierarchically (sections, subsections) while preserving **body content vs. marginal content**.  
4. **Content consolidation**: Repeated margin text across pages is treated as a single label/tag.  
5. **JSON-ready output**: The output can be used by downstream applications for UI rendering, NLP processing, or further analytics.

---

## 2. Scope

### In Scope
- Processing **multi-page PDFs** where each page contains a single document type.  
- Detecting **document type labels** at the top-middle of each page (e.g., “Clinical Notes”).  
- Grouping **consecutive pages with the same document label** into a single document node.  
- Hierarchical structuring of document content:
  - Section / subsection detection (headings, tables, paragraphs, charts).  
  - Margin vs. body text classification.  
  - Consolidation of repeated marginal content across pages.  
- Organizing documents under **predefined high-level categories**: Demographics, Vitals, Orders, Meds, Notes, Labs.  
- Exporting a **JSON schema** reflecting the hierarchy and content.  

### Out of Scope
- Handwriting recognition or images of handwritten notes.  
- Cross-patient or cross-file aggregation.  
- Full semantic understanding of complex clinical data (beyond label-based structuring).  

---

## 3. Assumptions

1. Each page contains **only one document type**.  
2. Document type is reliably indicated by a label at the **top-middle of the page**.  
3. Consecutive pages with the same label belong to the same document.  
4. LayoutParser + OCR already provides:
   - Detected text blocks (paragraphs, tables, charts).  
   - Bounding boxes (`bbox`) of each block.  
   - Page number and document coordinates.  
5. Margin text is predictable in position (left/right/top/bottom bands) and may contain repeated headers/footers.

---

## 4. Functional Requirements

### 4.1 Page-Level Processing
- **FR1.1**: Detect the document type label for each page using LayoutParser + OCR.  
- **FR1.2**: Classify all text blocks on the page as **body content** or **margin content**.  
- **FR1.3**: Extract tables and charts as structured objects.  
- **FR1.4**: Identify repeated margin text across pages for consolidation.  

### 4.2 Document Grouping
- **FR2.1**: Group consecutive pages with identical document labels into a single document entity.  
- **FR2.2**: Track page ranges for each document (`start_page`, `end_page`).  

### 4.3 Document Hierarchy
- **FR3.1**: Map document labels to **predefined categories**:  

Demographics
Vitals
Orders
Meds
Notes
- Visit Summaries
- Discharge Statements
- Progress Notes
Labs

- **FR3.2**: For multi-page documents, identify sections/subsections within the document.  
- **FR3.3**: Preserve order of content as it appears on the page(s).  
- **FR3.4**: Attach extra content (margin text) as **tags or labels** at the document or section level.  

### 4.4 JSON Schema Requirements
- **FR4.1**: Represent hierarchical structure of document content (document → sections → subsections → blocks).  
- **FR4.2**: Include `page`, `bbox`, `text`, `confidence`, `type` (paragraph/table/chart/label) for each block.  
- **FR4.3**: Consolidate repeated margin text as a single tag across pages.  
- **FR4.4**: Include metadata for each document:
- `document_type`
- `category` (Demographics, Vitals, etc.)
- `page_range`  
- `extra_labels` (from margins or repeated headers/footers)  

---

## 5. Non-Functional Requirements

- **NFR1**: System must scale to PDFs of 100+ pages without excessive memory usage.  
- **NFR2**: JSON output must be compatible with downstream UI rendering or NLP pipelines.  
- **NFR3**: Processing time should be linear with number of pages.  
- **NFR4**: Must allow for **manual overrides** if margin filtering incorrectly removes content.  

---

## 6. High-Level System Architecture

**Input:** Patient EHR PDF → **LayoutParser + OCR** → enriched blocks  

### Pipeline Diagram (ASCII / Markdown-friendly)



+---------------------+
| Input PDF |
+---------------------+
|
v
+---------------------+
| LayoutParser + OCR |
| - Detect blocks |
| - OCR text |
| - Bounding boxes |
+---------------------+
|
v
+---------------------+
| Page Processing |
| - Identify doc type |
| - Classify margin/ |
| body text |
| - Extract tables |
+---------------------+
|
v
+---------------------+
| Document Grouping |
| - Consecutive pages |
| with same label |
| - Track page ranges |
+---------------------+
|
v
+---------------------+
| Section / Subsection|
| Detection |
| - Heuristics + OCR |
| - Optional embeddings|
+---------------------+
|
v
+---------------------+
| Margin Deduplication|
| - Consolidate repeated|
| headers/footers |
+---------------------+
|
v
+---------------------+
| Category Mapping |
| - Demographics, Vitals,|
| Orders, Meds, Notes,|
| Labs |
+---------------------+
|
v
+---------------------+
| JSON Export |
| - Hierarchical tree |
| - Sections & blocks |
| - Margin tags |
+---------------------+


---

## 7. Proposed JSON Schema

```json
{
  "patient_id": "123456",
  "documents": [
    {
      "document_type": "Clinical Notes",
      "category": "Notes",
      "page_range": [5, 7],
      "extra_labels": ["Hospital Header", "Confidential Note Footer"],
      "sections": [
        {
          "heading": "Visit Summary",
          "level": 1,
          "children": [
            {
              "type": "paragraph",
              "text": "Patient was admitted ...",
              "page": 5,
              "bbox": [100, 450, 500, 600],
              "confidence": 0.98
            },
            {
              "type": "table",
              "rows": [...],
              "page": 6,
              "bbox": [80, 610, 520, 780]
            }
          ]
        },
        {
          "heading": "Progress Notes",
          "level": 1,
          "children": [
            {
              "type": "paragraph",
              "text": "...",
              "page": 7,
              "bbox": [...]
            }
          ]
        }
      ]
    }
  ]
}

## 8. Detailed Workflow

### For each page:
- Detect the document type label (top-middle).  
- OCR all text blocks.  
- Detect tables/charts.  
- Tag blocks as body vs. margin.

### Group pages into documents:
- Consecutive pages with the same label → same document.

### Identify repeated margin content:
- Compare OCR text from margins across pages.  
- Deduplicate into `extra_labels`.

### Detect sections/subsections within document:
- Use font size, bold/underline, capitalization heuristics.  
- Optionally apply semantic embeddings to improve section continuity across pages.

### Map documents to high-level categories:
- Example: “Clinical Notes → Notes → Visit Summaries / Progress Notes / Discharge Statements”.

### Construct hierarchical JSON:
- Root: Patient document  
- Level 1: Documents grouped by category  
- Level 2: Sections/subsections inside each document  
- Level 3: Content blocks

---

## 9. Open Design Questions
- Should tables and charts be normalized into a standardized format (JSON table rows) for downstream analysis?  
- Margin classification: fixed percentage threshold vs. ML-based classifier?  
- Section detection: purely heuristic (font/position) vs. embedding-based similarity for cross-page continuity?  
- Error handling: when a page OCR fails or document label is missing, how should it be flagged in JSON?  

---

## 10. Deliverables
- **Codebase**: Python scripts or pipeline modules using LayoutParser + OCR.  
- **Structured JSON**: Nested output for each patient PDF.  
- **Documentation**: PRD, JSON schema definition, margin & section classification rules.  
- **Optional visualization**: Web UI or script to display hierarchy for QA.  


## 11. Clarifications for First Iteration

- **Document Labels**:  
  - Labels are assumed to always be in the same position and font across all PDFs.  
  - Any pages or documents that fail to be grouped due to missing titles will be placed under an **"Unlabeled"** category.  
  - Any document that fails to be placed in a hierarchical position will be categorized under **"Miscellaneous"** for error evaluation.  

- **Page Grouping**:  
  - Only **consecutive pages** with the same document label will be grouped into a single document.  

- **Section/Subsection Detection**:  
  - For this first iteration, semantic similarity across pages will **not** be used.  
  - Detection will rely solely on **visual cues**, such as titles, font size, and formatting (bold/underline).  

- **JSON Schema**:  
  - All detected blocks will be included in the JSON output.  
  - If a block is not meaningful, it will be labeled as `"unmeaningful"`.  

- **Categories**:  
  - High-level categories and subcategories are **fixed** for this iteration.  

- **Performance & Scaling**:  
  - The system should handle very large PDFs (up to **600 pages**).  
  - For this first iteration, processing speed is **not a primary concern**.  
