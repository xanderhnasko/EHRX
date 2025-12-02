# Hierarchical Document Structuring - Implementation Summary

## Overview

This document summarizes the implementation of hierarchical document structuring functionality for the PDF2EHR project, as specified in `hierarchyPRD.md`.

## Implemented Features

### 1. Document Label Detection (`ehrx/hierarchy.py`)
**Class: `DocumentLabelDetector`**

- Detects document type labels at the top-middle region of pages
- Configurable detection region (default: top 15%, middle 60% horizontally)
- Filters candidates by:
  - Position (within detection region)
  - Text block height (15-80 pixels)
  - OCR confidence (>= 0.5)
- Returns the topmost, highest-confidence text block as the document label

**Configuration Options:**
- `top_region`: Vertical fraction for detection (default: 0.15)
- `horizontal_start`, `horizontal_end`: Horizontal boundaries (default: 0.2-0.8)
- `min_label_height`, `max_label_height`: Valid label height range (pixels)
- `min_label_confidence`: Minimum OCR confidence threshold

### 2. Document Grouping (`ehrx/hierarchy.py`)
**Class: `DocumentGrouper`**

- Groups consecutive pages with identical document labels
- Tracks page ranges for each document group
- Handles unlabeled pages (grouped as "Unlabeled")
- Creates `DocumentGroup` objects containing:
  - Document type (label text)
  - Page range (start, end - inclusive)
  - List of page numbers
  - Elements from all pages
  - Category assignment
  - Detected sections

### 3. Section Detection (`ehrx/hierarchy.py`)
**Class: `SectionDetector`**

- Detects section headings using visual heuristics:
  - **Text block height** (proxy for font size)
  - **Capitalization ratio** (all caps → likely heading)
  - **Brevity** (short lines → likely headings)
  - **Keyword matching** (using regex patterns from config)
  - **Gap above** (spacing before element)

- Assigns heading levels (H1, H2, H3+) based on scoring:
  - **H1**: Strong keyword + large text OR high caps ratio + large gap
  - **H2**: Keyword match OR tall + brief text
  - **H3+**: Nested under H2 or H1

- Builds hierarchical structure:
  - Sections can contain subsections
  - Each section contains child elements (text, tables, figures)

**Configuration Options:**
- `min_heading_height`: Minimum height for headings (default: 20px)
- `caps_ratio_min`: Minimum capitalization ratio (default: 0.6)
- `gap_above_px`: Minimum gap above headings (default: 18px)
- `heading_regex`: List of keyword patterns for heading detection

### 4. Category Mapping (`ehrx/hierarchy.py`)
**Class: `CategoryMapper`**

- Maps document labels to predefined EHR categories:
  - **Demographics**: Patient information, personal info
  - **Vitals**: Vital signs, blood pressure, temperature
  - **Orders**: Physician orders, prescriptions
  - **Meds**: Medications, medicines, pharmacy
  - **Notes**: Clinical notes, progress notes, visit summaries, discharge
  - **Labs**: Lab results, laboratory, test results
  - **Miscellaneous**: Fallback for unmatched documents

- Uses keyword matching (case-insensitive substring search)
- Extensible: new categories can be added easily

### 5. Hierarchy Builder (`ehrx/hierarchy.py`)
**Class: `HierarchyBuilder`**

- Main orchestrator that coordinates all sub-components
- Processing pipeline:
  1. Detect document labels for each page
  2. Group consecutive pages with same label
  3. For each document group:
     - Collect all elements
     - Detect sections/subsections
     - Map to category
  4. Build final JSON hierarchy structure

**Output Schema:**
```json
{
  "documents": [
    {
      "document_type": "Clinical Notes",
      "category": "Notes",
      "page_range": [5, 7],
      "pages": [5, 6, 7],
      "extra_labels": [],
      "sections": [
        {
          "heading": "Visit Summary",
          "level": 1,
          "page": 5,
          "children": [...],
          "sections": [...]
        }
      ]
    }
  ],
  "categories": ["Notes", "Labs", "Meds"],
  "total_documents": 3,
  "total_pages": 10
}
```

### 6. Visual Debug Output (`ehrx/visualize.py`)
**Class: `HierarchyVisualizer`**

Generates annotated images for debugging and validation:

#### Label Detection Visualization
- Shows detection region (semi-transparent overlay)
- Highlights detected label with orange border
- Shows all text block candidates in green
- Displays detected label text

#### Section Visualization
- Color-codes elements by type:
  - **Red**: H1 sections
  - **Dark Orange**: H2 subsections
  - **Yellow**: H3+ sections
  - **Green**: Text blocks
  - **Blue**: Tables
  - **Magenta**: Figures
- Shows section heading text as annotations
- Includes legend for element types

#### Document Overview
- Multi-page thumbnail grid
- Color-coded borders by document group
- Shows document type and page range
- Legend mapping colors to documents

#### Text Summary
- Hierarchical text outline of structure
- Shows document types, categories, page ranges
- Lists sections and subsection counts
- Saved as `{doc_id}_hierarchy_summary.txt`

### 7. CLI Integration (`ehrx/cli.py`)

**New Command-Line Options:**
```bash
python -m ehrx.cli \
  --in input.pdf \
  --out results/ \
  --pages 1-10 \
  --build-hierarchy true    # Enable hierarchy building (default: true)
  --debug-viz               # Generate visual debug output (default: false)
```

**Processing Pipeline Updates:**
- **Pass 1**: Layout detection for column analysis (unchanged)
- **Pass 2**: Enhanced element processing with OCR (unchanged)
  - **New**: Stores elements for hierarchy building
  - **New**: Stores page images for visualization
- **Pass 3**: Build document hierarchy
  - Runs `HierarchyBuilder.build_hierarchy()`
  - Detects labels, groups documents, finds sections
  - Maps to categories
- **Pass 4**: Generate debug visualizations (if `--debug-viz` enabled)
  - Label detection images
  - Section visualization images
  - Document overview
  - Text summary
- **Finalize**: Serialize with hierarchy
  - Includes hierarchy in `document.index.json`
  - Updates JSON schema to include hierarchical structure

**Output Files:**
- `document.elements.jsonl`: Element data (flat, unchanged)
- `document.index.json`: **Updated to include hierarchy**
- `assets/`: Cropped images, tables (unchanged)
- `debug/`: **New** - Visual debug outputs
  - `page_XXXX_label_detection.png`
  - `page_XXXX_sections.png`
  - `{doc_id}_document_overview.png`
  - `{doc_id}_hierarchy_summary.txt`

## Test Suite (`tests/test_hierarchy.py`)

### Unit Tests (All Passing ✓)
- **TestDocumentLabelDetector**: 4 tests
  - Initialization with config
  - Label detection with mock data
  - Handling missing labels
  - Detection region calculation

- **TestDocumentGrouper**: 3 tests
  - Grouping consecutive pages
  - Handling unlabeled pages
  - Empty input handling

- **TestSectionDetector**: 4 tests
  - Initialization
  - Keyword-based heading classification
  - Caps-ratio based classification
  - Section hierarchy building

- **TestCategoryMapper**: 5 tests
  - Mapping various document types to categories
  - Handling unlabeled/unknown documents

### Integration Test
- **TestHierarchyBuilderIntegration**:
  - End-to-end test with real PDF processing
  - Layout detection + OCR + hierarchy building
  - Visual debug output generation
  - **Note**: Currently blocked by PyTorch 2.8 model loading issue
  - Alternative standalone test script provided

## Known Issues

### PyTorch 2.8 Compatibility
**Issue**: Detectron2 model loading fails with PyTorch 2.8 due to `weights_only=True` default.

**Error**:
```
Weights only load failed. In PyTorch 2.6, we changed the default value of 
the `weights_only` argument in `torch.load` from `False` to `True`.
```

**Attempted Fix**: Added monkey-patch in `ehrx/detect.py` to set `weights_only=False`, but cached model file appears corrupted.

**Current Status**: Model loading is failing, blocking full integration testing.

**Workaround Options**:
1. **Downgrade PyTorch** to 2.5 or earlier
2. **Clear cache and re-download** model (requires network access and proper permissions)
3. **Use local model file** with explicit path in config
4. **Switch to PaddleDetection** backend (user mentioned this also didn't work)

**For Testing Without Model**:
- Unit tests all pass (don't require model loading)
- Standalone test script provided: `test_hierarchy_standalone.py`
- Can run with `--dry-run` to see configuration

## Configuration

### Default Configuration (`configs/default.yaml`)

```yaml
hierarchy:
  heading_regex:
    - "^(PROBLEMS?|PROBLEM LIST)\\b"
    - "^(MEDICATIONS?|CURRENT MEDS?)\\b"
    - "^(ALLERG(IES|Y))\\b"
    - "^(LABS?|RESULTS?)\\b"
    - "^(VITALS?)\\b"
    - "^(IMAGING|RADIOLOGY)\\b"
    - "^(ASSESSMENT|PLAN|A/P)\\b"
    - "^NOTES?\\b"
  caps_ratio_min: 0.6
  gap_above_px: 18
  left_margin_tolerance_px: 20
  levels:
    H1: strong_keyword|big_gap
    H2: weaker_keyword|indented
    H3: bullets|numbered
```

### Extending Configuration

To add custom label detection parameters, add to config:

```yaml
label_detection:
  top_region: 0.15
  horizontal_start: 0.2
  horizontal_end: 0.8
  min_label_height: 15
  max_label_height: 80
  min_label_confidence: 0.5
```

## Usage Examples

### Basic Hierarchy Building
```bash
python -m ehrx.cli \
  --in medical_records.pdf \
  --out results/ \
  --pages 1-20
```

### With Debug Visualizations
```bash
python -m ehrx.cli \
  --in medical_records.pdf \
  --out results/ \
  --pages 1-20 \
  --debug-viz
```

### Custom Configuration
```bash
python -m ehrx.cli \
  --in medical_records.pdf \
  --out results/ \
  --config custom_config.yaml \
  --debug-viz
```

### Disable Hierarchy (Flat Structure Only)
```bash
python -m ehrx.cli \
  --in medical_records.pdf \
  --out results/ \
  --build-hierarchy false
```

## Future Enhancements

### From PRD (Not Yet Implemented)
1. **Margin Detection**: Classify and filter marginal content
2. **Margin Deduplication**: Consolidate repeated headers/footers across pages
3. **Semantic Embeddings**: Use NLP models for improved section continuity across pages
4. **Table Normalization**: Structured table parsing into rows/columns

### Suggested Improvements
1. **Machine Learning Models**: Train custom models for heading detection
2. **Font Detection**: Extract actual font information when available (not just height)
3. **Page Layout Analysis**: Better handling of multi-column documents
4. **Cross-Page Section Continuation**: Detect when sections span multiple pages
5. **Confidence Scores**: Add confidence scores to hierarchy decisions

## Code Structure

```
ehrx/
├── hierarchy.py          # Main hierarchy building logic (760 lines)
│   ├── DocumentLabelDetector
│   ├── DocumentGrouper
│   ├── SectionDetector
│   ├── CategoryMapper
│   └── HierarchyBuilder
├── visualize.py          # Visual debug output (470 lines)
│   └── HierarchyVisualizer
└── cli.py                # CLI integration (updated)

tests/
├── test_hierarchy.py     # Comprehensive test suite
└── ...

test_hierarchy_standalone.py  # Standalone testing script

configs/
└── default.yaml          # Configuration with hierarchy section
```

## Summary of Changes

### New Files Created
1. `ehrx/hierarchy.py` - Complete hierarchy building pipeline
2. `ehrx/visualize.py` - Visual debug output generation
3. `tests/test_hierarchy.py` - Comprehensive test suite
4. `test_hierarchy_standalone.py` - Standalone test script
5. `HIERARCHY_IMPLEMENTATION.md` - This documentation

### Modified Files
1. `ehrx/cli.py` - Integrated hierarchy building into main pipeline
2. `ehrx/detect.py` - Added PyTorch 2.8 compatibility patch
3. `configs/default.yaml` - Already had hierarchy section

### Test Results
- ✅ All unit tests passing (16/16)
- ❌ Integration test blocked by model loading issue
- ✅ Code structure validated
- ✅ Configuration loading working
- ✅ Visual debug output module complete

## Next Steps

1. **Resolve Model Loading Issue**:
   - User needs to fix PyTorch/Detectron2 model loading
   - Try downgrading PyTorch or clearing cache
   - Consider using local model weights

2. **Run Integration Test**:
   - Once model loads, run full end-to-end test
   - Validate on SENSITIVE_ehr2_copy.pdf or available PDFs
   - Generate visual outputs for validation

3. **Iterate on Heuristics**:
   - Review visual outputs to tune detection parameters
   - Adjust heading detection thresholds
   - Refine category keywords

4. **Add Missing Features**:
   - Implement margin detection if needed
   - Add margin deduplication
   - Consider semantic embeddings for section continuity

## Contact & Questions

For questions about this implementation, refer to:
- `hierarchyPRD.md` - Original product requirements
- `SPECS.md` - Project specifications
- Code comments in `ehrx/hierarchy.py` and `ehrx/visualize.py`

