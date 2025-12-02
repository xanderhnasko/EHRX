# Hierarchy Structure Fixes - Implementation Summary

## Changes Made

### 1. **Updated JSON Hierarchy Structure**

**Previous Structure:**
```json
{
  "documents": [...],
  "categories": ["Notes", "Labs"],
  "total_documents": 3,
  "total_pages": 20
}
```

**New Structure:**
```json
{
  "ehr_id": "SENSITIVE_ehr2_copy",
  "total_pages": 20,
  "total_documents": 3,
  "categories": {
    "Notes": {
      "Visit Summaries": [
        {
          "document_name": "Coding Summary",
          "page_range": [0, 2],
          "pages": [
            {
              "page_num": 0,
              "elements": [
                {
                  "id": "E_0001",
                  "type": "text_block",
                  "text": "...",
                  "bbox_px": [...],
                  "bbox_pdf": [...],
                  "page": 0,
                  "confidence": 0.95
                }
              ]
            }
          ]
        }
      ],
      "Progress Notes": [...]
    },
    "Labs": {
      "documents": [...]  // No subcategories
    }
  }
}
```

**Hierarchy Levels:**
1. **Categories** (Demographics, Vitals, Orders, Meds, Notes, Labs)
2. **Subcategories** (only for Notes: Visit Summaries, Discharge Statements, Progress Notes)
3. **Documents** (grouped by document_name)
4. **Pages** (elements grouped by page_num)
5. **Elements** (individual text blocks, tables, figures)

### 2. **Removed Unnecessary Fields**

**Removed from document.index.json:**
- `manifest` (entire section)
- `column_layout`
- `stats`
- `sections` (old section detection structure)
- `children`
- `labels_used`

**Removed from elements:**
- `z_order`
- `column_index`
- `created_at`
- `detector_name`
- `detector_conf`
- `source`
- `payload` (text moved to top level)

**Kept in elements:**
- `id`
- `type`
- `text` (extracted from payload)
- `bbox_px`
- `bbox_pdf`
- `page`
- `confidence` (OCR confidence)

### 3. **Added Bounding Box Padding**

Added 5px padding to all bounding boxes before OCR cropping to prevent text cutoff.

**Implementation:**
- Padding is scaled proportionally with DPI (e.g., 5px at 200 DPI → 15px at 600 DPI)
- Applied to text blocks, tables, and figures
- Bounds are clipped to image dimensions to avoid out-of-bounds errors

```python
# Add 5px padding (scaled for DPI)
padding = int(5 * dpi_scale_factor)
h, w = page_image.shape[:2]
x0 = max(0, x0 - padding)
y0 = max(0, y0 - padding)
x1 = min(w, x1 + padding)
y1 = min(h, y1 + padding)
```

### 4. **File Organization**

**Moved Files:**
- `test_hierarchy_standalone.py` → `tests/test_hierarchy_standalone.py`
- `test_detection_visual.py` → `tests/test_detection_visual.py`
- `hierarchyPRD.md` → `docs/hierarchyPRD.md`

**Updated Imports:**
- Serializer now uses simplified `finalize(hierarchy, column_layout, elapsed)` signature
- Hierarchy builder returns new structure format
- CLI passes hierarchy directly to serializer

### 5. **Subcategory Detection Logic**

Added automatic subcategory detection for "Notes" category based on document name keywords:

```python
def _get_subcategory(document_type: str, category: str) -> Optional[str]:
    if category != "Notes":
        return None
    
    doc_type_lower = document_type.lower()
    
    if any(kw in doc_type_lower for kw in ["visit", "summary", "encounter"]):
        return "Visit Summaries"
    elif any(kw in doc_type_lower for kw in ["discharge", "discharge summary"]):
        return "Discharge Statements"
    elif any(kw in doc_type_lower for kw in ["progress", "progress note"]):
        return "Progress Notes"
    
    return None
```

### 6. **Element Cleaning**

Implemented `_clean_element()` method to extract and flatten element data:

```python
def _clean_element(element: Dict[str, Any]) -> Dict[str, Any]:
    # Extract text from payload
    text = ""
    confidence = 0.0
    
    payload = element.get("payload", {})
    if isinstance(payload, dict):
        text = payload.get("text", "")
        confidence = payload.get("confidence", 0.0)
        
        # For tables, get OCR lines
        if element.get("type") == "table":
            ocr_lines = payload.get("ocr_lines", [])
            if ocr_lines:
                text = "\n".join(ocr_lines)
    
    return {
        "id": element.get("id", ""),
        "type": element.get("type", ""),
        "text": text,
        "bbox_px": element.get("bbox_px", []),
        "bbox_pdf": element.get("bbox_pdf", []),
        "page": element.get("page", 0),
        "confidence": float(confidence)
    }
```

## Modified Files

1. **`ehrx/hierarchy.py`**:
   - Updated `_build_output_structure()` - new hierarchy format
   - Added `_get_subcategory()` - subcategory detection
   - Added `_clean_element()` - element field filtering
   - Removed `_serialize_sections()` - no longer needed

2. **`ehrx/serialize.py`**:
   - Updated `finalize()` signature - simplified parameters
   - Updated `build_index()` - clean output structure
   - Removed manifest and column_layout from output

3. **`ehrx/cli.py`**:
   - Added 5px padding to OCR cropping (text blocks)
   - Added 5px padding to OCR cropping (tables/figures)
   - Updated finalize call with new signature

## Testing

### To Test the Changes:

```bash
# Run on first 20 pages with hierarchy
python -m ehrx.cli \
  --in SENSITIVE_ehr2_copy.pdf \
  --out results_test/ \
  --pages 1-20

# With debug visualizations
python -m ehrx.cli \
  --in SENSITIVE_ehr2_copy.pdf \
  --out results_test/ \
  --pages 1-20 \
  --debug-viz
```

### Expected Output:

**`results_test/document.index.json`**:
```json
{
  "ehr_id": "SENSITIVE_ehr2_copy",
  "total_pages": 20,
  "total_documents": 5,
  "categories": {
    "Notes": {
      "Visit Summaries": [
        {
          "document_name": "Coding Summary",
          "page_range": [0, 2],
          "pages": [
            {
              "page_num": 0,
              "elements": [...]
            }
          ]
        }
      ]
    },
    "Labs": {
      "documents": [...]
    }
  }
}
```

**`results_test/document.elements.jsonl`**:
- Remains unchanged (one element per line)
- Still contains all elements in flat format

## Known Issues

### 1. **Title Detection**
**Issue**: LayoutParser sometimes classifies centered title boxes as "table" instead of "text_block", preventing proper document label detection.

**Status**: Deferred per user request ("Ignore fixing LayoutParser reading for now")

**Potential Fix**: Post-process detected tables in label detection region:
```python
# If table is in label region and has short text, reclassify as text_block
if block.type == "table" and is_in_label_region and len(text) < 100:
    block.type = "text_block"
```

### 2. **PyTorch Model Loading**
**Issue**: Detectron2 model loading still fails with PyTorch 2.8.

**Status**: Ongoing - user needs to resolve separately

## Migration Notes

If you have existing code that reads the old hierarchy format:

**Old code:**
```python
for doc in data["documents"]:
    doc_type = doc["document_type"]
    category = doc["category"]
    for section in doc["sections"]:
        ...
```

**New code:**
```python
for category_name, category_data in data["categories"].items():
    # Handle subcategories if present
    for key, value in category_data.items():
        if key == "documents":
            # No subcategory
            for doc in value:
                doc_name = doc["document_name"]
                for page in doc["pages"]:
                    for element in page["elements"]:
                        ...
        else:
            # Has subcategory
            subcategory_name = key
            for doc in value:
                ...
```

## Summary

All requested changes have been implemented:
- ✅ JSON hierarchy restructured with proper nesting
- ✅ Unnecessary fields removed from output
- ✅ 5px padding added to bounding boxes
- ✅ Test files moved to proper locations
- ✅ PRD moved to docs/ folder
- ⏸️ Title detection fix deferred per user request

The system now produces clean, hierarchical JSON output suitable for downstream processing and UI rendering.

