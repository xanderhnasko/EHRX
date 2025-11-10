This MVP is intentionally simple, reproducible, and PHI-safe. The only deliverable is a **script** that ingests a **scanned EHR PDF** and emits a **schematized, hierarchical JSON** (+ referenced assets) you can later feed to a DB, webapp, or RAG system.

---

# 0) Scope (MVP v1)

- **Input:** scanned EHR PDF (hundreds of pages OK).
- **Output:**
    - `document.elements.jsonl` — flat list of detected elements (text blocks, tables, figures, handwriting), each with `page`, `bbox`, `type`, and payload.
    - `document.index.json` — small file with the **hierarchy** (sections/subsections referencing element IDs) + manifest (run config, sizes, model versions).
    - `assets/` — cropped images saved for tables/figures/handwriting (and optional table CSVs).
- **Privacy:** local-only by default; no cloud calls; no PHI in logs.
- **Non-goals (for MVP):** clinical coding (LOINC/SNOMED), curve digitization from graphs, handwriting OCR quality guarantees.

---

# 1) Architecture (one-pass streaming)

```
PDF → (page raster) → Layout detection → Region routing
    → [text] OCR
    → [table] OCR + (heuristic structure)
    → [figure/handwriting] crop image (+ optional OCR)
    → Per-page JSONL append
    → After all pages: deterministic hierarchy build → index.json

```

**Principles**

- **LayoutParser = spine** for region detection + coarse reading order.
- **Deterministic hierarchy** built by simple heading rules (not clustering).
- **Vector-first, OCR-second:** if a page unexpectedly has vector text, use it; otherwise OCR just the cropped regions.
- **Everything has provenance:** always keep `page`, `bbox` (PDF + pixels), and `z_order`.

---

# 2) Dependencies (pin for sanity)

- Python 3.11
- `layoutparser` (with one detector backend):
    - **Option A (default):** Detectron2 CPU
        - `pip install 'torch==2.3.*' 'torchvision==0.18.*' --index-url https://download.pytorch.org/whl/cpu`
        - `pip install layoutparser detectron2==0.6`
    - **Option B (fallback):** PaddleDetection backend
        - `pip install layoutparser[paddledetection] paddlepaddle`
- OCR: `pytesseract`, system `tesseract-ocr` (≥ 5.0), `opencv-python`
- PDF raster: `pymupdf` (for page size/coords + fast raster), or `pdf2image`
- Tables (optional CSVs): `pandas`
- CLI/config: `typer` or `argparse`, `pydantic`, `pyyaml`

> If Detectron2 install is painful on a teammate’s machine, switch to the Paddle backend (one line change in config).
> 

---

# 3) Repo layout

```
ehrx/
  ehrx/
    __init__.py
    cli.py                 # entrypoint
    config.py              # load/validate YAML config
    pager.py               # PDF → page raster + coord mapping
    detect.py              # LayoutParser model wrapper
    route.py               # element routing (text/table/figure/handwriting)
    ocr.py                 # Tesseract wrappers + preprocessing
    tables.py              # OCR-based table heuristics + optional CSV
    hierarchy.py           # deterministic tree builder
    serialize.py           # write JSONL + index.json + assets
    utils.py               # bbox utils, id gen, logging, timers
  configs/default.yaml
  tests/smoke_test.py
  README.md

```

---

# 4) CLI (zero surprises)

```
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

- `-detector`: `pubLayNet` (Detectron2) or `paddle_ppstructure` (Paddle).
- `-ocr`: `tesseract` (only).
- `-allow-vector`: try vector text inside region before OCR.
- `-pages`: `"all"` or `"1-50,120-130"`.
- `-threads`: per-page/region OCR pool size.

---

# 5) Config (YAML) — edit without touching code

```yaml
detector:
  backend: detectron2   # or paddle
  model: "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
  label_map: { "Text": "text_block", "Table": "table", "Figure": "figure" }
  min_conf: 0.5
  nms_iou: 0.3

ocr:
  engine: tesseract
  psm_text: 6           # uniform block
  psm_table: 6
  lang: eng
  preprocess: { deskew: true, binarize: true }

tables:
  csv_guess: true       # heuristic row/col if grid not found
  min_row_height_px: 10
  y_cluster_tol_px: 6
  x_proj_bins: 40

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
  levels: { H1: strong_keyword|big_gap, H2: weaker_keyword|indented, H3: bullets|numbered }

privacy:
  local_only: true
  log_text_snippets: false   # never log OCR text

```

---

# 6) Element types & routing rules (MVP)

**Common fields on every element**

```
id, doc_id, page, type ∈ {text_block, table, figure, handwriting},
bbox_pdf [x0,y0,x1,y1], bbox_px [x0,y0,x1,y1], rotation, z_order,
source ∈ {vector, ocr}, created_at, detector_name, detector_conf

```

### text_block

- **Extract:** try vector text within bbox (if `allow_vector`); else OCR cropped region (`psm_text=6`).
- **Payload:** `{ text, tokens? }` (tokens optional for MVP).

### table

- **Extract:**
    - OCR cropped region (`psm_table=6`) to get line text.
    - **Grid if visible:** use simple OpenCV line detection to infer rows/cols.
    - **Else heuristic:** y-cluster words into rows; x-projection to split columns.
- **Payload:**
    
    `{ headers: [str]? , rows: [[str]]? , csv_ref? , ocr_lines: [str] }`
    
    - If structure is unclear, keep `ocr_lines` and a best-effort `rows`.
    - Save table crop to `assets/table_<id>.png`; if rows built, also write `assets/table_<id>.csv`.

### figure

- **Extract:** save crop `assets/figure_<id>.png`; try to attach nearby caption (scan text blocks ±300 px vertically in same column).
- **Payload:** `{ image_ref, caption? }`

### handwriting

- **Extract:** save crop `assets/hand_<id>.png`; try OCR if enabled; record `ocr_confidence`.
- **Payload:** `{ image_ref, ocr_text?, ocr_confidence? }`

---

# 7) Reading order & columns (simple and robust)

- **Reading order (per page):** sort detected blocks by `x` (column index) then by `y` (top to bottom).
- **Column detection:** cluster block left-edges (`k=1..3`, pick k by gap heuristic).
- Store `z_order` as the block’s ordinal in that page’s reading stream.

---

# 8) Deterministic hierarchy builder (no ML)

**Inputs:** flat `elements.jsonl` (text blocks include OCR text).

**Goal:** envelopes (sections/subsections) referencing element IDs.

Algorithm (single pass over reading stream):

1. **Heading detection (per text_block):**
    - **Lexical:** match `heading_regex` on the first 1–2 lines (case-insensitive).
    - **Visual proxies (scanned-safe):** line bbox height vs page median; ALL-CAPS ratio; whitespace `gap_above_px`; left-margin alignment.
    - Score → assign `H1/H2/H3` if above thresholds.
2. **Stacked envelopes:**
    - On `H1`: pop to root, open new H1 envelope.
    - On `H2`: pop to last H1, open H2.
    - On `H3`: pop to last H2, open H3.
    - Otherwise: attach element to current top envelope.
3. **Continuation across pages:** a repeated H1 within top 10–15% of the page and same text → treat as continuation (don’t open a new one).
4. **Tables/figures:** attach to nearest preceding heading **in same column**; else attach to current top envelope.

**Output (index.json):**

```json
{
  "doc_id": "scan-001",
  "manifest": {"pages": 642, "detector": "pubLayNet", "ocr": "tesseract", "created_at": "..."},
  "hierarchy": [
    {"id":"H1_0001","label":"PROBLEM LIST","children":["E_0012","E_0013","H2_0101", "..."]},
    {"id":"H1_0002","label":"MEDICATIONS","children":[ "..."]}
  ],
  "labels_used": ["PROBLEM LIST","MEDICATIONS","ALLERGIES","LABS","VITALS","IMAGING","NOTES"]
}

```

---

# 9) Serialization (stable & scalable)

## `document.elements.jsonl` (one JSON per line)

Example records:

```json
{"id":"E_0012","doc_id":"scan-001","page":1,"type":"text_block",
 "bbox_pdf":[72,118,540,210],"bbox_px":[95,156,712,278],"rotation":0,"z_order":3,
 "source":"ocr","detector_name":"pubLayNet","detector_conf":0.86,
 "payload":{"text":"PROBLEM LIST:\n1. Type 2 diabetes mellitus...\n"}}

{"id":"E_0045","doc_id":"scan-001","page":3,"type":"table",
 "bbox_pdf":[72,220,540,580],"bbox_px":[95,290,712,765],"rotation":0,"z_order":9,
 "source":"ocr","detector_name":"pubLayNet","detector_conf":0.79,
 "payload":{"headers":["Test","Result","Units","Ref"],"rows":[["WBC","9.8","10^3/uL","4.0–10.0"],["Hgb","12.1","g/dL","12–16"]],
            "csv_ref":"assets/table_E_0045.csv","ocr_lines":["Test Result Units Ref","WBC 9.8 10^3/uL 4.0–10.0","Hgb 12.1 g/dL 12–16"]}}

```

## `document.index.json`

(see example in §8)

## `assets/`

- `table_<id>.png`, optional `table_<id>.csv`
- `figure_<id>.png`
- `hand_<id>.png`

**Coordinate policy:** always store **both** `bbox_pdf` (points) and `bbox_px` (pixels) + page pixel size and scale in the **first element record of each page** or manifest.

---

# 10) Performance notes (works on 600+ pages)

- Raster each page at **150–200 DPI** for detection; crop regions at **300 DPI** for OCR.
- Stream: write JSONL **as you go**; don’t hold whole doc in memory.
- Thread pool for OCR over regions (`threads=4` default).
- Keep detector on CPU unless you have a GPU; it’s fine for MVP—just batch pages (e.g., 10 at a time).

---

# 11) Logging & PHI

- `INFO`: counts, timings, element types per page, IDs.
- `DEBUG` (opt-in): bbox coords, detector scores.
- **Never** log OCR text or images.
- Optional `-redact-preview` later can draw black boxes on page images (not part of MVP).

---

# 12) Acceptance checklist (copy into your PR)

- [ ]  Runs in **local-only** mode; no external calls.
- [ ]  Processes a 600+ page scanned PDF without OOM.
- [ ]  Produces `elements.jsonl`, `index.json`, and `assets/` with ≥ 3 H1 sections.
- [ ]  Every element has `page`, `bbox_pdf`, `bbox_px`, `type`, and `source`.
- [ ]  At least one table produces `rows` **or** falls back to `ocr_lines` + `image_ref`.
- [ ]  Hierarchy references **only** valid element IDs.
- [ ]  No PHI in logs; config captured in `manifest`.

---

# 13) Minimal pseudo-code (so nobody gets stuck)

```python
# cli.py
def extract(in_pdf, out_dir, cfg):
    manifest = init_run(in_pdf, out_dir, cfg)
    writer = JsonlWriter(out_dir/"document.elements.jsonl")
    page_iter = Pager(in_pdf).pages()
    for page in page_iter:
        img, scale = rasterize(page, dpi=200)
        blocks = detect_layout(img, cfg.detector)           # → list[{bbox_px, label, score}]
        blocks = postprocess_blocks(blocks, cfg.detector)
        for z, blk in enumerate(order(blocks)):
            crop = crop_image(img, blk.bbox_px)
            elem = route_element(blk, crop, page, scale, cfg)  # → dict(common+payload)
            writer.append(elem)
    hierarchy = build_hierarchy(out_dir/"document.elements.jsonl", cfg.hierarchy)
    write_index(out_dir/"document.index.json", manifest, hierarchy)

```

---

# 14) Future-safe knobs (do **not** build now)

- Summaries/normalization via LLM on **sanitized** text.
- Handwriting model; table structure upgrades (DeepDeSRT, PubTables-1M, etc.).
- FHIR/OMOP mapping; DB schemas; web viewer with highlight-on-click.

---

This is everything you need to start coding immediately, with clear division of labor (pager/detect/ocr/tables/hierarchy/serialize), reproducible configs, and outputs that are future-proof for your UI/DB/RAG phases.