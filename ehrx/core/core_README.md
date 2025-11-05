

### `ehrx/utils.py`
- Purpose: Shared utility primitives for geometry, IDs, timing, logging, manifests, and path checks.
- How it works: Independent helpers and small classes, no external state except a module-scoped logger created by `setup_logging`.

Key classes/functions
- `BBox`
  - **What**: Convenience class for rectangular regions.
  - **How**: Stores `x0,y0,x1,y1` and exposes computed properties (`width`, `height`, `area`, `center`), conversions (`to_list`, `to_dict`, `from_list`), transforms (`scale`), and spatial ops (`intersects`, `intersection`, `iou`).
- `pdf_to_pixel_coords(bbox_pdf, page_height_pdf, page_height_px, scale)` and `pixel_to_pdf_coords(bbox_px, page_height_pdf, page_height_px, scale)`
  - **What**: Coordinate conversions between PDF space (origin bottom-left, 72 DPI) and pixel space (origin top-left).
  - **How**: Applies scale and flips Y using page height so that tops and bottoms are mapped consistently.
- `IDGenerator(doc_id)`
  - **What**: Deterministic, incremental ID counter for elements and headings.
  - **How**: Maintains counters; yields IDs like `E_0001` or `H1_0001` based on requested level; `reset` clears counters.
- `Timer(name, logger=None)`
  - **What**: Context manager for timing blocks.
  - **How**: Records start/end times; logs duration on exit; `elapsed` returns seconds since start (or until end if closed).
- `setup_logging(level="INFO", log_text_snippets=False)`
  - **What**: Central logger setup for the project with PHI controls.
  - **How**: Configures an `ehrx` logger with a console handler and formatter; removes previous handlers; stores a custom attribute `log_text_snippets` to gate text logging.
- `safe_log_text(logger, text, max_length=50)`
  - **What**: PHI-safe text rendering for logs.
  - **How**: If `logger.log_text_snippets` is false, returns a length-only placeholder; otherwise outputs a possibly truncated quote.
- `create_manifest(doc_id, input_path, config)`
  - **What**: Captures run metadata for traceability.
  - **How**: Returns a dict with doc ID, path, timestamp, detector/OCR selections, and a short hash of the config.
- `ensure_output_dir(output_path)`
  - **What**: Idempotent output directory creation.
  - **How**: Ensures path and an `assets` subdirectory exist; returns `Path`.
- `validate_pdf_path(pdf_path)`
  - **What**: Input validation for PDFs.
  - **How**: Verifies existence and `.pdf` suffix; returns `Path` or raises.

### `ehrx/config.py`
- Purpose: Strict configuration schema and loading/validation for all processing knobs.
- How it works: Pydantic `BaseModel`s with defaults and field validators; loader reads YAML, applies model validation, and forbids unknown fields.

Key classes/functions
- `DetectorConfig`
  - **What**: Layout detection parameters.
  - **How**: Fields for `backend` (`detectron2` or `paddle`), model URI, label map, confidence/NMS thresholds; validates `backend`.
- `OCRPreprocessConfig`
  - **What**: Pre-OCR toggles.
  - **How**: `deskew`, `binarize` booleans.
- `OCRConfig`
  - **What**: OCR engine settings.
  - **How**: Fields for `engine` (validated to `tesseract`), page segmentation modes (`psm_text`, `psm_table`, each 0–13), language, and `preprocess` subconfig.
- `TablesConfig`
  - **What**: Heuristics for table structure extraction.
  - **How**: Controls CSV guessing, minimal row height, clustering tolerance, and projection bins.
- `HierarchyLevelsConfig` and `HierarchyConfig`
  - **What**: Heading detection and document hierarchy rules.
  - **How**: Default regexes for clinical headings, caps ratio threshold, gap/margin tolerances, and level strategies (`H1`, `H2`, `H3`).
- `PrivacyConfig`
  - **What**: Privacy behavior flags.
  - **How**: `local_only` and `log_text_snippets`.
- `EHRXConfig`
  - **What**: Root config aggregating all subconfigs.
  - **How**: Sets `model_config = extra="forbid"` to reject unknown keys.
- `load_config(config_path=None)`
  - **What**: Read and validate YAML into `EHRXConfig`.
  - **How**: Loads provided path or empty dict for defaults; raises on missing file or validation errors.
- `find_default_config()` and `load_default_config()`
  - **What**: Discovery and loading of default config.
  - **How**: Searches common locations (`configs/default.yaml`, `config.yaml`, `ehrx.yaml`, or repo `configs/default.yaml`); loads if found.
- `validate_environment(config)`
  - **What**: Dependency checks tailored to the config.
  - **How**: Verifies Tesseract availability (via `pytesseract`), the selected detector stack (`detectron2`/`paddle` + `layoutparser`), at least one PDF library (`PyMuPDF` or `pdf2image`), and OpenCV; returns a list of error messages.
- `setup_logging_from_config(config, level="INFO")`
  - **What**: Centralized logging initialization tied to privacy config.
  - **How**: Delegates to `utils.setup_logging`, passing `config.privacy.log_text_snippets`.

### `ehrx/pager.py`
- Purpose: Turn PDFs into page images with consistent coordinate mapping and optional vector text, abstracting over multiple backends.
- How it works: Chooses PyMuPDF when available (preferred), otherwise pdf2image. Provides a generator to iterate requested pages as numpy arrays plus page metadata and a coordinate mapper.

Key classes/functions
- `PageInfo`
  - **What**: Metadata for a rasterized page.
  - **How**: Holds PDF and pixel dimensions, DPI, rotation; exposes `scale_x` and `scale_y` for coordinate conversions.
- `CoordinateMapper(page_info)`
  - **What**: PDF↔pixel conversion consistent with the page’s geometry.
  - **How**: `pdf_to_pixel` and `pixel_to_pdf` flip Y using `height_pdf` and scale using `scale_x/scale_y` from `PageInfo`.
- `PDFRasterizer(pdf_path)`
  - **What**: Backend facade for rasterizing pages.
  - **How**:
    - Backend selection: PyMuPDF if importable; else pdf2image; else raise.
    - `_rasterize_page_pymupdf(page_num, dpi)`: uses `fitz` to render a page at scale `dpi/72`, converts samples to an RGB numpy array, and returns `PageInfo`.
    - `_rasterize_page_pdf2image(page_num, dpi)`: renders the specific page via `convert_from_path`, converts PIL image to RGB numpy array; estimates PDF dimensions from pixel size and DPI to produce `PageInfo`.
    - `extract_vector_text(page_num)`: for PyMuPDF only, pulls spans from `page.get_text("dict")` into a flat list with `bbox`, `text`, and indices.
    - `close()`: closes underlying document if present.
- `parse_page_range(page_range, total_pages)`
  - **What**: Flexible page-range parser.
  - **How**: Accepts `"all"`, single numbers, and comma-separated ranges like `"1-5,8-10"`; returns sorted, deduped 0-indexed list bounded by total pages.
- `Pager(pdf_path)`
  - **What**: User-facing iteration API over pages.
  - **How**:
    - `pages(page_range="all", dpi=150)`: yields `(image_array, page_info, coordinate_mapper)` for each requested page; logs progress.
    - `get_page_vector_text(page_num)`: convenience pass-through.
    - `close()`: closes rasterizer resources.

### How they fit together
- `config.py` defines and loads the run configuration, ensures dependencies are present, and initializes PHI-aware logging.
- `pager.py` uses available PDF backends to rasterize pages and provides precise mapping between PDF coordinates and pixels, plus vector text when possible.
- `utils.py` supplies supporting primitives (IDs, timers, logging helpers, manifests, filesystem checks, geometry ops) used across processing steps.

- Made no code changes; provided a concise high-level and functional summary of `utils.py`, `config.py`, and `pager.py`, and how they interoperate as the core runtime scaffold.