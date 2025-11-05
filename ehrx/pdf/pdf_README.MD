
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