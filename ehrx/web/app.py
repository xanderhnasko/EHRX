"""
FastAPI service for PDF upload, extraction, and querying.

This is a synchronous MVP; long-running extraction will block the request.
Deploy behind Cloud Run/uvicorn as needed.
"""

import os
import re
import logging
import tempfile
import uuid
import json
from pathlib import Path
from typing import Optional, Tuple, List
from threading import Lock
from collections import OrderedDict
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from ehrx.db.config import DBConfig
from ehrx.db.client import init_schema
from ehrx.web.db import DB
from ehrx.web.storage import GCSClient
from ehrx.vlm.pipeline import DocumentPipeline
from ehrx.vlm.grouping import SubDocumentGrouper, generate_hierarchical_index
from ehrx.vlm.config import VLMConfig
from ehrx.agent.query import HybridQueryAgent

load_dotenv()

# Basic logging so pipeline logs show up in Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(title="PDF2EHR API")

# CORS for local dev and deployed frontend (set FRONTEND_ORIGINS as comma-separated list)
allowed_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173"
).split(",")
allowed_origins = [o.strip() for o in allowed_origins if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_config = DBConfig.from_env()
db = DB(db_config)

GCS_BUCKET = os.getenv("GCS_BUCKET")
if not GCS_BUCKET:
    raise RuntimeError("GCS_BUCKET env var is required")
gcs = GCSClient(GCS_BUCKET)

PAGE_IMAGES_PUBLIC = os.getenv("PAGE_IMAGES_PUBLIC", "false").lower() == "true"


def _png_dimensions(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Fast PNG dimension reader with Pillow fallback.
    """
    try:
        with path.open("rb") as f:
            header = f.read(24)
            if len(header) >= 24 and header.startswith(b"\x89PNG\r\n\x1a\n"):
                w = int.from_bytes(header[16:20], "big")
                h = int.from_bytes(header[20:24], "big")
                if w and h:
                    return w, h
    except Exception:
        pass
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None, None


class SchemaCache:
    """Tiny thread-safe LRU for JSON schema blobs keyed by storage URL."""

    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._lock = Lock()
        self._data: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self.max_size:
                self._data.popitem(last=False)


SCHEMA_CACHE = SchemaCache(
    max_size=int(os.getenv("SCHEMA_CACHE_SIZE", "128"))
)


class ReconstructionCache:
    """LRU for reconstructed sections so we don't re-run LLM per request."""

    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._lock = Lock()
        self._data: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self.max_size:
                self._data.popitem(last=False)


RECONSTRUCTION_CACHE = ReconstructionCache(
    max_size=int(os.getenv("RECONSTRUCTION_CACHE_SIZE", "128"))
)
RECON_PROMPT_VERSION = "v4"

class QueryRequest(BaseModel):
    document_id: str
    question: str
    kind: str | None = "enhanced"


@app.on_event("startup")
def _ensure_schema():
    # Create tables if missing
    from ehrx.db.client import get_conn

    with get_conn(db_config) as conn:
        init_schema(conn)


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, store in GCS, create a document row."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / file.filename
        content = file.file.read()
        tmp_path.write_bytes(content)

        dest = f"documents/{uuid.uuid4()}/{file.filename}"
        storage_url = gcs.upload_file(tmp_path, dest)

    doc_id = db.create_document(original_filename=file.filename, storage_url=storage_url)
    return {"document_id": str(doc_id), "storage_url": storage_url}


@app.post("/documents/{document_id}/extract")
def extract_document(document_id: str, page_range: Optional[str] = None):
    """Run extraction + grouping, upload JSONs to GCS, and store metadata."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id")

    doc = db.get_document(doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / Path(doc["original_filename"]).name
        gcs.download_to_path(doc["storage_url"], pdf_path)

        output_dir = Path(td) / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = DocumentPipeline(vlm_config=VLMConfig.from_env(), checkpoint_interval=50, dpi=200)
        document = pipeline.process_document(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir),
            page_range=page_range or "all",
            document_context={"document_type": "Clinical EHR"},
            save_page_images=True,
        )

        grouper = SubDocumentGrouper(confidence_threshold=0.80)
        enhanced_doc = grouper.group_document(document)
        index = generate_hierarchical_index(enhanced_doc)

        full_path = output_dir / f"{document['document_id']}_full.json"
        enhanced_path = output_dir / f"{document['document_id']}_enhanced.json"
        index_path = output_dir / f"{document['document_id']}_index.json"

        # Save enhanced/index files locally before upload
        with open(enhanced_path, "w") as f:
            json.dump(enhanced_doc, f, indent=2)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        full_url = gcs.upload_file(full_path, f"extractions/{document_id}/{full_path.name}")
        enhanced_url = gcs.upload_file(enhanced_path, f"extractions/{document_id}/{enhanced_path.name}")
        index_url = gcs.upload_file(index_path, f"extractions/{document_id}/{index_path.name}")

        # Upload page images (signed URLs; optional public mode via PAGE_IMAGES_PUBLIC)
        page_images_dir = output_dir / "pages"
        page_image_map = {}
        page_dim_map = {}
        if page_images_dir.exists():
            # Capture page dimension map from document (pages contain page_info)
            for page in document.get("pages", []):
                info = page.get("page_info", {})
                page_num = page.get("page_number")
                if page_num:
                    page_dim_map[str(int(page_num))] = {
                        "width_px": info.get("width_px"),
                        "height_px": info.get("height_px"),
                        "width_pdf": info.get("width_pdf"),
                        "height_pdf": info.get("height_pdf"),
                    }

            for img_path in sorted(page_images_dir.glob("page-*.png")):
                page_number = img_path.stem.split("-")[-1]
                dest = f"extractions/{document_id}/pages/{img_path.name}"
                if PAGE_IMAGES_PUBLIC:
                    public_url = gcs.upload_file(img_path, dest, make_public=True, return_public_url=True)
                    page_image_map[str(int(page_number))] = public_url
                else:
                    gcs.upload_file(img_path, dest, make_public=False)
                    # Try signed URL; if it fails, optionally fallback to public by toggling env
                    try:
                        signed_url = gcs.generate_signed_url(
                            dest,
                            expiration_seconds=7 * 24 * 3600,
                            content_type="image/png"
                        )
                        page_image_map[str(int(page_number))] = signed_url
                    except Exception as e:
                        logging.warning(f"Failed to sign page image {dest}: {e}")
                        continue
                # Fallback: if dimensions missing for this page, derive from the image file
                page_key = str(int(page_number))
                if page_key not in page_dim_map:
                    w, h = _png_dimensions(img_path)
                    if w and h:
                        page_dim_map[page_key] = {
                            "width_px": w,
                            "height_px": h,
                            "width_pdf": page_dim_map.get(page_key, {}).get("width_pdf") if page_key in page_dim_map else None,
                            "height_pdf": page_dim_map.get(page_key, {}).get("height_pdf") if page_key in page_dim_map else None,
                        }
                    else:
                        logging.warning(f"Failed to read dimensions for {img_path}")
        logging.info(
            f"Page images uploaded for doc {document_id}: count={len(page_image_map)}, dims={len(page_dim_map)}"
        )

    stats = document.get("processing_stats", {})
    metadata_common = {
        "index_url": index_url,
        "page_images": page_image_map,
        "page_dimensions": page_dim_map,
    }
    db.upsert_extraction(doc_uuid, "full", full_url, stats.get("total_pages"), stats.get("total_elements"), metadata_common)
    db.upsert_extraction(doc_uuid, "enhanced", enhanced_url, stats.get("total_pages"), stats.get("total_elements"), metadata_common)
    db.upsert_extraction(doc_uuid, "index", index_url, stats.get("total_pages"), stats.get("total_elements"), metadata_common)

    return {
        "document_id": document_id,
        "extractions": {
            "full": full_url,
            "enhanced": enhanced_url,
            "index": index_url,
        },
    }


@app.get("/documents/{document_id}")
def get_document(document_id: str):
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id")

    doc = db.get_document(doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    extractions = db.get_extractions(doc_uuid)
    return {"document": doc, "extractions": extractions}


@app.get("/healthz")
@app.get("/api/healthz")
@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/query")
@app.post("/api/query")
def query_document(payload: QueryRequest):
    """Run a query against an extraction (prefers enhanced)."""
    if not payload.question:
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        doc_uuid = uuid.UUID(payload.document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id")

    extractions = db.get_extractions(doc_uuid)
    # Deduplicate by kind, preferring newest (get_extractions is ordered DESC)
    by_kind = {}
    for e in extractions:
        if e["kind"] not in by_kind:
            by_kind[e["kind"]] = e

    extraction = by_kind.get(payload.kind) or by_kind.get("enhanced") or by_kind.get("full")
    if not extraction:
        raise HTTPException(status_code=404, detail="No extraction available for document")

    cache_key = f"{extraction['storage_url']}::{RECON_PROMPT_VERSION}"
    schema = SCHEMA_CACHE.get(cache_key)

    if not schema:
        with tempfile.TemporaryDirectory() as td:
            json_path = Path(td) / "schema.json"
            gcs.download_to_path(extraction["storage_url"], json_path)
            with open(json_path, "r") as f:
                schema = json.load(f)
            SCHEMA_CACHE.set(cache_key, schema)

    agent = HybridQueryAgent(schema=schema, vlm_config=VLMConfig.from_env())
    result = agent.query(payload.question)

    # Attach image URLs from extraction metadata if available
    meta = extraction.get("metadata") or {}
    page_images = meta.get("page_images") or {}
    page_dims = meta.get("page_dimensions") or {}
    for el in result.get("matched_elements", []):
        page_key = el.get("page_key") or (str(el.get("page")) if el.get("page") is not None else None)
        if not page_key:
            continue
        img_url = page_images.get(page_key)
        if img_url:
            el["image_url"] = img_url
        dims = page_dims.get(page_key)
        if dims:
            el["page_width_px"] = dims.get("width_px")
            el["page_height_px"] = dims.get("height_px")
            el["page_width_pdf"] = dims.get("width_pdf")
            el["page_height_pdf"] = dims.get("height_pdf")
    logging.info(
        f"Query {payload.question[:50]}... matched {len(result.get('matched_elements', []))} elements; "
        f"with images={len([m for m in result.get('matched_elements', []) if m.get('image_url')])}"
    )

    return JSONResponse(result)


@app.get("/api/documents/{document_id}/structured-data")
def get_structured_data(document_id: str, kind: str = "enhanced"):
    """
    Extract structured data for frontend tabs (Summary, Meds, Labs, Procedures).

    This endpoint processes the document schema and extracts:
    - Summary: Document-level summary
    - Medications: Parsed medication data with drug name, dosage, frequency, dates
    - Labs: Parsed lab data with test name, date ordered, reason
    - Procedures: Parsed procedure data with name, date, purpose, results
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id")

    extractions = db.get_extractions(doc_uuid)
    by_kind = {}
    for e in extractions:
        if e["kind"] not in by_kind:
            by_kind[e["kind"]] = e

    extraction = by_kind.get(kind) or by_kind.get("enhanced") or by_kind.get("full")
    if not extraction:
        raise HTTPException(status_code=404, detail="No extraction available for document")

    cache_key = extraction["storage_url"]
    schema = SCHEMA_CACHE.get(cache_key)

    if not schema:
        with tempfile.TemporaryDirectory() as td:
            json_path = Path(td) / "schema.json"
            gcs.download_to_path(extraction["storage_url"], json_path)
            with open(json_path, "r") as f:
                schema = json.load(f)
            SCHEMA_CACHE.set(cache_key, schema)

    # Extract structured data
    structured_data = {
        "summary": None,
        "medications": [],
        "labs": [],
        "procedures": [],
        "patient_demographics": schema.get("patient_demographics"),
        "sections": [],
        "pages": [],
        "document_dates": [],
    }

    # Extract summary from document_summary field
    if "document_summary" in schema:
        summary_elem = schema["document_summary"]
        structured_data["summary"] = summary_elem.get("content", "No summary available")

    # Heuristic document dates (loose; surface to UI for context)
    date_candidates = set()
    for page in schema.get("pages", []):
        for element in page.get("elements", []):
            content = element.get("content") or ""
            for match in re.findall(r"(\d{1,2}/\d{1,2}/\d{2,4})", content):
                date_candidates.add(match)
    structured_data["document_dates"] = sorted(date_candidates)

    # Capture grouped sections with their raw elements for reconstruction UI
    sections = []
    for subdoc in schema.get("sub_documents", []):
        section_pages = []
        for page in subdoc.get("pages", []):
            elems = []
            for element in page.get("elements", []):
                elems.append(
                    {
                        "type": element.get("type"),
                        "content": element.get("content"),
                        "page_number": page.get("page_number"),
                        "bbox_pixel": element.get("bbox_pixel"),
                        "bbox_pdf": element.get("bbox_pdf"),
                        "confidence": (element.get("confidence") or {}).get("overall"),
                        "needs_review": element.get("needs_review"),
                    }
                )
            section_pages.append({"page_number": page.get("page_number"), "elements": elems})
        sections.append(
            {
                "id": subdoc.get("id"),
                "title": subdoc.get("title"),
                "type": subdoc.get("type"),
                "page_range": subdoc.get("page_range"),
                "page_count": subdoc.get("page_count"),
                "confidence": subdoc.get("confidence"),
                "pages": section_pages,
            }
        )
    structured_data["sections"] = sections

    # Lightweight per-page snapshot to support timeline view
    page_summaries = []
    for page in schema.get("pages", []):
        elems = []
        for element in page.get("elements", []):
            elems.append(
                {
                    "type": element.get("type"),
                    "content": element.get("content"),
                    "bbox_pixel": element.get("bbox_pixel"),
                    "bbox_pdf": element.get("bbox_pdf"),
                    "confidence": (element.get("confidence") or {}).get("overall"),
                    "needs_review": element.get("needs_review"),
                }
            )
        page_summaries.append(
            {
                "page_number": page.get("page_number"),
                "page_info": page.get("page_info"),
                "elements": elems,
            }
        )
    structured_data["pages"] = page_summaries

    # Extract medications, labs, and procedures from sub_documents or pages
    if "sub_documents" in schema:
        for subdoc in schema.get("sub_documents", []):
            subdoc_type = subdoc.get("type", "")

            # Extract medications
            if subdoc_type == "medications":
                for page in subdoc.get("pages", []):
                    for element in page.get("elements", []):
                        if element.get("type") in ["medication_table", "clinical_paragraph"]:
                            # Parse medication data using Gemini
                            med_data = _parse_medication_element(element.get("content", ""))
                            if med_data:
                                structured_data["medications"].extend(med_data)

            # Extract labs
            elif subdoc_type == "laboratory_results":
                for page in subdoc.get("pages", []):
                    for element in page.get("elements", []):
                        if element.get("type") in ["lab_results_table", "clinical_paragraph"]:
                            # Parse lab data using Gemini
                            lab_data = _parse_lab_element(element.get("content", ""))
                            if lab_data:
                                structured_data["labs"].extend(lab_data)

            # Extract procedures
            elif subdoc_type == "procedures":
                for page in subdoc.get("pages", []):
                    for element in page.get("elements", []):
                        if element.get("type") in ["clinical_paragraph", "section_header", "general_table"]:
                            # Parse procedure data using Gemini
                            proc_data = _parse_procedure_element(element.get("content", ""))
                            if proc_data:
                                structured_data["procedures"].extend(proc_data)

    # Fallback: if no sub_documents, scan all pages
    else:
        for page in schema.get("pages", []):
            for element in page.get("elements", []):
                elem_type = element.get("type", "")
                content = element.get("content", "")

                if elem_type == "medication_table":
                    med_data = _parse_medication_element(content)
                    if med_data:
                        structured_data["medications"].extend(med_data)

                elif elem_type == "lab_results_table":
                    lab_data = _parse_lab_element(content)
                    if lab_data:
                        structured_data["labs"].extend(lab_data)

    return JSONResponse(structured_data)


def _parse_medication_element(content: str) -> list:
    """Parse medication element content into structured format."""
    if not content or len(content) < 10:
        return []

    try:
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        prompt = f"""Extract medication information from this clinical text.

TEXT:
{content[:2000]}

Extract ALL medications mentioned. For each medication, provide:
- drug_name: Name of the medication
- dosage: Dosage (e.g., "500mg", "10 units")
- frequency: How often taken (e.g., "twice daily", "as needed")
- start_date: Start date if mentioned (or null)
- end_date: End date if mentioned (or null)
- notes: Any additional relevant information

Return as JSON array:
[
  {{
    "drug_name": "string",
    "dosage": "string or null",
    "frequency": "string or null",
    "start_date": "string or null",
    "end_date": "string or null",
    "notes": "string or null"
  }}
]

If no medications found, return empty array [].
Only return the JSON array, nothing else."""

        model = GenerativeModel(model_name="gemini-2.5-flash")
        generation_config = GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json"
        )

        response = model.generate_content(prompt, generation_config=generation_config)
        result = json.loads(response.text)
        return result if isinstance(result, list) else []

    except Exception as e:
        logging.error(f"Failed to parse medication element: {e}")
        return []


def _parse_lab_element(content: str) -> list:
    """Parse lab element content into structured format."""
    if not content or len(content) < 10:
        return []

    try:
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        prompt = f"""Extract laboratory test information from this clinical text.

TEXT:
{content[:2000]}

Extract ALL lab tests mentioned. For each test, provide:
- test_name: Name of the lab test
- date_ordered: Date the test was ordered (or null)
- result: Test result if available (or null)
- reason: Reason for ordering if mentioned (or null)
- notes: Any additional relevant information

Return as JSON array:
[
  {{
    "test_name": "string",
    "date_ordered": "string or null",
    "result": "string or null",
    "reason": "string or null",
    "notes": "string or null"
  }}
]

If no lab tests found, return empty array [].
Only return the JSON array, nothing else."""

        model = GenerativeModel(model_name="gemini-2.5-flash")
        generation_config = GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json"
        )

        response = model.generate_content(prompt, generation_config=generation_config)
        result = json.loads(response.text)
        return result if isinstance(result, list) else []

    except Exception as e:
        logging.error(f"Failed to parse lab element: {e}")
        return []


def _parse_procedure_element(content: str) -> list:
    """Parse procedure element content into structured format."""
    if not content or len(content) < 10:
        return []

    try:
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        prompt = f"""Extract procedure information from this clinical text.

TEXT:
{content[:2000]}

Extract ALL procedures mentioned. For each procedure, provide:
- procedure_name: Name/type of the procedure
- date: Date the procedure was performed (or null)
- purpose: Purpose/indication for the procedure (or null)
- result: Result or outcome if mentioned (or null)
- notes: Any additional relevant information

Return as JSON array:
[
  {{
    "procedure_name": "string",
    "date": "string or null",
    "purpose": "string or null",
    "result": "string or null",
    "notes": "string or null"
  }}
]

If no procedures found, return empty array [].
Only return the JSON array, nothing else."""

        model = GenerativeModel(model_name="gemini-2.5-flash")
        generation_config = GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json"
        )

        response = model.generate_content(prompt, generation_config=generation_config)
        result = json.loads(response.text)
        return result if isinstance(result, list) else []

    except Exception as e:
        logging.error(f"Failed to parse procedure element: {e}")
        return []


@app.get("/api/documents/{document_id}/reconstruction")
def reconstruct_document(document_id: str, kind: str = "enhanced", refresh: bool = False):
    """
    Reconstruct the document into clean sectioned bullets using Gemini, with caching.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id")

    extractions = db.get_extractions(doc_uuid)
    by_kind = {}
    for e in extractions:
        if e["kind"] not in by_kind:
            by_kind[e["kind"]] = e

    extraction = by_kind.get(kind) or by_kind.get("enhanced") or by_kind.get("full")
    if not extraction:
        raise HTTPException(status_code=404, detail="No extraction available for document")

    cache_key = f"{extraction['storage_url']}::{RECON_PROMPT_VERSION}"
    cached = RECONSTRUCTION_CACHE.get(cache_key)
    if cached and not refresh:
        return JSONResponse(cached)

    schema = SCHEMA_CACHE.get(cache_key)
    if not schema:
        with tempfile.TemporaryDirectory() as td:
            json_path = Path(td) / "schema.json"
            gcs.download_to_path(extraction["storage_url"], json_path)
            with open(json_path, "r") as f:
                schema = json.load(f)
            SCHEMA_CACHE.set(cache_key, schema)

    chunks = _chunk_schema(schema)
    if not chunks:
        raise HTTPException(status_code=500, detail="No content to reconstruct")

    reconstruction = _reconstruct_with_llm(chunks)

    RECONSTRUCTION_CACHE.set(cache_key, reconstruction)
    return JSONResponse(reconstruction)


def _chunk_schema(schema: dict, max_chunk_chars: int = 1600) -> List[str]:
    """
    Flatten schema sub-documents/pages into text chunks that fit LLM context.
    """
    chunks: List[str] = []

    def add_chunk(label: str, text: str):
        safe_text = text.strip()
        if not safe_text:
            return
        if len(safe_text) <= max_chunk_chars:
            chunks.append(f"{label}\n{text}")
            return
        # Split long text on sentences
        sentences = re.split(r"(?<=[.!?])\s+", safe_text)
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 > max_chunk_chars:
                if current:
                    chunks.append(f"{label}\n{current.strip()}")
                current = s
            else:
                current += " " + s
        if current.strip():
            chunks.append(f"{label}\n{current.strip()}")

    # Prioritize clinical element types
    priority_types = [
        "medication_table",
        "lab_results_table",
        "problem_list",
        "clinical_paragraph",
        "list_items",
        "general_table",
        "procedure",
        "radiology_imaging",
    ]
    def sort_key(e_type: str) -> int:
        try:
            return priority_types.index(e_type)
        except ValueError:
            return len(priority_types) + 1

    def process_pages(pages, title_prefix: str = ""):
        for page in pages:
            page_num = page.get("page_number")
            elements = sorted(page.get("elements", []), key=lambda el: sort_key(el.get("type", "")))
            for element in elements:
                content = element.get("content") or ""
                etype = element.get("type") or ""
                label = f"{title_prefix}type={etype} page={page_num}"
                add_chunk(label, content)

    if "sub_documents" in schema:
        for subdoc in schema.get("sub_documents", []):
            title = subdoc.get("title") or subdoc.get("type") or "Untitled"
            process_pages(subdoc.get("pages", []), title_prefix=f"[{title}] ")
    else:
        process_pages(schema.get("pages", []), title_prefix="[Page] ")

    return chunks


def _fallback_bucket(chunks: List[str]) -> dict:
    """
    Lightweight keyword-based bucketing to avoid empty responses if the LLM returns nothing.
    """
    buckets = {
        "summary": [],
        "patient": [],
        "problems": [],
        "meds": [],
        "labs": [],
        "procedures": [],
        "imaging": [],
        "notes": [],
        "other": [],
    }
    for c in chunks:
        lower = c.lower()
        target = None
        if any(k in lower for k in ["medication", "dose", "mg", "units", "tablet", "capsule", "q."]):
            target = "meds"
        elif any(k in lower for k in ["lab", "result", "cbc", "chemistry", "glucose", "creatinine"]):
            target = "labs"
        elif any(k in lower for k in ["procedure", "surgery", "operative", "colectomy", "biopsy"]):
            target = "procedures"
        elif any(k in lower for k in ["radiology", "imaging", "ct", "mri", "x-ray", "ultrasound"]):
            target = "imaging"
        elif any(k in lower for k in ["diagnosis", "history", "problem", "hx"]):
            target = "problems"
        elif any(k in lower for k in ["note", "plan", "assessment"]):
            target = "notes"
        elif any(k in lower for k in ["name:", "dob", "acct", "mrn", "sex", "age"]):
            target = "patient"
        if target:
            buckets[target].append(c.strip())
        else:
            buckets["other"].append(c.strip())
    return buckets


def _reconstruct_with_llm(chunks: List[str]) -> dict:
    """
    Use Gemini to bucket content into EHR sections and emit clean bullets.
    Runs in batches to avoid oversized single prompts.
    """
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    model = GenerativeModel(model_name="gemini-2.5-flash")
    generation_config = GenerationConfig(
        temperature=0.4,
        max_output_tokens=64000,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "array", "items": {"type": "string"}},
                "patient": {"type": "array", "items": {"type": "string"}},
                "problems": {"type": "array", "items": {"type": "string"}},
                "meds": {"type": "array", "items": {"type": "string"}},
                "labs": {"type": "array", "items": {"type": "string"}},
                "procedures": {"type": "array", "items": {"type": "string"}},
                "imaging": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "array", "items": {"type": "string"}},
                "other": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "patient", "problems", "meds", "labs", "procedures", "imaging", "notes", "other"],
        },
    )

    def run_batch(batch_chunks: List[str]) -> dict:
        prompt = f"""You are reconstructing an EHR from extracted fragments.

Goal: classify and rewrite every clinically meaningful fragment into concise bullets under the correct section. Use semantic understanding, not just headings.

Sections:
- patient: demographics, identifiers
- problems: diagnoses, history, assessments
- meds: medications (any dosage/frequency/route/timing/details)
- labs: lab tests and results (any units/formats)
- procedures: procedures/surgeries/operations
- imaging: radiology/imaging findings
- notes: narrative clinical notes/plans
- other: important details that don't fit the above

Rules:
- Use flexible bullets; include whatever fields exist. Do not drop information because a field is missing.
- Prefer clarity over uniformity; combine related details when helpful.
- Keep/merge duplicates; pick the clearest wording.
- Include dates, numbers, and qualifiers when present.
- Exclude boilerplate and empty headers. Do not echo placeholders like "DOCUMENT SUMMARY".
- If a fragment clearly belongs to a section, ensure that section has at least one bullet.

Fragments (label + text):
{json.dumps(batch_chunks, indent=2)}

Return JSON ONLY in this schema (arrays of strings; empty array only if truly no evidence):
{{
  "summary": ["bullet", "..."],
  "patient": ["bullet", "..."],
  "problems": ["bullet", "..."],
  "meds": ["bullet", "..."],
  "labs": ["bullet", "..."],
  "procedures": ["bullet", "..."],
  "imaging": ["bullet", "..."],
  "notes": ["bullet", "..."],
  "other": ["bullet", "..."]
}}

Keep lists concise but complete; typically 3-12 bullets per section when data exists."""
        response = model.generate_content(prompt, generation_config=generation_config)
        return json.loads(response.text)

    # Split into batches to reduce prompt size; target ~25 chunks or ~12000 chars per batch
    batches: List[List[str]] = []
    current: List[str] = []
    char_count = 0
    for c in chunks:
        c_len = len(c)
        if current and (len(current) >= 25 or char_count + c_len > 12000):
            batches.append(current)
            current = []
            char_count = 0
        current.append(c)
        char_count += c_len
    if current:
        batches.append(current)

    combined: dict = {
        "summary": [],
        "patient": [],
        "problems": [],
        "meds": [],
        "labs": [],
        "procedures": [],
        "imaging": [],
        "notes": [],
        "other": [],
    }

    for batch in batches:
        partial = run_batch(batch)
        for key in combined.keys():
            combined[key].extend(partial.get(key, []))

    # Deduplicate while preserving order
    def dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in seq:
            if not isinstance(item, str):
                continue
            norm = item.strip()
            if not norm or norm.lower() in seen:
                continue
            seen.add(norm.lower())
            out.append(norm)
        return out

    for key in combined.keys():
        combined[key] = dedupe(combined[key])

    return combined
