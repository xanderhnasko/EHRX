"""
FastAPI service for PDF upload, extraction, and querying.

This is a synchronous MVP; long-running extraction will block the request.
Deploy behind Cloud Run/uvicorn as needed.
"""

import os
import logging
import tempfile
import uuid
import json
from pathlib import Path
from typing import Optional
from threading import Lock
from collections import OrderedDict

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

        # Upload page images (public for frontend previews)
        page_images_dir = output_dir / "pages"
        page_image_map = {}
        if page_images_dir.exists():
            for img_path in sorted(page_images_dir.glob("page-*.png")):
                page_number = img_path.stem.split("-")[-1]
                dest = f"extractions/{document_id}/pages/{img_path.name}"
                gs_url = gcs.upload_file(img_path, dest, make_public=True)
                page_image_map[str(int(page_number))] = gs_url.replace("gs://", "https://storage.googleapis.com/")

    stats = document.get("processing_stats", {})
    metadata_common = {
        "index_url": index_url,
        "page_images": page_image_map,
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

    cache_key = extraction["storage_url"]
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
    for el in result.get("matched_elements", []):
        page = el.get("page")
        if page is None:
            continue
        img_url = page_images.get(str(page))
        if img_url:
            el["image_url"] = img_url

    return JSONResponse(result)
