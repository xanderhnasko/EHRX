"""
FastAPI service for PDF upload, extraction, and querying.

This is a synchronous MVP; long-running extraction will block the request.
Deploy behind Cloud Run/uvicorn as needed.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
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

app = FastAPI(title="PDF2EHR API")

db_config = DBConfig.from_env()
db = DB(db_config)

GCS_BUCKET = os.getenv("GCS_BUCKET")
if not GCS_BUCKET:
    raise RuntimeError("GCS_BUCKET env var is required")
gcs = GCSClient(GCS_BUCKET)

# Serve frontend static assets if present (will remain inert until frontend is added)
if Path("static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


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
        )

        grouper = SubDocumentGrouper(confidence_threshold=0.80)
        enhanced_doc = grouper.group_document(document)
        index = generate_hierarchical_index(enhanced_doc)

        full_path = output_dir / f"{document['document_id']}_full.json"
        enhanced_path = output_dir / f"{document['document_id']}_enhanced.json"
        index_path = output_dir / f"{document['document_id']}_index.json"

        full_url = gcs.upload_file(full_path, f"extractions/{document_id}/{full_path.name}")
        enhanced_url = gcs.upload_file(enhanced_path, f"extractions/{document_id}/{enhanced_path.name}")
        index_url = gcs.upload_file(index_path, f"extractions/{document_id}/{index_path.name}")

    stats = document.get("processing_stats", {})
    db.upsert_extraction(doc_uuid, "full", full_url, stats.get("total_pages"), stats.get("total_elements"), {"index_url": index_url})
    db.upsert_extraction(doc_uuid, "enhanced", enhanced_url, stats.get("total_pages"), stats.get("total_elements"), {"index_url": index_url})
    db.upsert_extraction(doc_uuid, "index", index_url, stats.get("total_pages"), stats.get("total_elements"), None)

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


@app.post("/query")
def query_document(document_id: str, kind: str = "enhanced", question: str = ""):
    """Run a query against an extraction (prefers enhanced)."""
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id")

    extractions = db.get_extractions(doc_uuid)
    by_kind = {e["kind"]: e for e in extractions}
    extraction = by_kind.get(kind) or by_kind.get("enhanced") or by_kind.get("full")
    if not extraction:
        raise HTTPException(status_code=404, detail="No extraction available for document")

    with tempfile.TemporaryDirectory() as td:
        json_path = Path(td) / "schema.json"
        gcs.download_to_path(extraction["storage_url"], json_path)

        agent = HybridQueryAgent(schema_path=str(json_path), vlm_config=VLMConfig.from_env())
        result = agent.query(question)

    return JSONResponse(result)
