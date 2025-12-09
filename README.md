# PDF2EHR (EHRX)

Extract structured clinical data from scanned EHR PDFs using a Vertex AI Gemini pipeline plus a lightweight FastAPI + React UI.

## Hosted App

* Live site: https://ehrx.netlify.app/ (uses our GCP credits, availability depends on those remaining)
* Workflow: upload a PDF (start small) → wait for extraction → ask questions against the document; provenance and structured views are returned.

## Run It Yourself (GCP-backed)

* Prereqs: Python 3.11, Node 18+, `poppler-utils` for PDF rasterization, a GCP project with Vertex AI + Cloud Storage enabled, and a Postgres instance reachable from where you run the API.
* Env file (`.env` at repo root) minimally needs:

```
GCP_PROJECT_ID=your-project
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/abs/path/to/service-account.json
GCS_BUCKET=your-bucket
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=ehrx
DB_USER=appuser
DB_PASSWORD=your-password
FRONTEND_ORIGINS=http://localhost:5173
```

### Backend API (FastAPI)

* Install deps: `pip install -e . && pip install -r requirements.txt`
* Run locally: `uvicorn ehrx.web.app:app --host 0.0.0.0 --port 8080`
* Endpoints: `POST /documents` (upload), `POST /documents/{id}/extract`, `POST /api/query`, `GET /api/healthz`

### CLI Pipelines (direct Gemini calls)

* Full extract on a PDF (incurs GCP cost): `python scripts/run_mvp_pipeline.py`
* Quick demo on existing extraction (no new model calls): `python scripts/run_sample_e2e.py --full-json output/test_20_pages/SENSITIVE_ehr1_copy_1763164390_full.json`
* Batch ontologies: `python scripts/batch_process_ontologies.py --in <pdf_dir> --out <ontologies_dir>`

### Frontend

* `cd frontend && npm install`
* Create `frontend/.env.local` with `VITE_API_URL=http://localhost:8080`
* Dev server: `npm run dev` (defaults to http://localhost:5173). Build for static hosting: `npm run build` (outputs to `frontend/dist/`).

### Tests

* `pytest tests`

## Notes

* Protect PHI: inputs/outputs under `output/` and sample PDFs are gitignored by default.
* Adjust `netlify.toml` redirects or `FRONTEND_ORIGINS` if you host your own API.