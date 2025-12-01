# PDF2EHR Frontend

React + Vite + Tailwind single-page UI for the PDF2EHR backend.

## Setup
```bash
cd frontend
npm install
```

## Development
```bash
npm run dev
```
Visit the URL printed by Vite (default `http://localhost:5173`).

## Build
```bash
npm run build
```
Static assets are emitted to `dist/` (suitable for Netlify/Vercel).

## Config
- `VITE_API_URL` (env): base URL for the backend (e.g., your Cloud Run `status.url`). If unset, defaults to `https://pdf2ehr-api3-3bf3r3croq-uw.a.run.app`.
- UI calls the real API; if served from a different origin, add CORS on the backend or proxy through your host.

## Backend endpoints (live)
- `POST /documents` (multipart, field `file` = PDF) → `{ document_id }`
- `POST /documents/{document_id}/extract` → runs extraction synchronously, returns extraction URLs
- `POST /api/query` with `document_id`, `question`, optional `kind`
- `GET /api/healthz` (or `/`)
