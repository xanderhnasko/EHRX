# Backend-only image for FastAPI service.
# When frontend is added, adjust this to build/copy frontend assets into /app/static.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port for Cloud Run
ENV PORT=8080
CMD ["uvicorn", "ehrx.web.app:app", "--host", "0.0.0.0", "--port", "8080"]
