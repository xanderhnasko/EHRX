# Backend-only image for FastAPI service.
# When frontend is added, adjust this to build/copy frontend assets into /app/static.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps for PDF processing and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      poppler-utils \
      libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port for Cloud Run
ENV PORT=8080
CMD ["sh", "-c", "uvicorn ehrx.web.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
