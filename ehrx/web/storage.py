"""
Google Cloud Storage helpers.
"""

import io
from pathlib import Path
from typing import Optional

from google.cloud import storage
from datetime import timedelta
from google.api_core import exceptions as gcs_exceptions


class GCSClient:
    """Thin wrapper around google-cloud-storage for simple upload/download."""

    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, local_path: Path, dest_blob: str, make_public: bool = False) -> str:
        blob = self.bucket.blob(dest_blob)
        blob.upload_from_filename(local_path)
        if make_public:
            try:
                blob.make_public()
            except gcs_exceptions.GoogleAPICallError:
                # If we cannot make public (e.g., policy), just return gs:// URL
                pass
        return f"gs://{self.bucket.name}/{dest_blob}"

    def generate_signed_url(self, dest_blob: str, expiration_seconds: int = 7 * 24 * 3600) -> str:
        blob = self.bucket.blob(dest_blob)
        return blob.generate_signed_url(expiration=timedelta(seconds=expiration_seconds))

    def upload_bytes(self, data: bytes, dest_blob: str, content_type: Optional[str] = None) -> str:
        blob = self.bucket.blob(dest_blob)
        blob.upload_from_file(io.BytesIO(data), size=len(data), content_type=content_type)
        return f"gs://{self.bucket.name}/{dest_blob}"

    def download_to_path(self, blob_path: str, dest_path: Path) -> None:
        """
        Download gs://bucket/key to dest_path.
        """
        blob = self._resolve_blob(blob_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(dest_path)

    def _resolve_blob(self, blob_path: str):
        # blob_path can be full gs://... or key
        if blob_path.startswith("gs://"):
            parts = blob_path.replace("gs://", "").split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid GCS path: {blob_path}")
            bucket_name, key = parts
            if bucket_name != self.bucket.name:
                raise ValueError(f"Bucket mismatch: expected {self.bucket.name}, got {bucket_name}")
            return self.bucket.blob(key)
        return self.bucket.blob(blob_path)
