"""
Google Cloud Storage helpers.
"""

import io
import os
from pathlib import Path
from typing import Optional

from google.cloud import storage
from datetime import timedelta
from google.api_core import exceptions as gcs_exceptions
from google.auth import iam, impersonated_credentials, credentials as google_auth_credentials
from google.auth.transport.requests import Request


class GCSClient:
    """Thin wrapper around google-cloud-storage for simple upload/download."""

    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        # Capture credentials and signer email up front; Cloud Run credentials lack a private key,
        # so we fall back to IAM SignBlob using the service account identity.
        self._credentials = self.client._credentials  # default creds from ADC
        self._signer_email = os.getenv("SIGNING_SERVICE_ACCOUNT_EMAIL") or getattr(
            self._credentials, "service_account_email", None
        )
        self._request = Request()

    def upload_file(self, local_path: Path, dest_blob: str, make_public: bool = False, return_public_url: bool = False) -> str:
        blob = self.bucket.blob(dest_blob)
        blob.upload_from_filename(local_path)
        if make_public:
            try:
                blob.make_public()
            except gcs_exceptions.GoogleAPICallError:
                # If we cannot make public (e.g., policy), just return gs:// URL
                pass
        if make_public and return_public_url:
            return blob.public_url
        return f"gs://{self.bucket.name}/{dest_blob}"

    def generate_signed_url(self, dest_blob: str, expiration_seconds: int = 7 * 24 * 3600, content_type: str | None = None) -> str:
        """
        Generate a V4 signed URL. Works on Cloud Run without a private key by using IAM
        SignBlob. Optionally override signer email via SIGNING_SERVICE_ACCOUNT_EMAIL.
        """
        blob = self.bucket.blob(dest_blob)
        # Ensure access token is fresh (required for IAM signing).
        if hasattr(self._credentials, "refresh"):
            self._credentials.refresh(self._request)

        signer_email = self._signer_email
        if not signer_email:
            raise RuntimeError("No service account email available for signing URLs")

        # Choose signing strategy:
        # - Use native signing credentials if available.
        # - Otherwise, impersonate the signer email to get Credentials that implement sign_bytes.
        if isinstance(self._credentials, google_auth_credentials.Signing):
            signing_credentials = self._credentials
        else:
            signing_credentials = impersonated_credentials.Credentials(
                source_credentials=self._credentials,
                target_principal=signer_email,
                target_scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
                lifetime=300,
            )
            signing_credentials.refresh(self._request)

        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expiration_seconds),
            method="GET",
            response_disposition="inline",
            response_type=content_type,
            service_account_email=signer_email,
            credentials=signing_credentials,
        )

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
