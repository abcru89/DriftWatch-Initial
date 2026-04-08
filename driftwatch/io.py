from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional
from typing import Dict
from urllib.parse import urlparse
from azure.storage.blob import ContentSettings

import pandas as pd
import requests


@dataclass
class LoadResult:
    df: Optional[pd.DataFrame]
    error: Optional[str]
    source: str


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_URL_BYTES = 10 * 1024 * 1024     # 10 MB


def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


def load_csv_from_path(path: str) -> LoadResult:
    """Load a CSV from a local path (used for sample_data and testing)."""
    try:
        df = pd.read_csv(path)
        return LoadResult(df=df, error=None, source="path")
    except Exception as e:
        return LoadResult(df=None, error=f"Failed to read CSV from path: {e}", source="path")


def load_csv_from_url(url: str, timeout_s: int = 20) -> LoadResult:
    """
    Load a CSV from a URL (http/https) with basic safety constraints.

    Controls:
    - Only http/https allowed
    - Timeout enforced
    - Response size capped
    """
    if not url or not url.strip():
        return LoadResult(df=None, error="URL is empty.", source="url")

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return LoadResult(df=None, error="Only http and https URLs are supported.", source="url")

    try:
        headers = {"User-Agent": "DriftWatch/1.0"}
        resp = requests.get(url, timeout=timeout_s, headers=headers)
        resp.raise_for_status()

        content_length = resp.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > MAX_URL_BYTES:
                    return LoadResult(df=None, error=f"URL content too large (>{MAX_URL_BYTES} bytes).", source="url")
            except ValueError:
                pass

        b = resp.content
        if len(b) > MAX_URL_BYTES:
            return LoadResult(df=None, error=f"URL content too large (>{MAX_URL_BYTES} bytes).", source="url")

        return LoadResult(df=_read_csv_bytes(b), error=None, source="url")
    except Exception as e:
        return LoadResult(df=None, error=f"Failed to load URL CSV: {e}", source="url")


def load_csv_from_upload(uploaded_file) -> LoadResult:
    """
    Load a CSV from a Streamlit uploaded file object.

    Controls:
    - Upload size capped
    """
    if uploaded_file is None:
        return LoadResult(df=None, error="No file uploaded.", source="file")
    try:
        # Streamlit UploadedFile supports getvalue()
        b = uploaded_file.getvalue()
        if len(b) > MAX_UPLOAD_BYTES:
            return LoadResult(df=None, error=f"Uploaded file too large (>{MAX_UPLOAD_BYTES} bytes).", source="file")
        return LoadResult(df=_read_csv_bytes(b), error=None, source="file")
    except Exception as e:
        return LoadResult(df=None, error=f"Failed to read uploaded CSV: {e}", source="file")

def _get_blob_service_client():
    from azure.storage.blob import BlobServiceClient

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING is not set. Add it to .streamlit/secrets.toml and restart Streamlit."
        )

    return BlobServiceClient.from_connection_string(conn_str)


def _get_container_name() -> str:
    return os.getenv("AZURE_STORAGE_CONTAINER", "driftwatch-artifacts")


def upload_text_to_blob(
    text: str,
    blob_name: str,
    *,
    content_type: str = "text/plain; charset=utf-8",
    overwrite: bool = True,
    metadata: Optional[dict[str, str]] = None,
) -> str:
    from azure.storage.blob import ContentSettings

    service = _get_blob_service_client()
    container_name = _get_container_name()
    container = service.get_container_client(container_name)

    try:
        container.create_container()
    except Exception:
        pass

    blob = container.get_blob_client(blob_name)
    blob.upload_blob(
        text.encode("utf-8"),
        overwrite=overwrite,
        content_settings=ContentSettings(content_type=content_type),
        metadata=metadata,
    )
    return blob.url


def upload_bytes_to_blob(
    data: bytes,
    blob_name: str,
    *,
    content_type: str = "application/octet-stream",
    overwrite: bool = True,
    metadata: Optional[dict[str, str]] = None,
) -> str:
    from azure.storage.blob import ContentSettings

    service = _get_blob_service_client()
    container_name = _get_container_name()
    container = service.get_container_client(container_name)

    try:
        container.create_container()
    except Exception:
        pass

    blob = container.get_blob_client(blob_name)
    blob.upload_blob(
        data,
        overwrite=overwrite,
        content_settings=ContentSettings(content_type=content_type),
        metadata=metadata,
    )
    return blob.url