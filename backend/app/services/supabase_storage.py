from __future__ import annotations

import urllib.parse
from typing import Any, Dict

import httpx
from app.core.config import settings


def _require_storage_config() -> None:
    if not settings.supabase_url:
        raise RuntimeError("SUPABASE_URL missing")
    if not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY missing")


def _storage_headers() -> Dict[str, str]:
    _require_storage_config()
    return {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
    }


def _encode_path(path: str) -> str:
    # Keep slashes, encode everything else safely
    return urllib.parse.quote(path.lstrip("/"), safe="/")


def _normalize_signed_url(base: str, signed: str) -> str:
    """
    Supabase may return:
      - "/storage/v1/object/sign/..."
      - "/object/sign/..."   (missing "/storage/v1")
      - full "https://..." URL
    Normalize to a full URL that is GETtable.
    """
    signed = str(signed)

    # already absolute
    if signed.startswith("http://") or signed.startswith("https://"):
        return signed

    # ensure leading slash
    if not signed.startswith("/"):
        signed = "/" + signed

    # if it starts with "/object/..." prepend "/storage/v1"
    if signed.startswith("/object/"):
        signed = "/storage/v1" + signed

    # if it doesn't start with "/storage/", assume missing prefix
    if not signed.startswith("/storage/"):
        signed = "/storage/v1" + signed

    return base + signed


async def create_signed_download_url(*, bucket: str, path: str, expires_in: int) -> str:
    """
    POST {SUPABASE_URL}/storage/v1/object/sign/{bucket}/{path}
    body: {"expiresIn": <seconds>}
    response: {"signedURL": "..."}
    """
    _require_storage_config()

    base = settings.supabase_url.rstrip("/")
    enc_path = _encode_path(path)

    sign_url = f"{base}/storage/v1/object/sign/{bucket}/{enc_path}"
    payload: Dict[str, Any] = {"expiresIn": int(expires_in)}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(sign_url, headers=_storage_headers(), json=payload)

    if r.status_code >= 300:
        raise RuntimeError(f"Signed URL creation failed ({r.status_code}): {r.text}")

    data = r.json() or {}
    signed = data.get("signedURL") or data.get("signedUrl")
    if not signed:
        raise RuntimeError(f"Supabase did not return signedURL. Response: {data}")

    return _normalize_signed_url(base, signed)


async def upload_bytes_to_storage(*, bucket: str, path: str, data: bytes, content_type: str) -> None:
    """
    POST {SUPABASE_URL}/storage/v1/object/{bucket}/{path}
    """
    _require_storage_config()

    base = settings.supabase_url.rstrip("/")
    enc_path = _encode_path(path)

    url = f"{base}/storage/v1/object/{bucket}/{enc_path}"

    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": content_type or "application/octet-stream",
        "x-upsert": "true",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, content=data)

    if r.status_code >= 300:
        raise RuntimeError(f"Storage upload failed ({r.status_code}): {r.text}")
