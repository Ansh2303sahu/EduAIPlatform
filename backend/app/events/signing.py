"""HMAC-SHA256 request signing for outbound n8n webhook calls.

Protocol
--------
1. Serialize the event body to canonical JSON (sorted keys, no whitespace).
2. Compute HMAC-SHA256 over the raw bytes using the shared secret.
3. Set headers:
     X-EduAI-Signature-256  : sha256=<hex-digest>
     X-EduAI-Timestamp      : <unix-epoch-seconds as string>
     X-EduAI-Idempotency-Key: <event_id>

The n8n Function node replicates steps 1–3 on its side and does a
constant-time compare to verify the signature.

Clock-skew guard
----------------
The timestamp is embedded in the signed payload (via ``OutboundEvent.timestamp``),
but a separate header carries the epoch seconds for easy threshold checking
without parsing ISO-8601 in JavaScript.

Security properties
-------------------
- Signature covers the full serialized body, not a subset of fields.
- Replay protection: n8n checks that |now - timestamp| < 5 minutes.
- Idempotency: n8n stores processed event_ids; duplicates are dropped.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any


def _canonical_json(data: dict[str, Any]) -> bytes:
    """Produce a deterministic UTF-8 JSON encoding with sorted keys."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_payload(body: dict[str, Any], *, secret: str) -> tuple[bytes, dict[str, str]]:
    """Sign *body* and return (raw_bytes, headers_to_attach).

    Args:
        body   : The dict that will be JSON-serialized and sent as the request body.
        secret : The HMAC-SHA256 secret (hex string or arbitrary str).

    Returns:
        A 2-tuple of:
          - raw_bytes : canonical JSON bytes (use this as the request body verbatim)
          - headers   : dict with Signature, Timestamp, and Idempotency headers
    """
    raw = _canonical_json(body)
    epoch_str = str(int(time.time()))

    mac = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256)
    signature = f"sha256={mac.hexdigest()}"

    headers = {
        "X-EduAI-Signature-256": signature,
        "X-EduAI-Timestamp": epoch_str,
        "X-EduAI-Idempotency-Key": body.get("event_id", ""),
        "Content-Type": "application/json",
    }
    return raw, headers


def verify_signature(
    raw_body: bytes,
    signature_header: str,
    *,
    secret: str,
    timestamp_header: str | None = None,
    tolerance_seconds: int = 300,
) -> bool:
    """Verify an inbound HMAC signature (used in tests and any future webhook receiver).

    Args:
        raw_body          : Raw request body bytes (before any decoding).
        signature_header  : Value of X-EduAI-Signature-256.
        secret            : Shared HMAC secret.
        timestamp_header  : Value of X-EduAI-Timestamp (epoch seconds as string).
        tolerance_seconds : Max allowed clock skew in seconds.

    Returns:
        True if signature is valid and timestamp is within tolerance.
    """
    if not signature_header.startswith("sha256="):
        return False

    expected_mac = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256)
    expected = f"sha256={expected_mac.hexdigest()}"

    if not hmac.compare_digest(expected.encode(), signature_header.encode()):
        return False

    if timestamp_header is not None:
        try:
            ts = int(timestamp_header)
        except (ValueError, TypeError):
            return False
        if abs(int(time.time()) - ts) > tolerance_seconds:
            return False

    return True
