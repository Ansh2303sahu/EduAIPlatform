from __future__ import annotations

import uuid
from typing import Any


UUID_INSERT_KEYS = {
    "actor_user_id",
    "correlation_id",
    "execution_id",
    "file_id",
    "job_id",
    "report_id",
    "request_id",
    "run_id",
    "submission_id",
    "trace_id",
    "user_id",
    "workflow_run_id",
}


def uuid_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return str(uuid.UUID(cleaned))
        except ValueError:
            return None
    return None


def normalize_uuid_insert_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return an insert payload with UUID-like fields normalized to UUID text or None."""

    normalized = dict(payload)
    for key in UUID_INSERT_KEYS:
        if key in normalized:
            normalized[key] = uuid_or_none(normalized.get(key))
    return normalized
