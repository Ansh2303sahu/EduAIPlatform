"""
Phase 11.2 MCP result cache.

Caches successful tool results in-process using a TTL dict.  Only tools whose
``ToolDefinition.supports_idempotency`` is ``True`` are eligible.  Failed
results are never cached.

Cache key composition
---------------------
``tool_name + version + role + sha256(json(sorted_payload))``

Normalising the payload via sorted JSON serialisation makes the key stable
across equivalent requests regardless of field ordering.  The role is included
so the same payload from a student and an admin produces separate cache entries.

TTL
---
Default 300 seconds (5 minutes), configurable via ``MCP_CACHE_TTL_SECONDS``.
Entries are evicted lazily on access.  A background sweep is not implemented to
keep the module dependency-free — the cache is sized for typical interactive
tool call volumes, not bulk batch processing.

Thread safety
-------------
CPython dict operations are GIL-protected for simple get/set.  The eviction
path modifies the dict under a lock to prevent corruption in concurrent workers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from threading import Lock
from typing import Any

logger = logging.getLogger("mcp.cache")

_lock = Lock()
# key → (stored_time, result_dict)
_STORE: dict[str, tuple[float, dict[str, Any]]] = {}

# Default TTL — overridden by config after import.
_DEFAULT_TTL: float = 300.0


def _make_key(
    tool_name: str,
    version: str,
    role: str,
    user_id: str,
    file_id: str | None,
    submission_id: str | None,
    payload: dict[str, Any],
) -> str:
    """Deterministic cache key from identity + normalised payload."""
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()[:16]
    file_scope = file_id or "-"
    submission_scope = submission_id or "-"
    return (
        f"{tool_name}|{version}|{role}|{user_id}|{file_scope}|"
        f"{submission_scope}|{payload_hash}"
    )


def get(
    tool_name: str,
    version: str,
    role: str,
    user_id: str,
    file_id: str | None,
    submission_id: str | None,
    payload: dict[str, Any],
    *,
    ttl_seconds: float = _DEFAULT_TTL,
) -> dict[str, Any] | None:
    """
    Return a cached result if available and not expired, else ``None``.

    Expired entries are evicted on the first access after their TTL expires.
    """
    key = _make_key(
        tool_name,
        version,
        role,
        user_id,
        file_id,
        submission_id,
        payload,
    )
    with _lock:
        entry = _STORE.get(key)
        if entry is None:
            return None
        stored_at, result = entry
        if time.monotonic() - stored_at > ttl_seconds:
            del _STORE[key]
            logger.debug("mcp.cache: evicted expired key tool=%s", tool_name)
            return None
        logger.debug("mcp.cache: hit tool=%s", tool_name)
        return result


def put(
    tool_name: str,
    version: str,
    role: str,
    user_id: str,
    file_id: str | None,
    submission_id: str | None,
    payload: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """
    Store a successful tool result.

    Only called for tools with ``supports_idempotency=True``; the executor
    is responsible for the guard — this function does not recheck it.
    """
    key = _make_key(
        tool_name,
        version,
        role,
        user_id,
        file_id,
        submission_id,
        payload,
    )
    with _lock:
        _STORE[key] = (time.monotonic(), result)
    logger.debug("mcp.cache: stored tool=%s", tool_name)


def invalidate(tool_name: str) -> int:
    """Evict all cached entries for ``tool_name``.  Returns count evicted."""
    prefix = f"{tool_name}|"
    with _lock:
        to_remove = [k for k in _STORE if k.startswith(prefix)]
        for k in to_remove:
            del _STORE[k]
    return len(to_remove)


def size() -> int:
    """Return the number of currently stored entries."""
    with _lock:
        return len(_STORE)
