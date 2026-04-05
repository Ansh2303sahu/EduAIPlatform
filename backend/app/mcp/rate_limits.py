"""
Phase 11 MCP in-process rate limiter.

Uses a sliding-window counter keyed by ``(tool_name, user_id)``.  This is an
intentionally simple in-process implementation — sufficient for a single-worker
deployment.  Replace with Redis-backed counting for multi-worker deployments.

Interface
---------
``check_rate_limit(tool_name, user_id)`` — raises ``RateLimitError`` if the
caller has exceeded the per-tool limit for the current window.  On success it
records the call timestamp so future calls are counted correctly.

The function is intentionally synchronous (no DB I/O) so it can run before any
awaits in the executor hot path.
"""

from __future__ import annotations

import time
from collections import defaultdict

from app.mcp.errors import RateLimitError

# Sliding-window history: (tool_name, user_id) → list of monotonic timestamps.
_CALL_HISTORY: dict[tuple[str, str], list[float]] = defaultdict(list)

# Defaults applied when the caller does not pass explicit limits.
_DEFAULT_MAX_CALLS: int = 20
_DEFAULT_WINDOW_SECONDS: float = 60.0


def check_rate_limit(
    tool_name: str,
    user_id: str,
    *,
    max_calls: int = _DEFAULT_MAX_CALLS,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
) -> None:
    """
    Enforce a sliding-window rate limit for ``(tool_name, user_id)``.

    Raises ``RateLimitError`` if the caller has already made ``max_calls``
    within the last ``window_seconds`` seconds.  Records the current call on
    success so subsequent calls are counted correctly.
    """
    key = (tool_name, user_id)
    now = time.monotonic()
    window_start = now - window_seconds

    history = _CALL_HISTORY[key]
    # Drop timestamps that have fallen outside the current window.
    history[:] = [t for t in history if t >= window_start]

    if len(history) >= max_calls:
        raise RateLimitError(
            f"Rate limit exceeded for tool {tool_name!r}: "
            f"max {max_calls} calls per {window_seconds:.0f}s",
            retryable=True,
        )

    # Record this call — only reached when not rate-limited.
    history.append(now)
