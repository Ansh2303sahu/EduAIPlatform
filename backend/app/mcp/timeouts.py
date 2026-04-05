"""
Phase 11 MCP timeout wrapper.

``run_with_timeout`` wraps any async callable with ``asyncio.wait_for`` and
converts ``asyncio.TimeoutError`` into a ``ToolTimeoutError`` so the executor
handles it uniformly.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

from app.mcp.errors import ToolTimeoutError

T = TypeVar("T")


async def run_with_timeout(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    timeout_seconds: float,
    tool_name: str,
) -> T:
    """
    Execute ``coro_fn()`` and raise ``ToolTimeoutError`` if it exceeds ``timeout_seconds``.

    Parameters
    ----------
    coro_fn
        Zero-argument async callable — ``lambda: handler(input, ctx)`` in practice.
    timeout_seconds
        Maximum allowed wall-clock seconds.
    tool_name
        Included in the error message for traceability.
    """
    try:
        return await asyncio.wait_for(coro_fn(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise ToolTimeoutError(
            f"Tool {tool_name!r} exceeded timeout of {timeout_seconds:.1f}s"
        )
