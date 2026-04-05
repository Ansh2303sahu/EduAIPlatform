"""
Phase 11 MCP central tool registry.

The registry is a module-level dict keyed by ``tool_name``.  Tools register
themselves by calling ``register_tool`` at import time — see ``mcp/tools/``.
The executor always resolves tools through ``resolve_tool`` so it never
references tool handlers directly.

Security properties
-------------------
- Unknown names → ``UnknownToolError`` (no info leakage about registered tools
  to unauthenticated callers — the API layer enforces auth before reaching here).
- Disabled tools → ``DisabledToolError`` regardless of who asks.
- Duplicate registration → logged as a warning; later registration wins.
  This makes hot-reloading predictable without silently masking bugs.
"""

from __future__ import annotations

import logging

from app.mcp.errors import DisabledToolError, UnknownToolError
from app.mcp.models import ToolDefinition

logger = logging.getLogger("mcp.registry")

# Module-level singleton registry.  Keys are canonical tool_name strings.
_REGISTRY: dict[str, ToolDefinition] = {}


def register_tool(defn: ToolDefinition) -> None:
    """Register a tool definition.  May be called multiple times; last write wins."""
    if defn.tool_name in _REGISTRY:
        logger.warning(
            "mcp.registry: duplicate registration for %r — overwriting", defn.tool_name
        )
    _REGISTRY[defn.tool_name] = defn
    logger.info(
        "mcp.registry: registered tool=%r namespace=%s version=%s enabled=%s risk=%s",
        defn.tool_name,
        defn.namespace.value,
        defn.version,
        defn.enabled,
        defn.risk_level.value,
    )


def resolve_tool(tool_name: str) -> ToolDefinition:
    """
    Resolve a tool by name.

    Raises
    ------
    UnknownToolError
        If no tool with that name is registered.
    DisabledToolError
        If the tool is registered but ``enabled=False``.
    """
    defn = _REGISTRY.get(tool_name)
    if defn is None:
        raise UnknownToolError(f"Tool not found: {tool_name!r}")
    if not defn.enabled:
        raise DisabledToolError(f"Tool is disabled: {tool_name!r}")
    return defn


def list_tools(*, include_disabled: bool = False) -> list[ToolDefinition]:
    """Return all registered tool definitions, optionally including disabled ones."""
    return [
        d for d in _REGISTRY.values()
        if include_disabled or d.enabled
    ]
