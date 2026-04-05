"""
Phase 11 MCP Tool Layer.

Importing this package automatically registers all built-in tools.  The
``tools`` sub-package is imported here purely for its side-effects
(``register_tool`` calls).  Application code should import from the specific
sub-modules rather than from this ``__init__``.
"""

from __future__ import annotations

# Trigger registration of all built-in tools.
import app.mcp.tools as _tools_bootstrap  # noqa: F401
