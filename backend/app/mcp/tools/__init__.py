"""
MCP tool package.

Importing this package triggers registration of all built-in tools via their
module-level ``register_tool(...)`` calls.  The order of imports is the
registration order — it does not affect runtime behaviour (the registry is
keyed by tool name, not insertion order) but is kept alphabetical for clarity.
"""

from __future__ import annotations

from app.mcp.tools import (  # noqa: F401
    admin_docker_service_health,
    admin_github_repo_status,
    professor_consistency_checker,
    professor_rubric_evaluator,
    student_citation_helper,
    student_structure_improver,
    student_summariser,
)
