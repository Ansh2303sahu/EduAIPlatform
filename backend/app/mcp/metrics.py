"""
Phase 11.2 MCP in-process metrics.

Simple thread-safe counters using Python's ``collections.Counter``.  These are
intentionally lightweight — no external dependencies, no network calls.  They
are suitable for single-worker deployments and provide observability without
adding operational complexity.

Counters tracked
----------------
- ``success``           — total successful tool executions
- ``failure``           — total failed executions (all error codes)
- ``timeout``           — subset of failures caused by ToolTimeoutError
- ``policy_denied``     — subset of failures caused by PolicyDeniedError
- ``rate_limited``      — subset of failures caused by RateLimitError
- ``fallback_used``     — executions where the LLM fallback path was triggered
- ``llm_used``          — executions where the LLM was called (success or not)
- ``llm_failed``        — LLM call attempts that returned ok=False

Latency
-------
``record_latency`` maintains a running sum and count per tool so that average
latency can be computed by any caller without storing all samples.

Thread safety
-------------
CPython's GIL makes Counter increments effectively atomic for our use case.
For multi-process or multi-threaded WSGI/ASGI setups, replace with
``multiprocessing.Value`` or a Prometheus Counter.

Usage
-----
Import and call from the executor::

    from app.mcp.metrics import record_success, record_failure, record_latency
    record_success("student.summariser.v1")
    record_failure("student.summariser.v1", error_code="mcp.timeout", timeout=True)
    record_latency("student.summariser.v1", execution_ms=42.1)

Read for observability::

    from app.mcp.metrics import snapshot
    stats = snapshot()  # returns dict safe to serialise as JSON
"""

from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock
from typing import TypedDict


class ToolMetricsSnapshot(TypedDict):
    success: int
    failure: int
    timeout: int
    policy_denied: int
    rate_limited: int
    llm_used: int
    llm_failed: int
    fallback_used: int
    cache_hit: int
    avg_latency_ms: float


class MetricsTotalsSnapshot(TypedDict):
    success: int
    failure: int
    timeout: int
    policy_denied: int
    rate_limited: int
    llm_used: int
    fallback_used: int
    cache_hit: int


class WorkflowMetricsSnapshot(TypedDict):
    runs: int
    failures: int
    partial_completions: int
    step_failures: int
    executed_orders: dict[str, int]


class WorkflowMetricsTotalsSnapshot(TypedDict):
    runs: int
    failures: int
    partial_completions: int
    step_failures: int


class MetricsSnapshot(TypedDict):
    tools: dict[str, ToolMetricsSnapshot]
    totals: MetricsTotalsSnapshot
    workflows: dict[str, WorkflowMetricsSnapshot]
    workflow_totals: WorkflowMetricsTotalsSnapshot

_lock = Lock()

# --- counters (per-tool) ---
_success: Counter[str] = Counter()
_failure: Counter[str] = Counter()
_timeout: Counter[str] = Counter()
_policy_denied: Counter[str] = Counter()
_rate_limited: Counter[str] = Counter()
_fallback_used: Counter[str] = Counter()
_cache_hit: Counter[str] = Counter()
_llm_used: Counter[str] = Counter()
_llm_failed: Counter[str] = Counter()

# --- workflow counters (per-workflow) ---
_workflow_runs: Counter[str] = Counter()
_workflow_failures: Counter[str] = Counter()
_workflow_partials: Counter[str] = Counter()
_workflow_step_failures: Counter[str] = Counter()
_workflow_orders: dict[str, Counter[str]] = defaultdict(Counter)

# --- latency accumulators (per-tool) ---
_latency_sum: dict[str, float] = defaultdict(float)
_latency_count: dict[str, int] = defaultdict(int)


def record_success(
    tool_name: str,
    *,
    llm_used: bool = False,
    fallback_used: bool = False,
    cache_hit: bool = False,
) -> None:
    with _lock:
        _success[tool_name] += 1
        if llm_used:
            _llm_used[tool_name] += 1
        if fallback_used:
            _fallback_used[tool_name] += 1
        if cache_hit:
            _cache_hit[tool_name] += 1


def record_failure(
    tool_name: str,
    *,
    error_code: str = "",
    timeout: bool = False,
    policy_denied: bool = False,
    rate_limited: bool = False,
    llm_used: bool = False,
    llm_failed: bool = False,
) -> None:
    with _lock:
        _failure[tool_name] += 1
        if timeout:
            _timeout[tool_name] += 1
        if policy_denied:
            _policy_denied[tool_name] += 1
        if rate_limited:
            _rate_limited[tool_name] += 1
        if llm_used:
            _llm_used[tool_name] += 1
        if llm_failed:
            _llm_failed[tool_name] += 1


def record_latency(tool_name: str, execution_ms: float) -> None:
    with _lock:
        _latency_sum[tool_name] += execution_ms
        _latency_count[tool_name] += 1


def record_workflow_started(workflow_name: str) -> None:
    with _lock:
        _workflow_runs[workflow_name] += 1


def record_workflow_finished(
    workflow_name: str,
    *,
    final_status: str,
    executed_tool_order: list[str],
    step_failures: int = 0,
) -> None:
    order_key = " > ".join(executed_tool_order) if executed_tool_order else "(none)"
    with _lock:
        if final_status in {"failed", "blocked"}:
            _workflow_failures[workflow_name] += 1
        if final_status == "partial":
            _workflow_partials[workflow_name] += 1
        if step_failures:
            _workflow_step_failures[workflow_name] += step_failures
        _workflow_orders[workflow_name][order_key] += 1


def snapshot() -> MetricsSnapshot:
    """Return a JSON-serialisable snapshot of all counters."""
    with _lock:
        tool_names = sorted(
            set(_success)
            | set(_failure)
            | set(_timeout)
            | set(_policy_denied)
            | set(_rate_limited)
        )
        per_tool: dict[str, ToolMetricsSnapshot] = {}
        for name in tool_names:
            count = _latency_count.get(name, 0)
            avg_ms = (
                round(_latency_sum[name] / count, 2) if count else 0.0
            )
            per_tool[name] = {
                "success": _success.get(name, 0),
                "failure": _failure.get(name, 0),
                "timeout": _timeout.get(name, 0),
                "policy_denied": _policy_denied.get(name, 0),
                "rate_limited": _rate_limited.get(name, 0),
                "llm_used": _llm_used.get(name, 0),
                "llm_failed": _llm_failed.get(name, 0),
                "fallback_used": _fallback_used.get(name, 0),
                "cache_hit": _cache_hit.get(name, 0),
                "avg_latency_ms": avg_ms,
            }
        workflow_names = sorted(
            set(_workflow_runs)
            | set(_workflow_failures)
            | set(_workflow_partials)
            | set(_workflow_step_failures)
            | set(_workflow_orders)
        )
        workflow_metrics: dict[str, WorkflowMetricsSnapshot] = {}
        for name in workflow_names:
            workflow_metrics[name] = {
                "runs": _workflow_runs.get(name, 0),
                "failures": _workflow_failures.get(name, 0),
                "partial_completions": _workflow_partials.get(name, 0),
                "step_failures": _workflow_step_failures.get(name, 0),
                "executed_orders": dict(_workflow_orders.get(name, Counter())),
            }
        return {
            "tools": per_tool,
            "totals": {
                "success": sum(_success.values()),
                "failure": sum(_failure.values()),
                "timeout": sum(_timeout.values()),
                "policy_denied": sum(_policy_denied.values()),
                "rate_limited": sum(_rate_limited.values()),
                "llm_used": sum(_llm_used.values()),
                "fallback_used": sum(_fallback_used.values()),
                "cache_hit": sum(_cache_hit.values()),
            },
            "workflows": workflow_metrics,
            "workflow_totals": {
                "runs": sum(_workflow_runs.values()),
                "failures": sum(_workflow_failures.values()),
                "partial_completions": sum(_workflow_partials.values()),
                "step_failures": sum(_workflow_step_failures.values()),
            },
        }


def reset() -> None:
    """Clear all in-process counters. Intended for unit tests."""
    with _lock:
        _success.clear()
        _failure.clear()
        _timeout.clear()
        _policy_denied.clear()
        _rate_limited.clear()
        _fallback_used.clear()
        _cache_hit.clear()
        _llm_used.clear()
        _llm_failed.clear()
        _workflow_runs.clear()
        _workflow_failures.clear()
        _workflow_partials.clear()
        _workflow_step_failures.clear()
        _workflow_orders.clear()
        _latency_sum.clear()
        _latency_count.clear()
