"""Consistency-check helpers for Phase 15/16."""

from __future__ import annotations

from typing import Any

from app.genai.schemas import ConsistencyFinding


_POSITIVE_TOKENS = {"excellent", "strong", "clear", "coherent", "effective", "good"}
_NEGATIVE_TOKENS = {"weak", "unclear", "missing", "inconsistent", "limited", "poor"}


def detect_report_contradictions(report: dict[str, Any]) -> list[ConsistencyFinding]:
    """Detect lightweight contradictions inside a generated report."""

    findings: list[ConsistencyFinding] = []
    summary = str(report.get("summary") or "").lower()
    strengths = " ".join(str(item) for item in report.get("strengths") or []).lower()
    weaknesses = " ".join(str(item) for item in report.get("weaknesses") or []).lower()
    suggestions = " ".join(str(item) for item in report.get("suggestions") or []).lower()

    if any(token in summary for token in _POSITIVE_TOKENS) and any(token in weaknesses for token in _NEGATIVE_TOKENS):
        findings.append(
            ConsistencyFinding(
                issue="summary_vs_weakness_tension",
                severity="medium",
                detail="The summary sounds strongly positive while the weaknesses indicate notable limitations.",
            )
        )

    if not suggestions and weaknesses:
        findings.append(
            ConsistencyFinding(
                issue="missing_action_guidance",
                severity="medium",
                detail="Weaknesses were identified without enough follow-up suggestions.",
            )
        )

    confidence = report.get("confidence") or {}
    score_raw = confidence.get("score") if isinstance(confidence, dict) else None
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        score = 0.0

    if score >= 0.8 and len(report.get("weaknesses") or []) >= 4:
        findings.append(
            ConsistencyFinding(
                issue="score_feedback_mismatch",
                severity="high",
                detail="Confidence is high even though several weaknesses were identified.",
            )
        )

    if any(token in suggestions for token in _NEGATIVE_TOKENS) and any(token in strengths for token in _POSITIVE_TOKENS):
        findings.append(
            ConsistencyFinding(
                issue="tone_mismatch",
                severity="low",
                detail="The suggestions use harsher wording than the overall strengths summary.",
            )
        )
    return findings
