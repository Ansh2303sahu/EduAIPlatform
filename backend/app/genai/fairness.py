"""Fairness and trust helpers for Phase 15/16."""

from __future__ import annotations

from app.genai.explainability import confidence_band
from app.genai.schemas import BiasSignal, ConsistencyFinding, FairnessTab, MultiModelAgreement


_HARSH_TERMS = {"lazy", "careless", "bad", "terrible", "hopeless", "awful"}
_ABSOLUTE_TERMS = {"always", "never", "obviously", "everyone knows", "clearly"}


def bias_signals_for_report(report: dict[str, object]) -> list[BiasSignal]:
    """Detect lightweight tone and consistency bias indicators."""

    text = " ".join(
        [
            str(report.get("summary") or ""),
            " ".join(str(x) for x in report.get("suggestions") or []),
            " ".join(str(x) for x in report.get("weaknesses") or []),
        ]
    ).lower()
    signals: list[BiasSignal] = []
    if any(token in text for token in _HARSH_TERMS):
        signals.append(
            BiasSignal(
                name="harsh_tone",
                severity="high",
                detail="The report uses language that may feel judgmental rather than constructive.",
            )
        )
    if any(token in text for token in _ABSOLUTE_TERMS):
        signals.append(
            BiasSignal(
                name="absolute_language",
                severity="medium",
                detail="The report includes absolute phrasing that may overstate certainty.",
            )
        )
    if not signals:
        signals.append(
            BiasSignal(
                name="constructive_tone_check",
                severity="low",
                detail="No strong tone-bias indicators were detected in the current wording.",
            )
        )
    return signals


def build_agreement(
    *,
    primary_model: str,
    validator_model: str,
    primary_confidence: float,
    validator_confidence: float,
    contradiction_count: int,
) -> MultiModelAgreement:
    """Build a compact multi-model agreement summary."""

    distance = abs(primary_confidence - validator_confidence)
    score = max(0.0, min(1.0, 1.0 - distance - (0.1 * contradiction_count)))
    band = confidence_band(score)
    return MultiModelAgreement(
        primary_model=primary_model,
        validator_model=validator_model,
        agreement_score=score,
        agreement_band=band,
        summary=(
            f"Agreement is {band} ({score:.2f}) based on confidence alignment between the primary "
            f"and validator models and the number of contradiction flags ({contradiction_count})."
        ),
    )


def build_fairness_tab(
    *,
    report: dict[str, object],
    counterfactual: str,
    consistency_findings: list[ConsistencyFinding],
    primary_model: str,
    validator_model: str,
    primary_confidence: float,
    validator_confidence: float,
) -> FairnessTab:
    """Build the fairness tab from bias, consistency, and agreement checks."""

    return FairnessTab(
        bias_signals=bias_signals_for_report(report),
        multi_model_agreement=build_agreement(
            primary_model=primary_model,
            validator_model=validator_model,
            primary_confidence=primary_confidence,
            validator_confidence=validator_confidence,
            contradiction_count=len(consistency_findings),
        ),
        counterfactual_explanation=counterfactual,
        consistency_findings=consistency_findings,
    )
