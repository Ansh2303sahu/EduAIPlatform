"""
ML context normalization for Phase 10.

These helpers accept raw Phase 6 outputs and convert them into a stable
``MLContextResult`` object plus a concise prompt-friendly text summary.
"""

from __future__ import annotations

from typing import Any, Mapping

from app.langchain.enums import DecisionSource
from app.langchain.models import MLContextResult


_PROFESSOR_BAND_RANKS = {
    "ineffective": 0,
    "poor": 0,
    "fail": 0,
    "adequate": 1,
    "pass": 1,
    "satisfactory": 1,
    "good": 2,
    "merit": 2,
    "effective": 2,
    "very good": 3,
    "distinction": 3,
    "excellent": 3,
}

_LEVEL_RANKS = {
    "low": 0,
    "shallow": 0,
    "basic": 1,
    "med": 1,
    "medium": 1,
    "mixed": 1,
    "high": 2,
    "developed": 2,
    "deep": 2,
    "consistent": 2,
}


def normalize_student_ml_context(ml: Mapping[str, Any] | None) -> MLContextResult:
    """Normalize student Phase 6 output into a stable ML context object."""
    raw = dict(ml or {})
    raw_bundle = _as_dict(raw.get("raw"))
    feedback_raw = _as_dict(raw_bundle.get("feedback"))
    confidence_raw = _as_dict(raw_bundle.get("confidence"))
    feedback_pred = _as_dict(feedback_raw.get("prediction"))
    confidence_pred = _as_dict(confidence_raw.get("prediction"))

    feedback_category = _as_str(
        raw.get("feedback_category")
        or feedback_pred.get("label")
        or raw.get("predicted_label"),
        default="unclassified",
    )

    confidence_bucket = _clamp_int(
        raw.get("confidence_0_to_4"),
        default=_bucket_from_unit_confidence(_as_float(confidence_pred.get("confidence"), default=0.5)),
        minimum=0,
        maximum=4,
    )
    confidence_score = round(
        _as_float(
            raw.get("confidence_score"),
            default=_as_float(confidence_pred.get("confidence"), default=confidence_bucket / 4.0),
        ),
        3,
    )
    quality_band = _as_str(raw.get("quality_band"), default=_band_from_bucket(confidence_bucket))
    confidence_label = _unit_confidence_label(confidence_score)

    modalities_used = _modalities_from_sources(feedback_raw, confidence_raw, raw)
    modality_summary = _modality_summary(modalities_used)
    model_metadata = {
        "models": [
            item
            for item in [
                _model_entry("feedback", feedback_raw, feedback_pred),
                _model_entry("confidence", confidence_raw, confidence_pred),
            ]
            if item
        ],
        "modalities_used": modalities_used,
    }

    disagreement_markers = _student_disagreement_markers(
        quality_band=quality_band,
        feedback_confidence=_as_float(feedback_pred.get("confidence"), default=0.0),
        confidence_score=confidence_score,
        feedback_category=feedback_category,
        raw=raw,
    )

    normalized = {
        "role": "student",
        "predicted_label": feedback_category,
        "predicted_class": feedback_category,
        "predicted_band": quality_band,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "confidence_0_to_4": confidence_bucket,
        "modality_evidence_summary": modality_summary,
        "model_metadata": model_metadata,
        "disagreement_markers": disagreement_markers,
    }

    lines = [
        "The Phase 6 ML stack assessed this submission:",
        f"- Predicted label/class: {feedback_category}",
        f"- Predicted band: {quality_band}",
        f"- Confidence: {confidence_label} ({confidence_score:.2f}; bucket {confidence_bucket}/4)",
    ]
    if modality_summary:
        lines.append(f"- Modality evidence: {modality_summary}")
    if model_metadata["models"]:
        lines.append(
            "- Supporting model metadata: "
            + "; ".join(
                f"{item['source']}={item['model'] or 'unknown'}@{item['version'] or 'unknown'}"
                for item in model_metadata["models"]
            )
        )
    if disagreement_markers:
        lines.append("- Disagreement markers: " + "; ".join(disagreement_markers))
    else:
        lines.append("- Disagreement markers: none")
    lines.append("Use these signals as calibration hints only; prefer submission evidence and approved retrieval.")

    return MLContextResult(
        raw=raw,
        normalized=normalized,
        context_text="\n".join(lines),
        decision_source=DecisionSource.HYBRID,
        confidence_score=confidence_score,
        predicted_label=feedback_category,
        predicted_class=feedback_category,
        predicted_band=quality_band,
        modality_evidence_summary=modalities_used,
        model_metadata=model_metadata,
        disagreement_markers=disagreement_markers,
    )


def normalize_professor_ml_context(ml: Mapping[str, Any] | None) -> MLContextResult:
    """Normalize professor Phase 6 output into a stable ML context object."""
    raw = dict(ml or {})
    raw_bundle = _as_dict(raw.get("raw"))
    predictions = _as_dict(raw_bundle.get("predictions"))

    rubric_pred = _as_dict(predictions.get("rubric_band"))
    depth_pred = _as_dict(predictions.get("argument_depth"))
    consistency_pred = _as_dict(predictions.get("moderation_consistency"))

    rubric_band = _as_str(
        raw.get("rubric_band") or rubric_pred.get("label"),
        default="adequate",
    )
    argument_depth = _as_str(
        raw.get("argument_depth") or _as_dict(raw.get("raw_labels")).get("argument_depth") or depth_pred.get("label"),
        default="med",
    )
    moderation_consistency = _as_str(
        raw.get("moderation_consistency") or _as_dict(raw.get("raw_labels")).get("moderation_consistency") or consistency_pred.get("label"),
        default="med",
    )

    confidence_candidates = [
        _as_float(rubric_pred.get("confidence"), default=-1.0),
        _as_float(depth_pred.get("confidence"), default=-1.0),
        _as_float(consistency_pred.get("confidence"), default=-1.0),
    ]
    confidence_values = [value for value in confidence_candidates if value >= 0.0]
    confidence_score = round(
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else {"high": 0.85, "med": 0.6, "low": 0.3}.get(moderation_consistency.lower(), 0.6),
        3,
    )
    confidence_label = _unit_confidence_label(confidence_score)

    modalities_used = _modalities_from_sources(raw_bundle, raw)
    modality_summary = _modality_summary(modalities_used)
    model_metadata = {
        "models": [
            item
            for item in [
                _model_entry("rubric_suite", raw_bundle, {}),
            ]
            if item
        ],
        "heads": {
            head: _compact_prediction_dict(_as_dict(predictions.get(head)))
            for head in ("rubric_band", "argument_depth", "moderation_consistency")
            if _as_dict(predictions.get(head))
        },
        "modalities_used": modalities_used,
    }

    disagreement_markers = _professor_disagreement_markers(
        rubric_band=rubric_band,
        argument_depth=argument_depth,
        moderation_consistency=moderation_consistency,
        raw=raw,
        predictions=predictions,
    )

    normalized = {
        "role": "professor",
        "predicted_label": rubric_band,
        "predicted_class": rubric_band,
        "predicted_band": rubric_band,
        "rubric_band": rubric_band,
        "argument_depth": argument_depth,
        "moderation_consistency": moderation_consistency,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "modality_evidence_summary": modality_summary,
        "model_metadata": model_metadata,
        "disagreement_markers": disagreement_markers,
    }

    lines = [
        "The Phase 6 ML rubric stack assessed this submission:",
        f"- Predicted rubric band: {rubric_band}",
        f"- Argument depth signal: {argument_depth}",
        f"- Moderation consistency signal: {moderation_consistency}",
        f"- Confidence: {confidence_label} ({confidence_score:.2f})",
    ]
    if modality_summary:
        lines.append(f"- Modality evidence: {modality_summary}")
    if model_metadata["models"]:
        lines.append(
            "- Supporting model metadata: "
            + "; ".join(
                f"{item['source']}={item['model'] or 'unknown'}@{item['version'] or 'unknown'}"
                for item in model_metadata["models"]
            )
        )
    if disagreement_markers:
        lines.append("- Disagreement markers: " + "; ".join(disagreement_markers))
    else:
        lines.append("- Disagreement markers: none")
    lines.append("Use these signals as moderation hints only; prefer submission evidence and approved retrieval.")

    return MLContextResult(
        raw=raw,
        normalized=normalized,
        context_text="\n".join(lines),
        decision_source=DecisionSource.HYBRID,
        confidence_score=confidence_score,
        predicted_label=rubric_band,
        predicted_class=rubric_band,
        predicted_band=rubric_band,
        modality_evidence_summary=modalities_used,
        model_metadata=model_metadata,
        disagreement_markers=disagreement_markers,
    )


def build_student_ml_context(ml: Mapping[str, Any] | None) -> str:
    """Backward-compatible helper returning only the student prompt text."""
    return normalize_student_ml_context(ml).context_text


def build_professor_ml_context(ml: Mapping[str, Any] | None) -> str:
    """Backward-compatible helper returning only the professor prompt text."""
    return normalize_professor_ml_context(ml).context_text


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str(value: Any, *, default: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            parsed = default
    return max(minimum, min(maximum, parsed))


def _bucket_from_unit_confidence(score: float) -> int:
    if score < 0.35:
        return 0
    if score < 0.55:
        return 1
    if score < 0.70:
        return 2
    if score < 0.85:
        return 3
    return 4


def _band_from_bucket(bucket: int) -> str:
    if bucket <= 1:
        return "low"
    if bucket == 2:
        return "med"
    return "high"


def _unit_confidence_label(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _modalities_from_sources(*sources: Mapping[str, Any]) -> list[str]:
    found: dict[str, bool] = {}
    for source in sources:
        modalities = _as_dict(_as_dict(source).get("modalities_used"))
        for name, enabled in modalities.items():
            normalized_name = _as_str(name).lower()
            if normalized_name and _as_bool(enabled):
                found[normalized_name] = True
    return sorted(found.keys())


def _modality_summary(modalities: list[str]) -> str:
    return ", ".join(modalities) if modalities else ""


def _model_entry(source: str, payload: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    data = _as_dict(payload)
    if not data:
        return {}
    entry = {
        "source": source,
        "model": _as_str(data.get("model")),
        "version": _as_str(data.get("version")),
        "confidence": _as_float(_as_dict(prediction).get("confidence"), default=-1.0),
    }
    if entry["confidence"] < 0.0:
        entry.pop("confidence")
    if not entry["model"] and not entry["version"] and len(entry) == 1:
        return {}
    return entry


def _student_disagreement_markers(
    *,
    quality_band: str,
    feedback_confidence: float,
    confidence_score: float,
    feedback_category: str,
    raw: Mapping[str, Any],
) -> list[str]:
    markers: list[str] = []

    if feedback_confidence > 0.0 and abs(feedback_confidence - confidence_score) >= 0.35:
        markers.append("feedback_vs_confidence_gap")

    if quality_band == "low" and confidence_score >= 0.75:
        markers.append("quality_band_vs_confidence_gap")
    elif quality_band == "high" and confidence_score <= 0.35:
        markers.append("quality_band_vs_confidence_gap")

    alternate_labels = {
        feedback_category,
        _as_str(_as_dict(raw).get("predicted_label")),
        _as_str(_as_dict(_as_dict(raw).get("raw")).get("label")),
    }
    alternate_labels.discard("")
    if len(alternate_labels) > 1:
        markers.append("multiple_student_labels_present")

    return markers


def _compact_prediction_dict(prediction: Mapping[str, Any]) -> dict[str, Any]:
    pred = _as_dict(prediction)
    return {
        key: value
        for key, value in {
            "label": pred.get("label"),
            "confidence": pred.get("confidence"),
            "uncertain": pred.get("uncertain"),
            "reason": pred.get("reason"),
            "temperature": pred.get("temperature"),
        }.items()
        if value not in (None, "")
    }


def _professor_disagreement_markers(
    *,
    rubric_band: str,
    argument_depth: str,
    moderation_consistency: str,
    raw: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> list[str]:
    markers: list[str] = []
    raw_bundle = _as_dict(_as_dict(raw).get("raw"))

    if _as_bool(raw_bundle.get("uncertain")):
        markers.append("phase6_uncertain")

    head_confidences: list[float] = []
    for head in ("rubric_band", "argument_depth", "moderation_consistency"):
        pred = _as_dict(predictions.get(head))
        if _as_bool(pred.get("uncertain")):
            markers.append(f"{head}_uncertain")
        conf = _as_float(pred.get("confidence"), default=-1.0)
        if conf >= 0.0:
            head_confidences.append(conf)

    if head_confidences and (max(head_confidences) - min(head_confidences)) >= 0.35:
        markers.append("head_confidence_dispersion")

    rubric_rank = _rank_professor_band(rubric_band)
    depth_rank = _rank_level(argument_depth)
    consistency_rank = _rank_level(moderation_consistency)

    if rubric_rank - depth_rank >= 2:
        markers.append("rubric_band_vs_argument_depth_gap")
    if rubric_rank - consistency_rank >= 2:
        markers.append("rubric_band_vs_consistency_gap")

    return markers


def _rank_professor_band(label: str) -> int:
    return _PROFESSOR_BAND_RANKS.get(_as_str(label).lower(), 1)


def _rank_level(label: str) -> int:
    return _LEVEL_RANKS.get(_as_str(label).lower(), 1)
