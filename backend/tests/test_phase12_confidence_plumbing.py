from __future__ import annotations

import json

import pytest

from app.langgraph.nodes import final_guardrail, validation
from app.langgraph.schemas import Phase12ExecutionRequest
from app.langgraph.state import Phase12GraphState
from app.langgraph.tracing.model_versions import build_phase12_model_versions
from app.langchain.models import RagContext


def _state(role: str) -> Phase12GraphState:
    return Phase12GraphState.create(
        Phase12ExecutionRequest(
            file_id=f"{role}-file",
            user_id=f"{role}-user",
            role=role,
            correlation_id=f"{role}-corr",
        )
    )


@pytest.mark.asyncio
async def test_student_validation_stores_dict_and_carries_ml_confidence() -> None:
    state = _state("student")
    state.pipeline_context.ml_raw = {
        "feedback_category": "structure",
        "quality_band": "high",
        "confidence_0_to_4": 3,
    }
    state.pipeline_context.raw_llm_output = json.dumps(
        {
            "summary": "The submission is coherent and gives enough detail for targeted feedback.",
            "issues": [
                {
                    "title": "Testing depth",
                    "evidence": "The testing discussion is present but does not cover edge cases clearly.",
                    "severity": "med",
                }
            ],
            "strengths": [
                {
                    "title": "Clear structure",
                    "evidence": "The report follows a readable sequence from context to evaluation.",
                }
            ],
            "confidence": {"mode": "normal", "overall": 0.68},
            "safety": {"needs_review": False, "reason": ""},
        }
    )

    state = await validation.run(state)

    assert state.pipeline_context.validation_result.valid is True
    assert isinstance(state.pipeline_context.report, dict)
    agreement = state.pipeline_context.report["model_agreement"]
    assert agreement["ml_confidence"] == pytest.approx(0.75)
    assert agreement["llm_confidence"] > 0
    assert agreement["final_confidence"] > 0

    versions = build_phase12_model_versions(state)
    assert versions["agreement"]["ml_confidence"] == pytest.approx(0.75)
    assert versions["agreement"]["ml_bucket_0_to_4"] == 3
    assert versions["ml"]["confidence_score"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_professor_validation_sets_final_confidence_without_report_confidence_field() -> None:
    state = _state("professor")
    state.evidence_quality_score = 0.5
    state.pipeline_context.ml_raw = {
        "rubric_band": "good",
        "argument_depth": "high",
        "moderation_consistency": "high",
        "raw": {
            "predictions": {
                "rubric_band": {"label": "good", "confidence": 0.7},
                "argument_depth": {"label": "high", "confidence": 0.6},
                "moderation_consistency": {"label": "high", "confidence": 0.8},
            }
        },
    }
    state.pipeline_context.raw_llm_output = json.dumps(
        {
            "rubric_breakdown": [
                {
                    "criterion": "Evidence use",
                    "band": "Good",
                    "justification": "The submission provides enough evidence to support most claims.",
                },
                {
                    "criterion": "Analysis",
                    "band": "Good",
                    "justification": "The explanation links evidence to judgement with only minor gaps.",
                },
            ],
            "feedback_explanation": "The overall judgement is grounded in the submitted evidence. A human marker should still verify the borderline details.",
            "moderation_notes": [
                {
                    "risk": "Borderline evidence",
                    "note": "Check whether the weaker sections affect the final band.",
                }
            ],
            "safety": {"needs_review": False, "reason": ""},
        }
    )

    state = await validation.run(state)
    state = await final_guardrail.final_guardrail_node(state)

    assert state.pipeline_context.validation_result.valid is True
    assert isinstance(state.pipeline_context.report, dict)
    assert not any("missing expected top-level keys" in warning for warning in state.warnings)
    assert state.pipeline_context.execution_meta.agreement_score > 0

    versions = build_phase12_model_versions(state)
    assert versions["agreement"]["final_confidence"] > 0
    assert versions["agreement"]["ml_confidence"] > 0
    assert versions["ml"]["confidence_score"] > 0


@pytest.mark.asyncio
async def test_validation_node_caps_student_confidence_by_grounding_quality() -> None:
    state = _state("student")
    state.pipeline_context.ingestion.text_content = (
        "Introduction. This chapter explores formative feedback in higher education and argues that faster automated "
        "feedback must also preserve pedagogical depth."
    )
    state.pipeline_context.ml_raw = {
        "feedback_category": "structure",
        "quality_band": "medium",
        "confidence_0_to_4": 2,
    }
    state.pipeline_context.rag = RagContext(
        enabled=False,
        context="",
        context_text="",
        citations=[],
        retrieved_chunks=[],
        chunk_count=0,
        weak_retrieval=True,
        confidence_score=0.25,
        confidence_label="low",
        safe_review=True,
        trace={"keywords_used": ["formative feedback", "higher education", "pedagogical depth"]},
    )
    state.draft_report = {
        "summary": "The submission shows some promising work, but it needs clearer support and more evidence needed.",
        "overall_judgment": "Promising work overall, but clearer support is still needed.",
        "strengths": ["Promising work"],
        "weaknesses": ["More evidence needed."],
        "suggestions": ["Provide clearer support for the main claim."],
        "confidence_score": 0.92,
        "confidence": {"score": 0.92, "band": "high", "rationale": "The draft sounds confident."},
        "safety": {"needs_review": False, "reason": ""},
    }

    state = await validation.validation_node(state)

    report = state.pipeline_context.report
    assert report["confidence_score"] <= 0.28
    assert report["confidence"]["score"] <= 0.28
    assert report["confidence"]["band"] == "low"
    assert state.pipeline_context.execution_meta.agreement_score == pytest.approx(report["confidence_score"])


@pytest.mark.asyncio
async def test_validation_fallback_uses_submission_grounded_language_for_student_report() -> None:
    state = _state("student")
    state.pipeline_context.ingestion.text_content = (
        "Chapter 4: Research Methodology\n\n"
        "This chapter documents the research strategy, development method, resource selection, requirements modelling, "
        "and evaluation framework for EduAIPlatform. Design Science Research (DSR) was adopted as the methodological "
        "foundation (Hevner et al., 2004; Peffers et al., 2007).\n\n"
        "The chapter explains why DSR fits an artefact-focused project, but the connection between the framework and "
        "the concrete evaluation steps still needs to be made more explicit."
    )
    state.pipeline_context.rag = RagContext(
        enabled=False,
        context="",
        context_text="",
        citations=[],
        retrieved_chunks=[],
        chunk_count=0,
        weak_retrieval=True,
        confidence_score=0.2,
        confidence_label="low",
        safe_review=True,
        trace={"mode": "chapter"},
    )
    state.draft_report = {}

    state = await validation.validation_node(state)

    report = state.pipeline_context.report
    flattened = " ".join(
        [
            str(report.get("summary") or ""),
            str(report.get("overall_judgment") or ""),
            " ".join(str(item) for item in report.get("strengths") or []),
            " ".join(str(item) for item in report.get("weaknesses") or []),
            str((report.get("priority_issue") or {}).get("title") or ""),
        ]
    ).lower()

    assert "promising work" not in flattened
    assert "design science research" in flattened or "research methodology" in flattened
    assert any("deterministic validation fallback" in warning.lower() for warning in state.warnings)
