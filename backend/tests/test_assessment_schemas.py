"""Phase 14 — Unit tests for assessment Pydantic schemas.

Tests cover:
  - Default construction and field validation for all schema types
  - OpenAI result instantiation with rubric scores
  - Claude review with corrections
  - Gemini extraction (used vs skipped)
  - Gate decision logic fields
  - MultiModelAssessmentResult assembly
  - AssessmentAuditRecord field constraints
  - AssessmentEscalationRecord status defaults
  - AssessmentResultIn round-trip serialisation
  - AssessmentRequestedPayload constraints
  - RubricContextOut field population
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.assessment import (
    AssessmentAuditRecord,
    AssessmentEscalateIn,
    AssessmentEscalationRecord,
    AssessmentGateDecision,
    AssessmentIssue,
    AssessmentMetricIn,
    AssessmentRequestedPayload,
    AssessmentResultIn,
    AssessmentStrength,
    AssessmentValidateIn,
    ClaudeCorrection,
    ClaudeReviewResult,
    FigureAnalysis,
    GeminiExtractionResult,
    ImprovementAction,
    MultiModelAssessmentResult,
    OpenAIAssessmentResult,
    ProviderUsageStats,
    RubricContextOut,
    RubricCriterion,
    RubricScore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_result(**kwargs) -> OpenAIAssessmentResult:
    defaults = dict(
        rubric_scores=[
            RubricScore(
                criterion="Technical Accuracy",
                band="Merit",
                score=72.0,
                justification="Solid understanding with minor gaps.",
            )
        ],
        overall_grade="Merit",
        overall_score=72.0,
        summary="A competent submission with room for improvement.",
        confidence=0.82,
        usage=ProviderUsageStats(
            model="gpt-4o",
            prompt_tokens=1500,
            completion_tokens=400,
            total_tokens=1900,
            latency_ms=2300,
            cost_usd=0.0057,
        ),
    )
    defaults.update(kwargs)
    return OpenAIAssessmentResult(**defaults)


def _make_claude_review(**kwargs) -> ClaudeReviewResult:
    defaults = dict(
        consistent=True,
        reviewer_confidence=0.91,
        overall_verdict="approved",
        usage=ProviderUsageStats(
            model="claude-sonnet-4-6",
            prompt_tokens=2000,
            completion_tokens=300,
            total_tokens=2300,
            latency_ms=1800,
            cost_usd=0.0069,
        ),
    )
    defaults.update(kwargs)
    return ClaudeReviewResult(**defaults)


def _make_gate(pass_gate: bool = True, escalate: bool = False) -> AssessmentGateDecision:
    return AssessmentGateDecision(
        pass_gate=pass_gate,
        escalate=escalate,
        hitl_required=escalate,
        final_confidence=0.82,
        confidence_sources={"openai": 0.82, "claude": 0.91},
    )


# ---------------------------------------------------------------------------
# RubricCriterion
# ---------------------------------------------------------------------------

class TestRubricCriterion:
    def test_defaults(self):
        rc = RubricCriterion(criterion="Accuracy")
        assert rc.weight == 1.0
        assert rc.description == ""

    def test_weight_bounds(self):
        with pytest.raises(ValidationError):
            RubricCriterion(criterion="X", weight=1.5)

    def test_empty_criterion_rejected(self):
        with pytest.raises(ValidationError):
            RubricCriterion(criterion="")


# ---------------------------------------------------------------------------
# RubricScore
# ---------------------------------------------------------------------------

class TestRubricScore:
    def test_valid(self):
        rs = RubricScore(criterion="Clarity", band="Distinction", score=88.0)
        assert rs.evidence_quotes == []

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            RubricScore(criterion="X", band="Pass", score=101.0)


# ---------------------------------------------------------------------------
# AssessmentIssue / AssessmentStrength / ImprovementAction
# ---------------------------------------------------------------------------

class TestSubSchemas:
    def test_issue_severity_values(self):
        for sev in ("low", "med", "high"):
            ai = AssessmentIssue(title="Bug", evidence="Line 42", severity=sev)
            assert ai.severity == sev

    def test_issue_invalid_severity(self):
        with pytest.raises(ValidationError):
            AssessmentIssue(title="Bug", evidence="X", severity="critical")

    def test_strength_defaults(self):
        s = AssessmentStrength(title="Clear code", evidence="Section 2")
        assert s.title == "Clear code"

    def test_improvement_priority_bounds(self):
        with pytest.raises(ValidationError):
            ImprovementAction(action="Fix", why="Because", how="Do it", priority=0)
        with pytest.raises(ValidationError):
            ImprovementAction(action="Fix", why="Because", how="Do it", priority=11)


# ---------------------------------------------------------------------------
# ProviderUsageStats
# ---------------------------------------------------------------------------

class TestProviderUsageStats:
    def test_defaults(self):
        pus = ProviderUsageStats(model="gpt-4o")
        assert pus.total_tokens == 0
        assert pus.cost_usd == 0.0

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError):
            ProviderUsageStats(model="gpt-4o", cost_usd=-0.01)


# ---------------------------------------------------------------------------
# OpenAIAssessmentResult
# ---------------------------------------------------------------------------

class TestOpenAIAssessmentResult:
    def test_defaults(self):
        r = OpenAIAssessmentResult()
        assert r.model_id == "gpt-4o"
        assert r.rubric_scores == []
        assert r.confidence == 0.0
        assert r.needs_human_review is False

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            OpenAIAssessmentResult(confidence=1.5)

    def test_full_construction(self):
        r = _make_openai_result()
        assert r.overall_grade == "Merit"
        assert r.overall_score == 72.0
        assert len(r.rubric_scores) == 1
        assert r.rubric_scores[0].band == "Merit"

    def test_serialisation_round_trip(self):
        r = _make_openai_result()
        dumped = r.model_dump(mode="json")
        restored = OpenAIAssessmentResult.model_validate(dumped)
        assert restored.overall_score == r.overall_score
        assert restored.usage.model == "gpt-4o"


# ---------------------------------------------------------------------------
# ClaudeReviewResult
# ---------------------------------------------------------------------------

class TestClaudeReviewResult:
    def test_defaults(self):
        r = ClaudeReviewResult()
        assert r.consistent is True
        assert r.corrections == []
        assert r.flagged_for_hitl is False
        assert r.overall_verdict == "approved"

    def test_invalid_verdict(self):
        with pytest.raises(ValidationError):
            ClaudeReviewResult(overall_verdict="unknown")

    def test_with_corrections(self):
        correction = ClaudeCorrection(
            field_path="overall_grade",
            original_value="Distinction",
            suggested_value="Merit",
            reason="Score of 72 does not meet Distinction threshold.",
        )
        r = ClaudeReviewResult(
            consistent=False,
            reviewer_confidence=0.88,
            corrections=[correction],
            flagged_for_hitl=True,
            overall_verdict="needs_correction",
        )
        assert len(r.corrections) == 1
        assert r.corrections[0].field_path == "overall_grade"
        assert r.flagged_for_hitl is True


# ---------------------------------------------------------------------------
# GeminiExtractionResult
# ---------------------------------------------------------------------------

class TestGeminiExtractionResult:
    def test_defaults_skip(self):
        r = GeminiExtractionResult()
        assert r.multimodal_used is False
        assert r.figures == []

    def test_with_figures(self):
        fig = FigureAnalysis(
            figure_id="fig-1",
            figure_type="diagram",
            description="UML class diagram",
            quality_score=0.85,
        )
        r = GeminiExtractionResult(
            multimodal_used=True,
            figures=[fig],
            additional_context="UML diagram shows clean MVC separation.",
        )
        assert r.multimodal_used is True
        assert r.figures[0].figure_type == "diagram"

    def test_invalid_figure_type(self):
        with pytest.raises(ValidationError):
            FigureAnalysis(figure_id="f1", figure_type="spreadsheet")


# ---------------------------------------------------------------------------
# AssessmentGateDecision
# ---------------------------------------------------------------------------

class TestAssessmentGateDecision:
    def test_gate_pass(self):
        g = _make_gate(pass_gate=True, escalate=False)
        assert g.pass_gate is True
        assert g.hitl_required is False

    def test_gate_escalate(self):
        g = AssessmentGateDecision(
            pass_gate=False,
            escalate=True,
            hitl_required=True,
            escalation_reasons=["confidence < 0.40"],
            final_confidence=0.35,
        )
        assert "confidence < 0.40" in g.escalation_reasons

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            AssessmentGateDecision(pass_gate=True, escalate=False, hitl_required=False, final_confidence=1.5)


# ---------------------------------------------------------------------------
# MultiModelAssessmentResult
# ---------------------------------------------------------------------------

class TestMultiModelAssessmentResult:
    def test_construction(self):
        r = MultiModelAssessmentResult(
            file_id="file-123",
            user_id="user-456",
            role="student",
            overall_grade="Merit",
            overall_score=72.0,
            gate=_make_gate(),
        )
        assert r.file_id == "file-123"
        assert r.assessment_id != ""  # UUID auto-generated
        assert r.workflow_version == "v2"

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            MultiModelAssessmentResult(
                file_id="f1",
                user_id="u1",
                role="admin",
                gate=_make_gate(),
            )


# ---------------------------------------------------------------------------
# AssessmentAuditRecord
# ---------------------------------------------------------------------------

class TestAssessmentAuditRecord:
    def test_defaults(self):
        r = AssessmentAuditRecord(file_id="f1", user_id="u1", role="student")
        assert r.final_status == "failed"
        assert r.openai_invoked is False
        assert r.audit_id != ""

    def test_invalid_final_status(self):
        with pytest.raises(ValidationError):
            AssessmentAuditRecord(
                file_id="f1", user_id="u1", role="student", final_status="unknown"
            )

    def test_completed_status(self):
        r = AssessmentAuditRecord(
            file_id="f1",
            user_id="u1",
            role="student",
            openai_invoked=True,
            claude_invoked=True,
            gate_passed=True,
            final_status="completed",
            total_cost_usd=0.0126,
        )
        assert r.gate_passed is True
        assert r.total_cost_usd == pytest.approx(0.0126)


# ---------------------------------------------------------------------------
# AssessmentEscalationRecord
# ---------------------------------------------------------------------------

class TestAssessmentEscalationRecord:
    def test_defaults(self):
        r = AssessmentEscalationRecord(file_id="f1", user_id="u1", role="student")
        assert r.status == "pending"
        assert r.severity == "medium"
        assert r.escalation_id != ""

    def test_severity_values(self):
        for sev in ("low", "medium", "high", "critical"):
            r = AssessmentEscalationRecord(file_id="f", user_id="u", role="student", severity=sev)
            assert r.severity == sev


# ---------------------------------------------------------------------------
# AssessmentResultIn (inbound payload from n8n)
# ---------------------------------------------------------------------------

class TestAssessmentResultIn:
    def test_valid_construction(self):
        body = AssessmentResultIn(
            event_id="evt-abc",
            file_id="file-123",
            user_id="user-456",
            role="student",
            openai_result=_make_openai_result(),
            claude_review=_make_claude_review(),
            gate_decision=_make_gate(),
        )
        assert body.event_id == "evt-abc"
        assert body.gemini_extraction is None

    def test_with_gemini(self):
        body = AssessmentResultIn(
            event_id="evt-xyz",
            file_id="f1",
            user_id="u1",
            role="professor",
            openai_result=_make_openai_result(),
            claude_review=_make_claude_review(),
            gemini_extraction=GeminiExtractionResult(
                multimodal_used=True,
                figures=[FigureAnalysis(figure_id="fig1", figure_type="chart")],
            ),
            gate_decision=_make_gate(),
        )
        assert body.gemini_extraction is not None
        assert body.gemini_extraction.multimodal_used is True

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            AssessmentResultIn(
                event_id="e1",
                # file_id missing
                user_id="u1",
                role="student",
                openai_result=_make_openai_result(),
                claude_review=_make_claude_review(),
                gate_decision=_make_gate(),
            )


# ---------------------------------------------------------------------------
# AssessmentEscalateIn
# ---------------------------------------------------------------------------

class TestAssessmentEscalateIn:
    def test_valid(self):
        body = AssessmentEscalateIn(
            event_id="evt-esc-1",
            file_id="f1",
            user_id="u1",
            reasons=["confidence < 0.40"],
            openai_confidence=0.32,
            severity="high",
        )
        assert body.severity == "high"
        assert body.openai_confidence == pytest.approx(0.32)

    def test_invalid_severity(self):
        with pytest.raises(ValidationError):
            AssessmentEscalateIn(
                event_id="e1", file_id="f1", user_id="u1", severity="extreme"
            )


# ---------------------------------------------------------------------------
# AssessmentMetricIn
# ---------------------------------------------------------------------------

class TestAssessmentMetricIn:
    def test_valid(self):
        m = AssessmentMetricIn(metric="openai.calls", value=1)
        assert m.metric == "openai.calls"

    def test_invalid_metric_pattern(self):
        with pytest.raises(ValidationError):
            AssessmentMetricIn(metric="openai calls")  # space not allowed

    def test_zero_value_rejected(self):
        with pytest.raises(ValidationError):
            AssessmentMetricIn(metric="openai.calls", value=0)


# ---------------------------------------------------------------------------
# AssessmentRequestedPayload
# ---------------------------------------------------------------------------

class TestAssessmentRequestedPayload:
    def test_defaults(self):
        p = AssessmentRequestedPayload(
            file_id="f1", user_id="u1", role="student"
        )
        assert p.has_images is False
        assert p.workflow_version == "v2"
        assert p.pipeline == "phase12_langgraph"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            AssessmentRequestedPayload(
                file_id="f1", user_id="u1", role="student", draft_confidence=1.5
            )


# ---------------------------------------------------------------------------
# RubricContextOut
# ---------------------------------------------------------------------------

class TestRubricContextOut:
    def test_defaults(self):
        ctx = RubricContextOut(
            file_id="f1", user_id="u1", role="student"
        )
        assert ctx.rubric_criteria == []
        assert ctx.has_images is False
        assert ctx.schema_version == "14.1"

    def test_with_criteria(self):
        ctx = RubricContextOut(
            file_id="f1",
            user_id="u1",
            role="professor",
            rubric_criteria=[
                RubricCriterion(criterion="Accuracy", weight=0.5),
                RubricCriterion(criterion="Clarity", weight=0.5),
            ],
        )
        assert len(ctx.rubric_criteria) == 2
        assert ctx.rubric_criteria[0].weight == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# AssessmentValidateIn
# ---------------------------------------------------------------------------

class TestAssessmentValidateIn:
    def test_valid_no_gemini(self):
        v = AssessmentValidateIn(
            openai_result=_make_openai_result(),
            claude_review=_make_claude_review(),
            gate_decision=_make_gate(),
        )
        assert v.gemini_extraction is None

    def test_serialise_and_restore(self):
        v = AssessmentValidateIn(
            openai_result=_make_openai_result(),
            claude_review=_make_claude_review(),
            gate_decision=_make_gate(),
            final_report={"summary": "Good work"},
        )
        data = v.model_dump(mode="json")
        restored = AssessmentValidateIn.model_validate(data)
        assert restored.final_report["summary"] == "Good work"
