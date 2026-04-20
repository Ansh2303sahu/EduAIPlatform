"""
Tests for the Phase 10 prompt builders.
"""

from __future__ import annotations

from app.langchain.enums import AnalysisType, PipelineMode, PipelineRole
from app.langchain.models import IngestionBundle, PipelineContext, RagContext
from app.langchain.prompts.professor import (
    build_professor_prompt,
    build_professor_safe_prompt,
)
from app.langchain.prompts.student import (
    build_student_prompt,
    build_student_safe_prompt,
)


def _student_ctx() -> PipelineContext:
    return PipelineContext(
        request_id="req-student",
        file_id="file-student",
        submission_id="sub-student",
        role=PipelineRole.STUDENT,
        analysis_type=AnalysisType.STUDENT_PROJECT,
        mode=PipelineMode.NORMAL,
        submission_kind="project",
        ingestion=IngestionBundle(
            text_content="A FastAPI and React project with authentication and dashboards.",
            ocr_text="System diagram image caption.",
            audio_transcript="",
            tables_json={"features": ["auth", "dashboards"]},
        ),
        ml_raw={
            "feedback_category": "project_review",
            "quality_band": "high",
            "confidence_0_to_4": 3,
        },
        ml_context_text="Phase 6 student prediction summary.",
        rag=RagContext(
            enabled=True,
            context="Project guidance says to justify architecture and testing claims.",
            instruction="Use the approved project guidance conservatively.",
            citations=[{"index": 1, "title": "Project Guide", "section": "Testing"}],
            retrieved_chunks=[{"document_title": "Project Guide", "section": "Testing", "score": 0.91}],
            confidence_score=0.91,
            confidence_label="high",
            safe_review=False,
            trace={"mode": "code", "keywords_used": ["fastapi", "react"], "selected_titles": ["Project Guide"]},
        ),
    )


def _professor_ctx() -> PipelineContext:
    return PipelineContext(
        request_id="req-professor",
        file_id="file-professor",
        submission_id="sub-professor",
        role=PipelineRole.PROFESSOR,
        analysis_type=AnalysisType.PROFESSOR_ACADEMIC,
        mode=PipelineMode.NORMAL,
        submission_kind="academic",
        ingestion=IngestionBundle(
            text_content="An essay on cloud computing trade-offs and governance risks.",
            ocr_text="",
            audio_transcript="Governance was only briefly discussed.",
            tables_json=None,
        ),
        ml_raw={
            "rubric_band": "merit",
            "argument_depth": "high",
            "moderation_consistency": "high",
        },
        ml_context_text="Phase 6 professor prediction summary.",
        rag=RagContext(
            enabled=True,
            context="Moderation guidance says to compare structure, evidence, and clarity.",
            instruction="Use moderation guidance only when it clearly matches the essay evidence.",
            citations=[{"index": 1, "title": "Moderation Handbook", "section": "Rubric Use"}],
            retrieved_chunks=[{"document_title": "Moderation Handbook", "section": "Rubric Use", "score": 0.89}],
            confidence_score=0.89,
            confidence_label="high",
            safe_review=False,
            trace={"mode": "rubric", "keywords_used": ["rubric", "moderation"], "selected_titles": ["Moderation Handbook"]},
        ),
    )


def test_student_prompt_contains_required_sections() -> None:
    prompt = build_student_prompt(_student_ctx())

    assert "SHARED RULES" in prompt
    assert "ASSIGNMENT CONTEXT" in prompt
    assert "SUBMISSION DIGEST" in prompt
    assert "REPRESENTATIVE EXCERPTS" in prompt
    assert "PHASE 6 ML PREDICTIONS" in prompt
    assert "STUDENT RAG CONTEXT" in prompt
    assert "Never invent citations" in prompt
    assert '"summary"' in prompt
    assert '"overall_judgment"' in prompt
    assert '"section_feedback"' in prompt
    assert '"priority_issue"' in prompt
    assert '"strengths"' in prompt
    assert "Every criticism must explain what is weak" in prompt
    assert "quote or closely paraphrase a specific sentence" in prompt
    assert "Submission evidence is primary" in prompt
    assert "Focus on the actual submission" in prompt
    assert "Retrieval trace summary:" in prompt


def test_student_safe_prompt_contains_restricted_rules() -> None:
    prompt = build_student_safe_prompt(_student_ctx())

    assert 'Set confidence.mode to "restricted".' in prompt
    assert "safety.needs_review must be true." in prompt


def test_professor_prompt_contains_discrepancy_guidance() -> None:
    prompt = build_professor_prompt(_professor_ctx())

    assert "DISCREPANCY AND REVIEW BEHAVIOR" in prompt
    assert "ASSIGNMENT CONTEXT" in prompt
    assert "REPRESENTATIVE EXCERPTS" in prompt
    assert "PROFESSOR RAG CONTEXT" in prompt
    assert "rubric_breakdown" in prompt
    assert "evaluator_overview" in prompt
    assert "rubric_alignment" in prompt
    assert "section_observations" in prompt
    assert "Never invent citations" in prompt
    assert "rubric, policy, moderation consistency" in prompt
    assert "Every criticism must explain what is weak" in prompt


def test_professor_safe_prompt_contains_review_rules() -> None:
    prompt = build_professor_safe_prompt(_professor_ctx())

    assert "safety.needs_review must be true." in prompt
    assert "At least one moderation_notes item should explain the review risk" in prompt
