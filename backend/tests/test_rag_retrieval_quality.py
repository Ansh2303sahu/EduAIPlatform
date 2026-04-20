"""Tests for student RAG retrieval quality improvements.

Covers:
- submission-aware query building (essay vs code keywords)
- metadata filter correctness (audience=student, status=active always set)
- context_builder keyword extraction
- build_queries honours text_excerpt / keywords for submission-specific results
- active indexed docs from different categories are eligible under correct filters
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make sure the backend package is importable regardless of working directory.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# ---------------------------------------------------------------------------
# query_builder tests — no DB needed
# ---------------------------------------------------------------------------

class TestBuildQueries:
    """Test that build_queries produces submission-specific queries."""

    def _make_req(self, query: str, analysis_type: str = "", submission_type: str = "",
                  keywords: list[str] | None = None, text_excerpt: str | None = None):
        from app.rag.schemas import RetrievalFilters, RetrievalQuery
        return RetrievalQuery(
            audience="student",
            query=query,
            top_k=4,
            filters=RetrievalFilters(status="active"),
            analysis_type=analysis_type or None,
            submission_type=submission_type or None,
            keywords=keywords or [],
            text_excerpt=text_excerpt,
        )

    def test_essay_keywords_appear_in_queries(self):
        from app.rag.query_builder import build_queries
        req = self._make_req(
            query="student feedback on essay",
            keywords=["thesis", "argument", "harvard", "citation"],
        )
        queries = build_queries(req)
        combined = " ".join(queries).lower()
        # At least one submission-specific term should anchor the first query
        assert any(kw in combined for kw in ["thesis", "argument", "citation", "harvard"])

    def test_code_keywords_appear_in_queries(self):
        from app.rag.query_builder import build_queries
        req = self._make_req(
            query="student project review",
            analysis_type="student_project_review",
            keywords=["fastapi", "react", "database", "testing"],
        )
        queries = build_queries(req)
        combined = " ".join(queries).lower()
        assert any(kw in combined for kw in ["fastapi", "react", "database", "testing"])

    def test_submission_context_query_is_first(self):
        """When keywords are supplied, the first deduplicated query must reflect them."""
        from app.rag.query_builder import build_queries
        req = self._make_req(
            query="feedback",
            keywords=["django", "postgresql", "celery"],
        )
        queries = build_queries(req)
        # submission_extras are prepended, so they survive deduplication at slot 0
        assert len(queries) >= 1
        first = queries[0].lower()
        assert any(kw in first for kw in ["django", "postgresql", "celery"])

    def test_text_excerpt_expands_queries(self):
        """A text_excerpt without explicit keywords still produces submission-specific signal."""
        from app.rag.query_builder import build_queries
        req = self._make_req(
            query="project evaluation",
            analysis_type="student_project_review",
            text_excerpt="The backend is built with FastAPI and the database uses PostgreSQL.",
        )
        queries = build_queries(req)
        combined = " ".join(queries).lower()
        # fastapi or postgresql should appear after extraction from excerpt
        assert "fastapi" in combined or "postgresql" in combined

    def test_essay_submission_type_does_not_add_code_categories(self):
        """Essay-style query expansion should not inject code-specific terms."""
        from app.rag.query_builder import build_queries
        req = self._make_req(
            query="essay on risk assessment frameworks NIST STRIDE",
            keywords=["nist", "stride", "risk", "cia triad"],
        )
        queries = build_queries(req)
        combined = " ".join(queries).lower()
        # Should not have architecture_review or features_built language
        assert "features_built" not in combined
        assert "architecture_review" not in combined

    def test_query_count_does_not_exceed_max(self):
        from app.rag.query_builder import build_queries, _MAX_QUERY_COUNT
        req = self._make_req(
            query="essay argument thesis structure evidence referencing",
            keywords=["thesis", "harvard", "citation", "argument", "evidence"],
        )
        queries = build_queries(req)
        assert len(queries) <= _MAX_QUERY_COUNT

    def test_no_keywords_still_produces_queries(self):
        """Without any submission signal, generic base queries are still produced."""
        from app.rag.query_builder import build_queries
        req = self._make_req(query="student feedback")
        queries = build_queries(req)
        assert len(queries) >= 3


class TestPhase7SubmissionFocusedQuery:
    def test_project_query_keeps_project_anchor_for_title_heavy_submission(self):
        from app.api import phase7

        ingestion = {
            "text_content": (
                "EduAIPlatform A Locally Deployable AI-Driven Feedback System for Higher Education "
                "Final Report: Introduction | Literature Review | System Architecture | Methodology | "
                "ARU Harvard Referencing Style | 40+ sources. "
                "The implemented system uses FastAPI, React, Docker, analytics dashboards, and testing."
            )
        }

        query = phase7._student_project_query(ingestion)
        lowered = query.lower()

        assert "student software engineering project architecture" in lowered
        assert "architecture" in lowered
        assert "harvard referencing style" not in lowered
        assert query != " ".join((ingestion["text_content"] or "").split())[:200]

    def test_academic_query_falls_back_to_base_when_excerpt_has_no_academic_anchor(self):
        from app.api import phase7

        ingestion = {"text_content": "Appendix A | Contents | Page 1 | Submitted on time"}
        query = phase7._student_academic_query(ingestion)

        assert query == (
            "student academic writing structure argument evidence "
            "critical analysis clarity referencing citation"
        )


class TestSubmissionFormRouting:
    def test_chapter_like_academic_submission_is_not_routed_as_code(self):
        from app.services.report_generation_support import (
            classify_submission_form,
            detect_submission_kind,
        )

        ingestion = {
            "text_content": (
                "Introduction\n\n"
                "Background and Context\n\n"
                "This chapter examines formative feedback in higher education and the tension between timeliness "
                "and pedagogical depth in automated feedback systems. The discussion reviews related literature, "
                "frames the argument in academic prose, and cites prior work throughout the chapter (Smith, 2024). "
                "A later section discusses system architecture and testing, but the chapter remains prose-led and "
                "citation-heavy rather than code-heavy.\n\n"
                "References"
            )
        }

        assert classify_submission_form(ingestion) == "chapter"
        assert detect_submission_kind(ingestion) == "academic"

    def test_technical_exercise_summary_routes_as_code_project(self):
        from app.services.report_generation_support import (
            classify_submission_form,
            detect_submission_kind,
        )

        ingestion = {
            "text_content": (
                "Week 1 This exercise introduced basic drawing using WinForms and the System.Drawing library. "
                "Various shapes such as triangles, rectangles, ellipses, and polygons were created. "
                "Week 2 This exercise demonstrates recursive drawing of triangles using midpoint calculations. "
                "This highlights recursion and geometric subdivision techniques commonly used in computer graphics."
            )
        }

        assert classify_submission_form(ingestion) == "code"
        assert detect_submission_kind(ingestion) == "project"


# ---------------------------------------------------------------------------
# context_builder keyword extraction tests
# ---------------------------------------------------------------------------

class TestKeywordExtraction:

    def test_essay_text_extracts_essay_keywords(self):
        from app.rag.retrieval.context_builder import _extract_submission_keywords
        text = (
            "This essay critically analyses the CIA triad framework using APA referencing. "
            "The argument is supported by a literature review and Harvard citation style."
        )
        kws = _extract_submission_keywords(text, analysis_type=None)
        assert "apa" in kws or "harvard" in kws or "citation" in kws or "argument" in kws

    def test_code_text_extracts_code_keywords(self):
        from app.rag.retrieval.context_builder import _extract_submission_keywords
        text = (
            "The backend is implemented with FastAPI and SQLAlchemy. "
            "The frontend uses React with TailwindCSS. "
            "Authentication is handled via JWT. Testing is done with pytest."
        )
        kws = _extract_submission_keywords(text, analysis_type="student_project_review")
        assert any(kw in kws for kw in ["fastapi", "react", "testing", "auth", "authentication"])

    def test_essay_text_does_not_extract_code_keywords(self):
        from app.rag.retrieval.context_builder import _extract_submission_keywords
        text = (
            "This critical analysis examines the NIST cybersecurity framework "
            "and evaluates its effectiveness through a systematic literature review."
        )
        kws = _extract_submission_keywords(text, analysis_type=None)
        # essay keywords should dominate
        assert "nist" in kws or "methodology" in kws or "evaluation" in kws
        # No code-specific ones like "fastapi", "react"
        assert "fastapi" not in kws
        assert "react" not in kws

    def test_empty_text_returns_empty_list(self):
        from app.rag.retrieval.context_builder import _extract_submission_keywords
        assert _extract_submission_keywords("", analysis_type=None) == []
        assert _extract_submission_keywords(None, analysis_type=None) == []

    def test_text_excerpt_truncates_cleanly(self):
        from app.rag.retrieval.context_builder import _submission_text_excerpt
        long_text = "word " * 1000
        excerpt = _submission_text_excerpt(long_text, limit=500)
        assert excerpt is not None
        assert len(excerpt) <= 500

    def test_text_excerpt_collapses_whitespace(self):
        from app.rag.retrieval.context_builder import _submission_text_excerpt
        messy = "  hello   world\n\nfoo  "
        excerpt = _submission_text_excerpt(messy)
        assert excerpt == "hello world foo"

    def test_dynamic_keywords_supplement_fixed_vocab(self):
        from app.rag.retrieval.context_builder import _extract_submission_keywords

        text = (
            "The report evaluates a moving average alert engine for portfolio analytics. "
            "The moving average alert engine is compared against a momentum alert engine, "
            "and the discussion revisits the moving average alert engine in the evaluation section."
        )
        kws = _extract_submission_keywords(
            text,
            analysis_type="student_project_review",
            audience="student",
            title_hint="Portfolio Analytics Moving Average Alert Engine",
            mode="code",
        )
        assert any("moving average" in kw or "alert engine" in kw for kw in kws)

    def test_technical_exercise_keywords_drop_weekly_scaffold_noise(self):
        from app.rag.retrieval.context_builder import _extract_submission_keywords

        text = (
            "Week 1 This exercise introduced basic drawing using WinForms and the System.Drawing library. "
            "Week 2 This exercise demonstrates recursive drawing of triangles using midpoint calculations."
        )
        kws = _extract_submission_keywords(
            text,
            analysis_type="student_project_review",
            audience="student",
            title_hint="Week 1 WinForms recursive drawing exercise",
            mode="code",
        )

        assert "week" not in kws
        assert "exercise" not in kws
        assert any(
            kw in kws
            for kw in ["winforms", "system.drawing", "recursive drawing", "midpoint calculations", "midpoint"]
        )


class TestContextBuilderGrounding:
    def test_student_rag_payload_derives_query_from_submission_text(self, monkeypatch):
        from app.rag.retrieval import context_builder
        from app.rag.schemas import RetrievalResult, RetrievalTrace

        captured: dict[str, object] = {}

        def _fake_retrieve_student_context(**kwargs):
            captured.update(kwargs)
            return RetrievalResult(
                audience="student",
                query=kwargs["query"],
                chunks=[],
                citations=[],
                confidence_score=0.82,
                confidence_label="high",
                trace=RetrievalTrace(
                    audience="student",
                    query=kwargs["query"],
                    mode=kwargs["mode"],
                    keywords_used=list(kwargs["keywords"]),
                    title_hint=kwargs["title_hint"],
                    text_excerpt=kwargs["text_excerpt"],
                ),
                safe_review=False,
            )

        monkeypatch.setattr(context_builder, "retrieve_student_context", _fake_retrieve_student_context)
        body = {
            "query": "student architecture implementation testing security",
            "ingestion": {
                "text_content": (
                    "Introduction\n\n"
                    "This chapter explores formative feedback in higher education, focusing on automated feedback, "
                    "timeliness, and pedagogical depth. The introduction argues that faster feedback is not enough "
                    "without stronger educational value and cites recent higher-education studies (Smith, 2024)."
                )
            },
        }

        payload = context_builder.build_student_rag_payload(body)
        query = str(captured["query"]).lower()

        assert captured["mode"] == "chapter"
        assert "formative feedback" in query or "higher education" in query
        assert query != "student architecture implementation testing security"
        assert payload["trace"]["grounding_rejected"] is True

    def test_student_rag_payload_routes_technical_exercise_summary_to_project_queries(self, monkeypatch):
        from app.rag.retrieval import context_builder
        from app.rag.schemas import RetrievalResult, RetrievalTrace

        captured: dict[str, object] = {}

        def _fake_retrieve_student_context(**kwargs):
            captured.update(kwargs)
            return RetrievalResult(
                audience="student",
                query=kwargs["query"],
                chunks=[],
                citations=[],
                confidence_score=0.74,
                confidence_label="medium",
                trace=RetrievalTrace(
                    audience="student",
                    query=kwargs["query"],
                    mode=kwargs["mode"],
                    keywords_used=list(kwargs["keywords"]),
                    title_hint=kwargs["title_hint"],
                    text_excerpt=kwargs["text_excerpt"],
                ),
                safe_review=False,
            )

        monkeypatch.setattr(context_builder, "retrieve_student_context", _fake_retrieve_student_context)
        body = {
            "ingestion": {
                "text_content": (
                    "Week 1 This exercise introduced basic drawing using WinForms and the System.Drawing library. "
                    "Various shapes such as triangles, rectangles, ellipses, and polygons were created. "
                    "Week 2 This exercise demonstrates recursive drawing of triangles using midpoint calculations. "
                    "This highlights recursion and geometric subdivision techniques commonly used in computer graphics."
                )
            }
        }

        payload = context_builder.build_student_rag_payload(body)
        query = str(captured["query"]).lower()

        assert captured["mode"] == "code"
        assert any(term in query for term in ["winforms", "system.drawing", "recursion", "midpoint"])
        assert "academic writing" not in query
        assert "software_engineering" in captured["preferred_categories"]
        assert payload["trace"]["grounding_rejected"] is True

    def test_student_rag_payload_rejects_generic_off_topic_chunks(self, monkeypatch):
        from app.rag.retrieval import context_builder
        from app.rag.schemas import CitationOut, RetrievedChunk, RetrievalResult, RetrievalTrace

        def _fake_retrieve_student_context(**kwargs):
            return RetrievalResult(
                audience="student",
                query=kwargs["query"],
                chunks=[
                    RetrievedChunk(
                        chunk_id="chunk-1",
                        document_id="doc-1",
                        document_title="Generic Writing Guide",
                        section="Advice",
                        category="writing",
                        audience="student",
                        content="Use clear structure, coherent paragraphs, and evidence where appropriate.",
                        score=0.88,
                    ),
                    RetrievedChunk(
                        chunk_id="chunk-2",
                        document_id="doc-2",
                        document_title="Generic Referencing Guide",
                        section="Advice",
                        category="referencing",
                        audience="student",
                        content="Follow the chosen referencing style consistently and support claims with evidence.",
                        score=0.84,
                    ),
                ],
                citations=[
                    CitationOut(
                        title="Generic Writing Guide",
                        section="Advice",
                        document_id="doc-1",
                        chunk_id="chunk-1",
                        category="writing",
                        score=0.88,
                    )
                ],
                confidence_score=0.87,
                confidence_label="high",
                trace=RetrievalTrace(
                    audience="student",
                    query=kwargs["query"],
                    mode=kwargs["mode"],
                    keywords_used=list(kwargs["keywords"]),
                    title_hint=kwargs["title_hint"],
                    text_excerpt=kwargs["text_excerpt"],
                ),
                safe_review=False,
            )

        monkeypatch.setattr(context_builder, "retrieve_student_context", _fake_retrieve_student_context)
        payload = context_builder.build_student_rag_payload(
            {
                "ingestion": {
                    "text_content": (
                        "Introduction\n\n"
                        "This chapter investigates formative feedback in higher education and how automated feedback "
                        "can improve timeliness without sacrificing pedagogical depth or dialogic support."
                    )
                }
            }
        )

        assert payload["context"] == ""
        assert payload["retrieved_chunks"] == []
        assert payload["citations"] == []
        assert payload["safe_review"] is True
        assert payload["confidence_label"] == "low"
        assert "insufficient_topic_alignment" in payload["trace"]["grounding_rejection_reasons"]


# ---------------------------------------------------------------------------
# metadata filter correctness
# ---------------------------------------------------------------------------

class TestMetadataFilters:

    def _make_req(self, audience="student", category=None, status="active"):
        from app.rag.schemas import RetrievalFilters, RetrievalQuery
        return RetrievalQuery(
            audience=audience,
            query="test query",
            top_k=4,
            filters=RetrievalFilters(status=status, category=category),
        )

    def test_student_filter_always_includes_active_and_audience(self):
        from app.rag.retriever import _metadata_filter_dict
        req = self._make_req(audience="student")
        f = _metadata_filter_dict(req)
        # Must have both status=active and audience=student
        assert f is not None
        flat = str(f)
        assert "active" in flat
        assert "student" in flat

    def test_category_filter_included_when_set(self):
        from app.rag.retriever import _metadata_filter_dict
        req = self._make_req(audience="student", category="referencing")
        f = _metadata_filter_dict(req)
        assert f is not None
        assert "referencing" in str(f)

    def test_no_category_filter_does_not_restrict_categories(self):
        """Without a category filter, all active student docs across categories are eligible."""
        from app.rag.retriever import _metadata_filter_dict
        req = self._make_req(audience="student", category=None)
        f = _metadata_filter_dict(req)
        assert f is not None
        # category key must NOT appear as a filter condition
        assert "category" not in str(f)

    def test_professor_filter_includes_professor_audience(self):
        from app.rag.retriever import _metadata_filter_dict
        req = self._make_req(audience="professor")
        f = _metadata_filter_dict(req)
        assert "professor" in str(f)
        assert "student" not in str(f)


class TestContextBuilderFallbacks:
    def test_professor_payload_uses_fallback_text_when_ingestion_missing(self, monkeypatch):
        from app.rag.retrieval import context_builder

        captured: dict[str, object] = {}

        class DummyResult:
            def __init__(self):
                self.chunks = []
                self.citations = []
                self.confidence_score = 0.62
                self.confidence_label = "medium"
                self.safe_review = False
                self.trace = MagicMock(model_dump=lambda: {"ok": True})

        def fake_retrieve_professor_context(**kwargs):
            captured.update(kwargs)
            return DummyResult()

        monkeypatch.setattr(context_builder, "retrieve_professor_context", fake_retrieve_professor_context)
        monkeypatch.setattr(context_builder, "build_grounding_context", lambda _chunks: "")

        payload = context_builder.build_professor_rag_payload(
            {
                "analysis_type": "professor_academic_review",
                "query": "rubric marking policy moderation for cybersecurity governance report",
                "task": "Assess the submission against rubric and moderation expectations",
                "ml": {"rubric_band": "merit"},
                "official_only": True,
            }
        )

        assert captured["degraded_input"] is True
        assert isinstance(captured["text_excerpt"], str) and len(str(captured["text_excerpt"])) > 0
        assert isinstance(captured["keywords"], list) and len(captured["keywords"]) > 0
        assert captured["title_hint"]
        assert payload["trace"]["ok"] is True
        assert payload["trace"]["grounding_rejected"] is True

    def test_professor_policy_mode_prefers_policy_categories(self, monkeypatch):
        from app.rag.retrieval import context_builder

        captured: dict[str, object] = {}

        class DummyResult:
            def __init__(self):
                self.chunks = []
                self.citations = []
                self.confidence_score = 0.58
                self.confidence_label = "medium"
                self.safe_review = False
                self.trace = MagicMock(model_dump=lambda: {"ok": True})

        def fake_retrieve_professor_context(**kwargs):
            captured.update(kwargs)
            return DummyResult()

        monkeypatch.setattr(context_builder, "retrieve_professor_context", fake_retrieve_professor_context)
        monkeypatch.setattr(context_builder, "build_grounding_context", lambda _chunks: "")

        context_builder.build_professor_rag_payload(
            {
                "analysis_type": "professor_academic_review",
                "query": "marking policy moderation feedback template for written submissions",
                "ingestion": {"text_content": "The brief requires moderation-safe rubric feedback."},
            }
        )

        assert captured["mode"] in {"policy", "feedback", "moderation", "rubric"}
        assert "marking_policy" in captured["preferred_categories"]


class TestRerankingAndTrace:
    def test_reranker_prefers_aligned_category_for_student_essay(self):
        from app.rag.reranker import rerank_chunks
        from app.rag.schemas import RetrievalFilters, RetrievalQuery, RetrievedChunk

        req = RetrievalQuery(
            audience="student",
            query="essay feedback on argument evidence and referencing",
            top_k=4,
            filters=RetrievalFilters(status="active"),
            analysis_type="student_academic_review",
            preferred_categories=["writing", "referencing", "critical_thinking"],
            title_hint="Argument and Evidence in Platform Governance",
            keywords=["argument", "evidence", "referencing"],
            text_excerpt="The essay compares viewpoints and evaluates evidence.",
            mode="essay",
        )
        aligned = RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            document_title="Academic Writing Guide",
            section="Evidence",
            category="writing",
            audience="student",
            content="Strong essays explain why evidence matters and compare viewpoints clearly.",
            score=0.55,
        )
        unrelated = RetrievedChunk(
            chunk_id="c2",
            document_id="d2",
            document_title="Software Architecture Notes",
            section="Layers",
            category="software_engineering",
            audience="student",
            content="Discuss controller-service-repository separation in a web stack.",
            score=0.55,
        )

        ranked = rerank_chunks(req, [unrelated, aligned])
        assert ranked[0].chunk_id == "c1"

    def test_reranker_prefers_technical_category_for_student_project(self):
        from app.rag.reranker import rerank_chunks
        from app.rag.schemas import RetrievalFilters, RetrievalQuery, RetrievedChunk

        req = RetrievalQuery(
            audience="student",
            query="software project review architecture testing security maintainability",
            top_k=4,
            filters=RetrievalFilters(status="active"),
            analysis_type="student_project_review",
            preferred_categories=["software_engineering", "project_evaluation", "research_support"],
            title_hint="GrafPack Windows Forms Drawing Application",
            keywords=["windows forms", "c#", "testing", "drawing application"],
            text_excerpt="The coursework evaluates implementation, testing, usability, and architecture decisions.",
            mode="code",
        )
        technical = RetrievedChunk(
            chunk_id="tech-1",
            document_id="d-tech",
            document_title="Software Architecture Notes",
            section="Testing and Security",
            category="software_engineering",
            audience="student",
            content="Discuss module boundaries, testing strategy, validation, and secure input handling.",
            score=0.52,
        )
        writing = RetrievedChunk(
            chunk_id="writing-1",
            document_id="d-writing",
            document_title="Essay Structure",
            section="Paragraphing",
            category="writing",
            audience="student",
            content="Use clear topic sentences and paragraph transitions in essays.",
            score=0.57,
        )

        ranked = rerank_chunks(req, [writing, technical])
        assert ranked[0].chunk_id == "tech-1"

    def test_build_trace_populates_extended_fields(self):
        from app.rag.schemas import RetrievalFilters, RetrievalQuery, RetrievedChunk
        from app.rag.trace import build_trace

        req = RetrievalQuery(
            audience="student",
            query="essay feedback",
            top_k=4,
            filters=RetrievalFilters(status="active"),
            keywords=["argument", "citation"],
            text_excerpt="This essay compares evidence and argument quality.",
            title_hint="Essay on Governance and Evidence",
            mode="essay",
            degraded_input=True,
        )
        final_chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                document_title="Writing Guide",
                section="Argument",
                category="writing",
                audience="student",
                content="Focus on argument quality.",
                score=0.81,
            )
        ]

        trace = build_trace(
            req=req,
            queries=["essay governance evidence", "citation referencing"],
            retrieved_chunk_ids=["c1", "c2"],
            final_chunks=final_chunks,
            retrieved_titles=["Writing Guide", "Referencing Guide"],
            scores=[0.81, 0.65],
            collection_name="student_kb",
            applied_filters={"$and": [{"status": "active"}, {"audience": "student"}]},
            final_categories=["writing"],
            reranking_changed_order=True,
            initial_candidate_count=2,
            confidence_score=0.78,
            confidence_label="high",
            eligibility_summary={"eligible_chunks": 12},
        )

        assert trace.expanded_queries == ["essay governance evidence", "citation referencing"]
        assert trace.keywords_used == ["argument", "citation"]
        assert trace.text_excerpt == "This essay compares evidence and argument quality."
        assert trace.title_hint == "Essay on Governance and Evidence"
        assert trace.mode == "essay"
        assert trace.degraded_input is True
        assert trace.initial_candidate_count == 2
        assert trace.confidence_label == "high"
        assert trace.eligibility_summary["eligible_chunks"] == 12
