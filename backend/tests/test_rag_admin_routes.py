from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_rag_admin_coverage_endpoint_returns_expected_shape(async_client, monkeypatch):
    from app.rag import debug_coverage

    monkeypatch.setattr(
        debug_coverage,
        "coverage_report",
        lambda audience: {
            "audience": audience,
            "collection_name": f"{audience}_kb",
            "total_chunks": 10,
            "active_chunks": 8,
            "inactive_chunks": 2,
            "by_category": {"writing": 4},
            "active_by_category": {"writing": 3},
            "by_document": {"Guide": 4},
            "by_official": {"official": 6, "non_official": 4},
            "active_by_official": {"official": 5, "non_official": 3},
            "by_priority_bucket": {"80-100": 2},
            "active_by_priority_bucket": {"80-100": 2},
            "sample_chunk_ids": ["c1"],
        },
    )

    async with async_client as ac:
        response = await ac.get(
            "/api/rag/admin/coverage/student",
            headers={"Authorization": "Bearer token-admin"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["audience"] == "student"
    assert body["total_chunks"] == 10
    assert "by_official" in body
    assert "by_priority_bucket" in body


@pytest.mark.asyncio
async def test_rag_admin_retrieval_debug_endpoint_returns_result_and_eligibility(async_client, monkeypatch):
    import app.api.rag_admin as rag_admin
    from app.rag import debug_coverage

    class DummyResult:
        def __init__(self):
            self.trace = SimpleNamespace(applied_filters={"$and": [{"status": "active"}, {"audience": "student"}]})

        def model_dump(self):
            return {
                "ok": True,
                "audience": "student",
                "query": "essay feedback",
                "chunks": [],
                "citations": [],
                "confidence_score": 0.77,
                "confidence_label": "high",
                "safe_review": False,
                "trace": {
                    "expanded_queries": ["essay governance evidence"],
                    "applied_filters": {"$and": [{"status": "active"}, {"audience": "student"}]},
                    "keywords_used": ["argument", "evidence"],
                },
            }

    monkeypatch.setattr(rag_admin, "retrieve_student_context", lambda **_kwargs: DummyResult())
    monkeypatch.setattr(
        debug_coverage,
        "eligibility_report",
        lambda audience, applied_filters: {
            "audience": audience,
            "applied_filters": applied_filters,
            "eligible_chunks": 12,
            "eligible_by_category": {"writing": 5},
        },
    )

    async with async_client as ac:
        response = await ac.post(
            "/api/rag/admin/retrieval-debug/student",
            headers={"Authorization": "Bearer token-admin"},
            json={
                "query": "essay feedback",
                "analysis_type": "student_academic_review",
                "text_excerpt": "This essay compares evidence and argument quality.",
                "keywords": ["argument", "evidence"],
                "mode": "essay",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["confidence_label"] == "high"
    assert body["eligibility_summary"]["eligible_chunks"] == 12
    assert body["result"]["trace"]["keywords_used"] == ["argument", "evidence"]
