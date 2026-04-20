from __future__ import annotations

from app.langchain.models import RagContext
from app.langgraph.adapters import rag
from app.langgraph.schemas import Phase12ExecutionRequest
from app.langgraph.state import Phase12GraphState


def test_pack_rag_for_state_returns_context_object(monkeypatch):
    state = Phase12GraphState.create(
        Phase12ExecutionRequest(
            file_id="student-file",
            user_id="student-user",
            role="student",
            correlation_id="student-corr",
        )
    )

    def _fake_pack_student_rag(payload: dict[str, object]):
        enriched = {**payload, "query": "winforms recursion midpoint"}
        context = RagContext(
            enabled=True,
            context="Grounded retrieval context",
            context_text="Grounded retrieval context",
            retrieved_chunks=[{"chunk_id": "chunk-1", "score": 0.91}],
            chunk_count=1,
            confidence_score=0.91,
            confidence_label="high",
            trace={"query": "winforms recursion midpoint"},
        )
        return enriched, context

    monkeypatch.setattr(rag, "pack_student_rag", _fake_pack_student_rag)

    payload, context = rag.pack_rag_for_state(state)

    assert payload["query"] == "winforms recursion midpoint"
    assert isinstance(context, RagContext)
    assert context.trace["query"] == "winforms recursion midpoint"
