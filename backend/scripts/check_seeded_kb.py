from __future__ import annotations

import _bootstrap  # noqa: F401

from app.rag.retrieval.student_retriever import retrieve_student_context
from app.rag.retrieval.professor_retriever import retrieve_professor_context


def main() -> None:
    print("\n=== Student retrieval check ===")
    student_result = retrieve_student_context(
        query="citation help and essay structure",
        top_k=4,
    )
    print(student_result.model_dump())

    print("\n=== Professor retrieval check ===")
    professor_result = retrieve_professor_context(
        query="rubric guidance and actionable feedback",
        top_k=4,
    )
    print(professor_result.model_dump())


if __name__ == "__main__":
    main()
