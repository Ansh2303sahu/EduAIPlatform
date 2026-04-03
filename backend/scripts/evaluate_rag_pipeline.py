from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict

import _bootstrap  # noqa: F401

from app.rag.analytics.evaluation import evaluate_retrieval_cases
from app.rag.analytics.retrieval_metrics import summarize_retrieval_results
from app.rag.schemas import RetrievalFilters, RetrievalQuery


class EvaluationRow(TypedDict):
    role: str
    query: str
    expected_topic: str
    top_result_category: str
    retrieved_correct: str
    confidence_score: float
    confidence_label: str
    safe_review: str
    top_document: str


TEST_CASES = [
    {"role": "student", "query": "citation help", "expected_topic": "referencing"},
    {"role": "student", "query": "essay structure", "expected_topic": "writing"},
    {"role": "student", "query": "paragraph development", "expected_topic": "writing"},
    {"role": "student", "query": "critical analysis", "expected_topic": "critical_thinking"},
    {"role": "student", "query": "avoid plagiarism", "expected_topic": "academic_integrity"},
    {"role": "professor", "query": "rubric guidance", "expected_topic": "rubrics"},
    {"role": "professor", "query": "actionable feedback", "expected_topic": "feedback_templates"},
    {"role": "professor", "query": "marking consistency", "expected_topic": "marking_policy"},
    {"role": "professor", "query": "moderation notes", "expected_topic": "moderation"},
    {"role": "professor", "query": "high quality writing", "expected_topic": "academic_quality"},
    {"role": "student", "query": "football scores tonight", "expected_topic": "out_of_scope"},
    {"role": "professor", "query": "astrophysics derivation", "expected_topic": "out_of_scope"},
]


def _to_query(case: dict[str, str]) -> RetrievalQuery:
    return RetrievalQuery(
        audience=case["role"],
        query=case["query"],
        top_k=4,
        filters=RetrievalFilters(status="active"),
    )


def main() -> None:
    rows: list[EvaluationRow] = []
    queries = [_to_query(case) for case in TEST_CASES]
    results = evaluate_retrieval_cases(queries)

    for case, result in zip(TEST_CASES, results):

        top_category = result.chunks[0].category if result.chunks else ""
        retrieved_correct = "yes" if top_category == case["expected_topic"] else "no"

        if case["expected_topic"] == "out_of_scope":
            retrieved_correct = "yes" if result.safe_review or result.confidence_label == "low" else "no"

        rows.append(
            {
                "role": case["role"],
                "query": case["query"],
                "expected_topic": case["expected_topic"],
                "top_result_category": top_category,
                "retrieved_correct": retrieved_correct,
                "confidence_score": round(result.confidence_score, 2),
                "confidence_label": result.confidence_label,
                "safe_review": str(result.safe_review).lower(),
                "top_document": result.chunks[0].document_title if result.chunks else "",
            }
        )

    out_path = Path("rag_evaluation_results.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "role",
                "query",
                "expected_topic",
                "top_result_category",
                "retrieved_correct",
                "confidence_score",
                "confidence_label",
                "safe_review",
                "top_document",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved evaluation results to {out_path.resolve()}")

    if not rows:
        print("\nNo evaluation rows were generated.")
        return

    retrieval_accuracy = (
        sum(1 for row in rows if row["retrieved_correct"] == "yes") / len(rows)
    )
    summary = summarize_retrieval_results(results)

    print("\n===== RAG EVALUATION SUMMARY =====")
    print("Avg Confidence:", summary["avg_confidence"])
    print("High Confidence Rate:", summary["high_confidence_rate"])
    print("Safe Review Rate:", summary["safe_review_rate"])
    print("Avg Chunks:", summary["avg_chunks"])

    print("\nRetrieval Accuracy:", retrieval_accuracy)
    print("Confidence > 0.3 Rate:", sum(1 for r in results if r.confidence_score > 0.3) / len(results))


if __name__ == "__main__":
    main()
