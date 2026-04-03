from typing import Any

from app.rag import retriever as rag_retriever
from app.rag.schemas import RetrievalQuery


def build_context(chunks: list[Any]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        score = getattr(c, "score", None)
        official = "official" if bool(getattr(c, "is_official", False)) else "supporting"
        parts.append(
            "\n".join(
                [
                    f"[Source {i}]",
                    f"Title: {getattr(c, 'document_title', 'Untitled')}",
                    f"Section: {getattr(c, 'section', '')}",
                    f"Category: {getattr(c, 'category', '')}",
                    f"Type: {official}",
                    f"Score: {score:.2f}" if isinstance(score, (int, float)) else "Score: unknown",
                    f"Content: {getattr(c, 'content', '')}",
                ]
            )
        )
    return "\n\n".join(parts)


async def run_rag_llm(
    *,
    retrieval_query: RetrievalQuery,
    extracted_text: str,
    llm_client,
):
    result = rag_retriever.retrieve(retrieval_query)

    context = build_context(result.chunks)

    prompt = f"""
Use ONLY the provided sources.

Confidence: {result.confidence_score} ({result.confidence_label})
Safe review: {result.safe_review}

Sources:
{context}

User submission:
{extracted_text}

Return JSON only.
"""

    llm_output = await llm_client.generate_json(
        prompt=prompt,
        safe_mode=result.safe_review,
    )

    return {
        "llm": llm_output,
        "rag": result,
    }
