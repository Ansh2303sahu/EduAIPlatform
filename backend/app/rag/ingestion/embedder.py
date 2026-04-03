from __future__ import annotations

from langchain_core.documents import Document

from app.rag.embeddings import get_embedding_model


def embed_chunk_texts(documents: list[Document]) -> list[list[float]]:
    texts = [doc.page_content for doc in documents]
    if not texts:
        return []
    return get_embedding_model().embed_documents(texts)
