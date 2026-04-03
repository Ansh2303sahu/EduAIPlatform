from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split loaded documents into retrieval-ready chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    enriched: List[Document] = []
    for idx, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata["chunk_index"] = idx
        chunk.metadata = metadata
        enriched.append(chunk)

    return enriched