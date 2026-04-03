from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from langchain_core.documents import Document

from app.rag.ingestion.document_loader import load_documents
from app.rag.ingestion.chunker import chunk_documents
from app.rag.ingestion.validation import validate_ingestion_request
from app.rag.vector_store import get_vector_store


def _sha256_file(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_metadata_value(value: Any) -> str | int | float | bool:
    """
    Chroma only accepts primitive metadata values.
    Convert None and complex types into safe strings.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return str(value)


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, str | int | float | bool]:
    return {str(k): _safe_metadata_value(v) for k, v in metadata.items()}


def _flatten_ids(raw_ids: Any) -> list[str]:
    if raw_ids is None:
        return []
    if isinstance(raw_ids, str):
        return [raw_ids]
    flattened: list[str] = []
    if isinstance(raw_ids, (list, tuple)):
        for item in raw_ids:
            flattened.extend(_flatten_ids(item))
    return flattened


def _delete_existing_document_chunks(store: Any, document_id: str) -> int:
    existing = store.get(where={"document_id": document_id}, include=[])
    ids = _flatten_ids(existing.get("ids"))
    if ids:
        store.delete(ids=ids)
    return len(ids)


def _prepare_chunk_metadata(
    *,
    chunk: Document,
    chunk_id: str,
    document_id: str,
    document_title: str,
    audience: str,
    category: str,
    version: str,
    uploaded_by: str,
    source_priority: int,
    is_official: bool,
    parent_doc_id: str | None,
    effective_date: str | None,
    status: str,
    total_chunks: int,
) -> Dict[str, Any]:
    chunk_index = int(chunk.metadata.get("chunk_index", 0))

    prev_chunk_id = f"{document_id}:chunk:{chunk_index - 1}" if chunk_index > 0 else ""
    next_chunk_id = f"{document_id}:chunk:{chunk_index + 1}" if chunk_index < total_chunks - 1 else ""

    page_or_section = (
        chunk.metadata.get("page")
        or chunk.metadata.get("page_number")
        or chunk.metadata.get("section")
        or "unknown"
    )

    metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_title": document_title,
        "section": str(page_or_section),
        "audience": audience,
        "category": category,
        "version": version,
        "effective_date": effective_date,
        "status": status,
        "uploaded_by": uploaded_by,
        "source_priority": source_priority,
        "is_official": is_official,
        "parent_doc_id": parent_doc_id,
        "chunk_index": chunk_index,
        "prev_chunk_id": prev_chunk_id,
        "next_chunk_id": next_chunk_id,
    }

    return _sanitize_metadata(metadata)


def ingest_knowledge_document(
    *,
    file_path: str,
    audience: str,
    category: str,
    uploaded_by: str,
    document_title: str | None = None,
    document_id: str | None = None,
    version: str = "v1",
    source_priority: int = 50,
    is_official: bool = False,
    parent_doc_id: str | None = None,
    effective_date: str | None = None,
    status: str = "active",
) -> Dict[str, Any]:
    """
    Phase 9 ingestion pipeline:
    load -> clean -> chunk -> enrich metadata -> embed -> store
    """
    audience = (audience or "").strip().lower()
    validate_ingestion_request(file_path, audience, category)
    path = Path(file_path)

    docs = load_documents(file_path)
    if not docs:
        raise ValueError("No readable content found in knowledge document")

    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError("Chunking failed; no chunks were produced")

    document_id = document_id or str(uuid4())
    document_title = document_title or path.name
    file_hash = _sha256_file(file_path)

    ids: List[str] = []
    final_chunks: List[Document] = []

    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{document_id}:chunk:{idx}"
        metadata = _prepare_chunk_metadata(
            chunk=chunk,
            chunk_id=chunk_id,
            document_id=document_id,
            document_title=document_title,
            audience=audience,
            category=category,
            version=version,
            uploaded_by=uploaded_by,
            source_priority=source_priority,
            is_official=is_official,
            parent_doc_id=parent_doc_id,
            effective_date=effective_date,
            status=status,
            total_chunks=total_chunks,
        )

        metadata["file_hash"] = file_hash
        metadata["source_path"] = str(path)

        metadata = _sanitize_metadata(metadata)

        final_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )
        ids.append(chunk_id)

    store = get_vector_store(audience)
    replaced_chunks = _delete_existing_document_chunks(store, document_id)
    store.add_documents(documents=final_chunks, ids=ids)

    return {
        "ok": True,
        "document_id": document_id,
        "document_title": document_title,
        "audience": audience,
        "category": category,
        "version": version,
        "status": status,
        "chunk_count": len(final_chunks),
        "replaced_chunk_count": replaced_chunks,
        "file_hash": file_hash,
        "collection": get_vector_store(audience)._collection.name,
    }
