from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from app.rag.ingestion.pipeline import ingest_knowledge_document
from app.rag.vector_store import get_vector_store


BASE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


def build_document_id(file_path: Path, audience_root: Path, audience: str) -> str:
    relative = file_path.relative_to(audience_root)
    raw = f"{audience}:{relative.as_posix()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def infer_category(file_path: Path, audience_root: Path) -> str:
    try:
        relative = file_path.relative_to(audience_root)
        if len(relative.parts) >= 2:
            return relative.parts[0]
    except Exception:
        pass
    return "general"


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


def delete_existing_seed_chunks(
    *,
    audience: str,
    category: str,
    document_title: str,
) -> int:
    store = get_vector_store(audience)
    existing = store.get(
        where={
            "$and": [
                {"audience": audience},
                {"category": category},
                {"document_title": document_title},
                {"uploaded_by": "system_seed"},
            ]
        },
        include=[],
    )
    ids = _flatten_ids(existing.get("ids"))
    if ids:
        store.delete(ids=ids)
    return len(ids)


def seed_audience(audience: str) -> None:
    audience_root = BASE_DIR / audience

    if not audience_root.exists():
        print(f"[WARN] Skipping {audience}: folder not found -> {audience_root}")
        return

    files = sorted(audience_root.rglob("*.md"))
    if not files:
        print(f"[WARN] No markdown files found for {audience} in {audience_root}")
        return

    print(f"\n=== Seeding {audience} knowledge base ===")
    print(f"Found {len(files)} files")

    success_count = 0
    fail_count = 0

    for file_path in files:
        category = infer_category(file_path, audience_root)
        document_title = file_path.stem.replace("_", " ").replace("-", " ").title()
        document_id = build_document_id(file_path, audience_root, audience)

        try:
            deleted_chunks = delete_existing_seed_chunks(
                audience=audience,
                category=category,
                document_title=document_title,
            )
            result = ingest_knowledge_document(
                file_path=str(file_path),
                audience=audience,
                category=category,
                uploaded_by="system_seed",
                document_title=document_title,
                document_id=document_id,
                version="v1",
                source_priority=90 if audience == "professor" else 80,
                is_official=True,
                parent_doc_id=document_id,
                effective_date=None,
                status="active",
            )

            success_count += 1
            print(
                f"[OK] {audience} | {category} | {file_path.name} | "
                f"document_id={document_id} | deleted_chunks={deleted_chunks} | result={result}"
            )
        except Exception as e:
            fail_count += 1
            print(f"[FAIL] {audience} | {category} | {file_path.name} | error={e}")

    print(f"\nFinished {audience}: success={success_count}, failed={fail_count}")


def main() -> None:
    seed_audience("student")
    seed_audience("professor")
    print("\nKnowledge base seeding complete.")


if __name__ == "__main__":
    main()
