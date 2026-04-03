from __future__ import annotations

import _bootstrap  # noqa: F401

from chromadb import PersistentClient
from app.core.config import settings


def main() -> None:
    persist_path = str(settings.rag_persist_path)
    print(f"Using Chroma persist path: {persist_path}")

    client = PersistentClient(path=persist_path)

    targets = [
        settings.rag_student_collection,
        settings.rag_professor_collection,
    ]

    existing = {c.name for c in client.list_collections()}

    for name in targets:
        if name in existing:
            client.delete_collection(name)
            print(f"Deleted collection: {name}")
        else:
            print(f"Collection not found: {name}")

    print("RAG collection reset complete.")


if __name__ == "__main__":
    main()
