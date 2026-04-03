from __future__ import annotations

from pathlib import Path

from app.rag.ingestion.document_loader import SUPPORTED_EXTENSIONS


VALID_AUDIENCES = {'student', 'professor'}


def validate_ingestion_request(file_path: str, audience: str, category: str) -> None:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f'Knowledge file not found: {file_path}')
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f'Unsupported knowledge file extension: {path.suffix.lower()}')
    if (audience or '').strip().lower() not in VALID_AUDIENCES:
        raise ValueError("audience must be 'student' or 'professor'")
    if not (category or '').strip():
        raise ValueError('category must not be empty')
