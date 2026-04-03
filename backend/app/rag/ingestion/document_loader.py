from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\x00", " ").split())


def _load_pdf(path: Path) -> List[Document]:
    reader = PdfReader(str(path))
    docs: List[Document] = []

    for i, page in enumerate(reader.pages):
        text = _normalize_text(page.extract_text() or "")
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": i + 1,
                    "source_path": str(path),
                },
            )
        )
    return docs


def load_documents(file_path: str) -> List[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported knowledge file extension: {ext}")

    if ext == ".pdf":
        docs = _load_pdf(path)
    elif ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    cleaned: List[Document] = []
    for doc in docs:
        text = _normalize_text(doc.page_content)
        if not text.strip():
            continue
        metadata = dict(doc.metadata or {})
        metadata["source_path"] = str(path)
        cleaned.append(Document(page_content=text, metadata=metadata))

    return cleaned