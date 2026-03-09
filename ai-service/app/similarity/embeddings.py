from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def embed_similarity(text_a: str, text_b: str) -> float:
    m = get_model()
    emb = m.encode([text_a, text_b], normalize_embeddings=True)
    # cosine similarity since normalized
    return float(np.dot(emb[0], emb[1]))

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Return normalized embeddings for a list of texts.
    Shape: (len(texts), 384)
    """
    m = get_model()
    return m.encode(texts, normalize_embeddings=True)

def embed_text(text: str) -> np.ndarray:
    """
    Return normalized embedding for a single text.
    Shape: (384,)
    """
    return embed_texts([text])[0]