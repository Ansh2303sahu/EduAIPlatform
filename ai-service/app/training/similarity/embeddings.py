from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def embed_similarity(text_a: str, text_b: str) -> float:
    m = get_model()
    emb = m.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))