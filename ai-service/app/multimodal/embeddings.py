from __future__ import annotations

from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

@torch.no_grad()
def encode_texts(texts: List[str]) -> torch.Tensor:
    """
    Returns FloatTensor [N, 384]
    Uses CPU by default (SentenceTransformer). Fast enough for precompute.
    """
    m = get_model()
    # sentence-transformers returns numpy array
    emb = m.encode(
        texts,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    if isinstance(emb, list):
        emb = np.array(emb, dtype=np.float32)
    emb = emb.astype(np.float32, copy=False)
    return torch.from_numpy(emb)

@torch.no_grad()
def encode_text(text: str) -> torch.Tensor:
    return encode_texts([text])[0]