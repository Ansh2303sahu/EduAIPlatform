from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformerEmbeddings:
    from app.core.config import settings
    return SentenceTransformerEmbeddings(settings.rag_embedding_model)
