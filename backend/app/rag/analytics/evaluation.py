from __future__ import annotations

from typing import Iterable

from app.rag.schemas import RetrievalQuery, RetrievalResult
from app.rag.service_wrapper import run_rag


def evaluate_retrieval_cases(cases: Iterable[RetrievalQuery]) -> list[RetrievalResult]:
    return [run_rag(case) for case in cases]
