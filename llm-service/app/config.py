from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Local runs of llm-service should pick up the same repo env defaults as the
# backend, while still allowing shell/Docker env vars to override them.
for _env_file in (_REPO_ROOT / ".env", _SERVICE_ROOT / ".env"):
    if _env_file.exists():
        load_dotenv(_env_file, override=False)


def _looks_like_placeholder_secret(value: str) -> bool:
    token = str(value or "").strip().lower()
    if not token:
        return True
    if token in {"placeholder", "changeme", "your_key_here"}:
        return True
    if token.endswith("..."):
        return True
    return "<" in token and ">" in token


class Settings(BaseModel):
    # Provider selection: "ollama" or "anthropic"
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    # Ollama settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    primary_model: str = os.getenv("OLLAMA_PRIMARY_MODEL", "gemma3:4b")
    fallback_model: str = os.getenv("OLLAMA_FALLBACK_MODEL", "phi3:mini")
    ollama_options_json: str = os.getenv("OLLAMA_OPTIONS_JSON", "")
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    ollama_max_num_ctx: int = int(
        os.getenv("OLLAMA_MAX_NUM_CTX", os.getenv("OLLAMA_NUM_CTX", "8192"))
    )
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "1200"))
    ollama_max_num_predict: int = int(
        os.getenv("OLLAMA_MAX_NUM_PREDICT", os.getenv("OLLAMA_NUM_PREDICT", "1400"))
    )
    ollama_num_batch: int = int(os.getenv("OLLAMA_NUM_BATCH", "16"))
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.15"))
    ollama_top_p: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    ollama_top_k: int = int(os.getenv("OLLAMA_TOP_K", "40"))
    ollama_repeat_penalty: float = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.12"))
    ollama_fallback_num_ctx: int = int(os.getenv("OLLAMA_FALLBACK_NUM_CTX", "3072"))
    ollama_fallback_num_predict: int = int(os.getenv("OLLAMA_FALLBACK_NUM_PREDICT", "512"))
    ollama_fallback_num_batch: int = int(os.getenv("OLLAMA_FALLBACK_NUM_BATCH", "16"))

    # Anthropic / Claude settings
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_primary_model: str = os.getenv("ANTHROPIC_PRIMARY_MODEL", "claude-sonnet-4-6")
    anthropic_fallback_model: str = os.getenv("ANTHROPIC_FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    anthropic_max_tokens: int = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))

    timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    max_input_chars: int = int(os.getenv("LLM_MAX_INPUT_CHARS", "12000"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "1"))
    student_report_max_output_tokens: int = int(
        os.getenv("LLM_STUDENT_REPORT_MAX_OUTPUT_TOKENS", "1200")
    )
    professor_report_max_output_tokens: int = int(
        os.getenv("LLM_PROFESSOR_REPORT_MAX_OUTPUT_TOKENS", "1200")
    )
    repair_max_output_tokens: int = int(os.getenv("LLM_REPAIR_MAX_OUTPUT_TOKENS", "700"))

    service_secret: str = os.getenv("LLM_SERVICE_SECRET", "")

    @property
    def effective_provider(self) -> str:
        provider = str(self.llm_provider or "").strip().lower()
        if provider == "anthropic" and not _looks_like_placeholder_secret(self.anthropic_api_key):
            return "anthropic"
        return "ollama"

    @property
    def using_ollama(self) -> bool:
        return self.effective_provider == "ollama"


settings = Settings()


class RAGCitation(BaseModel):
    index: int
    title: str
    section: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None


class RAGMeta(BaseModel):
    enabled: bool = False
    audience: Optional[str] = None
    context: Optional[str] = None
    citations: Optional[List[RAGCitation]] = None
    confidence_score: Optional[float] = None
    confidence_label: Optional[str] = None
    safe_review: Optional[bool] = None
    trace: Optional[Dict[str, Any]] = None
    instruction: Optional[str] = None

class StudentReportRequest(BaseModel):
    text: str
    submission_type: Optional[str] = None
    ml_signals: Optional[Dict[str, Any]] = None

    rag: Optional[RAGMeta] = None
    grounding_context: Optional[str] = None
    grounding_citations: Optional[List[RAGCitation]] = None
    grounding_instruction: Optional[str] = None

class ProfessorReportRequest(BaseModel):
    text: str
    submission_type: Optional[str] = None
    ml_signals: Optional[Dict[str, Any]] = None

    rag: Optional[RAGMeta] = None
    grounding_context: Optional[str] = None
    grounding_citations: Optional[List[RAGCitation]] = None
    grounding_instruction: Optional[str] = None
