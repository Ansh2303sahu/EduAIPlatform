from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import cast

from pydantic import ConfigDict, Field

try:
    _pydantic_settings = import_module("pydantic_settings")
    BaseSettings = _pydantic_settings.BaseSettings
    SettingsConfigDict = _pydantic_settings.SettingsConfigDict
except ModuleNotFoundError:
    from pydantic import BaseModel

    class BaseSettings(BaseModel):
        pass

    def SettingsConfigDict(**kwargs):  # type: ignore[override]
        return dict(**kwargs)


_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILES = tuple(
    str(path)
    for path in (_BACKEND_ROOT / ".env",)
    if path.exists()
)


class Settings(BaseSettings):
    model_config: ConfigDict = cast(
        ConfigDict,
        SettingsConfigDict(
            case_sensitive=False,
            env_file=_ENV_FILES,
            env_file_encoding="utf-8",
            extra="ignore",
        ),
    )

    env: str = Field(default="development", alias="ENV")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")

    # CORS
    allowed_origins_raw: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="ALLOWED_ORIGINS",
    )

    @property
    def allowed_origins(self) -> list[str]:
        return [o.strip() for o in (self.allowed_origins_raw or "").split(",") if o.strip()]

    # Supabase
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_anon_key: str = Field(default="", alias="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(default="", alias="SUPABASE_SERVICE_ROLE_KEY")

    # Buckets
    assignments_bucket: str = Field(default="assignments", alias="ASSIGNMENTS_BUCKET")
    uploads_bucket: str = Field(default="edu-uploads", alias="UPLOADS_BUCKET")

    # Limits
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")
    signed_url_expires_seconds: int = Field(default=60, alias="SIGNED_URL_EXPIRES_SECONDS")
    max_audio_bytes: int = Field(default=30_000_000, alias="MAX_AUDIO_BYTES")
    max_video_bytes: int = Field(default=50_000_000, alias="MAX_VIDEO_BYTES")

    # Parser
    parser_url: str = Field(default="", alias="PARSER_URL")
    parser_secret: str = Field(default="", alias="PARSER_SECRET")

    # ClamAV
    clamd_host: str = Field(default="clamav", alias="CLAMD_HOST")
    clamd_port: int = Field(default=3310, alias="CLAMD_PORT")

    # AI service
    ai_service_url: str = Field(default="http://ai-service:8000", alias="AI_SERVICE_URL")
    ai_service_secret: str = Field(default="", alias="AI_SERVICE_SECRET")

    # LLM service
    llm_service_url: str = Field(default="http://llm-service:8030", alias="LLM_SERVICE_URL")
    llm_service_secret: str = Field(default="", alias="LLM_SERVICE_SECRET")
    # Display labels stored in model_versions — set to match LLM_PROVIDER choice
    llm_primary_label: str = Field(default="mistral:latest", alias="LLM_PRIMARY_LABEL")
    llm_fallback_label: str = Field(default="gemma3:latest", alias="LLM_FALLBACK_LABEL")

    # -------------------------
    # Phase 9 — RAG core
    # -------------------------
    rag_enabled: bool = Field(default=True, alias="RAG_ENABLED")
    rag_provider: str = Field(default="langchain", alias="RAG_PROVIDER")

    # Vector DB / persistence
    rag_vector_db: str = Field(default="chroma", alias="RAG_VECTOR_DB")
    rag_persist_dir: str = Field(default="./storage/rag", alias="RAG_PERSIST_DIR")
    rag_student_collection: str = Field(
        default="student_knowledge_base",
        alias="RAG_STUDENT_COLLECTION",
    )
    rag_professor_collection: str = Field(
        default="professor_knowledge_base",
        alias="RAG_PROFESSOR_COLLECTION",
    )

    # Embeddings
    rag_embedding_provider: str = Field(
        default="sentence_transformers",
        alias="RAG_EMBEDDING_PROVIDER",
    )
    rag_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="RAG_EMBEDDING_MODEL",
    )

    # Retrieval tuning
    rag_top_k_initial: int = Field(default=8, alias="RAG_TOP_K_INITIAL")
    rag_top_k_final: int = Field(default=4, alias="RAG_TOP_K_FINAL")
    rag_enable_hybrid: bool = Field(default=True, alias="RAG_ENABLE_HYBRID")
    rag_semantic_weight: float = Field(default=0.65, alias="RAG_SEMANTIC_WEIGHT")
    rag_keyword_weight: float = Field(default=0.35, alias="RAG_KEYWORD_WEIGHT")
    rag_enable_rerank: bool = Field(default=True, alias="RAG_ENABLE_RERANK")
    rag_mmr_lambda: float = Field(default=0.55, alias="RAG_MMR_LAMBDA")
    rag_mmr_fetch_k_min: int = Field(default=12, alias="RAG_MMR_FETCH_K_MIN")
    rag_mmr_fetch_k_max: int = Field(default=24, alias="RAG_MMR_FETCH_K_MAX")
    rag_enable_contextual_compression: bool = Field(
        default=True,
        alias="RAG_ENABLE_CONTEXTUAL_COMPRESSION",
    )
    rag_enable_chunk_neighbors: bool = Field(
        default=True,
        alias="RAG_ENABLE_CHUNK_NEIGHBORS",
    )
    rag_neighbor_window: int = Field(default=1, alias="RAG_NEIGHBOR_WINDOW")

    # Chunking
    rag_chunk_size: int = Field(default=900, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=150, alias="RAG_CHUNK_OVERLAP")

    # Query optimization
    rag_enable_query_rewrite: bool = Field(default=True, alias="RAG_ENABLE_QUERY_REWRITE")
    rag_enable_multi_query: bool = Field(default=True, alias="RAG_ENABLE_MULTI_QUERY")
    rag_enable_topic_detection: bool = Field(default=True, alias="RAG_ENABLE_TOPIC_DETECTION")
    rag_enable_ml_guided_retrieval: bool = Field(
        default=True,
        alias="RAG_ENABLE_ML_GUIDED_RETRIEVAL",
    )
    rag_enable_adaptive_depth: bool = Field(default=True, alias="RAG_ENABLE_ADAPTIVE_DEPTH")

    # Token / prompt budget
    rag_context_max_chunks: int = Field(default=4, alias="RAG_CONTEXT_MAX_CHUNKS")
    rag_context_max_tokens: int = Field(default=2500, alias="RAG_CONTEXT_MAX_TOKENS")
    rag_inject_confidence_note: bool = Field(default=True, alias="RAG_INJECT_CONFIDENCE_NOTE")
    rag_query_excerpt_chars: int = Field(default=650, alias="RAG_QUERY_EXCERPT_CHARS")
    rag_query_title_chars: int = Field(default=160, alias="RAG_QUERY_TITLE_CHARS")
    rag_dynamic_keyword_limit: int = Field(default=12, alias="RAG_DYNAMIC_KEYWORD_LIMIT")

    # Confidence / grounding
    rag_min_confidence: float = Field(default=0.45, alias="RAG_MIN_CONFIDENCE")
    rag_low_confidence_threshold: float = Field(
        default=0.35,
        alias="RAG_LOW_CONFIDENCE_THRESHOLD",
    )
    rag_enable_grounding_guard: bool = Field(default=True, alias="RAG_ENABLE_GROUNDING_GUARD")
    rag_enable_unsupported_claim_detector: bool = Field(
        default=True,
        alias="RAG_ENABLE_UNSUPPORTED_CLAIM_DETECTOR",
    )
    rag_rerank_preferred_category_bonus: float = Field(
        default=0.09,
        alias="RAG_RERANK_PREFERRED_CATEGORY_BONUS",
    )
    rag_rerank_off_topic_penalty: float = Field(
        default=0.03,
        alias="RAG_RERANK_OFF_TOPIC_PENALTY",
    )
    rag_rerank_professor_policy_bonus: float = Field(
        default=0.05,
        alias="RAG_RERANK_PROFESSOR_POLICY_BONUS",
    )
    rag_rerank_official_bonus: float = Field(
        default=0.04,
        alias="RAG_RERANK_OFFICIAL_BONUS",
    )

    # Caching
    rag_cache_enabled: bool = Field(default=True, alias="RAG_CACHE_ENABLED")
    rag_cache_ttl_seconds: int = Field(default=1800, alias="RAG_CACHE_TTL_SECONDS")
    rag_cache_namespace: str = Field(default="phase9_rag", alias="RAG_CACHE_NAMESPACE")

    # Trace / analytics
    rag_enable_trace_logging: bool = Field(default=True, alias="RAG_ENABLE_TRACE_LOGGING")
    rag_enable_analytics: bool = Field(default=True, alias="RAG_ENABLE_ANALYTICS")

    # Admin / integrity
    rag_admin_only_ingestion: bool = Field(default=True, alias="RAG_ADMIN_ONLY_INGESTION")
    rag_require_official_for_professor: bool = Field(
        default=False,
        alias="RAG_REQUIRE_OFFICIAL_FOR_PROFESSOR",
    )
    rag_enable_integrity_hashing: bool = Field(
        default=True,
        alias="RAG_ENABLE_INTEGRITY_HASHING",
    )

    @property
    def rag_persist_path(self) -> Path:
        return Path(self.rag_persist_dir).resolve()


settings = Settings()
