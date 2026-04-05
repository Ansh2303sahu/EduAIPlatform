"""
Internal Pydantic models for Phase 10 pipelines.

These models carry state through pipeline stages and are NOT exposed in API
responses.  The public request/response contracts live in ``schemas.py``.

Model hierarchy
---------------
PipelineContext          — mutable carrier object threaded through all stages
  ├─ IngestionBundle     — text/OCR/audio/tables from Phase 6 parsing
  ├─ RagContext          — packaged RAG retrieval result (Phase 9)
  ├─ MLContextResult     — Phase 6 ML signals + human-readable context text
  ├─ ExecutionMetadata   — timing, provider, versions
  └─ ValidationResult    — Pydantic validation outcome per stage

Standalone result models
  ├─ RetrievalPackageResult — full output of the RAG packager service
  ├─ GenerationResult       — raw LLM output + model label
  └─ StudentMLSignals / ProfessorMLSignals — typed ML signal wrappers
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.langchain.enums import (
    AnalysisType,
    ChainRole,
    DecisionSource,
    ExecutionMode,
    LLMProvider,
    PipelineMode,
    PipelineRole,
    PipelineStage,
)


# ---------------------------------------------------------------------------
# Input bundles
# ---------------------------------------------------------------------------

class IngestionBundle(BaseModel):
    """Text extracted from a submission file — mirrors llm-service IngestionBundle."""

    text_content: str = ""
    ocr_text: str = ""
    audio_transcript: str = ""
    tables_json: Optional[Dict[str, Any]] = None


class StudentMLSignals(BaseModel):
    """
    ML signals produced by the Phase 6 ai-service for a student submission.

    All fields have safe defaults so that partial AI-service responses do not
    crash the pipeline.
    """

    feedback_category: str = "unclassified"
    quality_band: str = "med"        # "low" | "med" | "high"
    confidence_0_to_4: int = Field(default=2, ge=0, le=4)


class ProfessorMLSignals(BaseModel):
    """
    ML signals produced by the Phase 6 ai-service for a professor submission.
    """

    rubric_band: str = "adequate"
    argument_depth: str = "med"          # "low" | "med" | "high"
    moderation_consistency: str = "med"  # "low" | "med" | "high"


# ---------------------------------------------------------------------------
# RAG context
# ---------------------------------------------------------------------------

class RagContext(BaseModel):
    """Packaged RAG retrieval result ready for prompt injection (Phase 9 output)."""

    enabled: bool = False
    context: str = ""
    context_text: str = ""
    instruction: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    chunk_count: int = 0
    weak_retrieval: bool = False
    confidence_score: float = 0.0
    confidence_label: str = "low"
    safe_review: bool = False
    trace: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Intermediate result models
# ---------------------------------------------------------------------------

class MLContextResult(BaseModel):
    """
    Enriched ML context produced by ``ml_context_builder``.

    Carries both the raw ML dict and the human-readable text injected into
    the prompt, so both can be stored for audit purposes.
    """

    raw: Dict[str, Any] = Field(default_factory=dict)
    normalized: Dict[str, Any] = Field(default_factory=dict)
    context_text: str = ""
    decision_source: DecisionSource = DecisionSource.ML
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    predicted_label: str = ""
    predicted_class: str = ""
    predicted_band: str = ""
    modality_evidence_summary: List[str] = Field(default_factory=list)
    model_metadata: Dict[str, Any] = Field(default_factory=dict)
    disagreement_markers: List[str] = Field(default_factory=list)

    @classmethod
    def from_student_signals(cls, ml: Dict[str, Any], context_text: str) -> "MLContextResult":
        conf_raw = ml.get("confidence_0_to_4", 2)
        try:
            conf_0_to_4 = int(conf_raw)
        except (TypeError, ValueError):
            conf_0_to_4 = 2
        feedback_category = str(ml.get("feedback_category") or "unclassified")
        quality_band = str(ml.get("quality_band") or "med")
        return cls(
            raw=ml,
            normalized={
                "role": "student",
                "feedback_category": feedback_category,
                "quality_band": quality_band,
                "confidence_0_to_4": conf_0_to_4,
            },
            context_text=context_text,
            decision_source=DecisionSource.HYBRID,
            confidence_score=round(conf_0_to_4 / 4.0, 3),
            predicted_label=feedback_category,
            predicted_class=feedback_category,
            predicted_band=quality_band,
        )

    @classmethod
    def from_professor_signals(cls, ml: Dict[str, Any], context_text: str) -> "MLContextResult":
        consistency = str(ml.get("moderation_consistency", "med")).lower()
        conf = {"high": 0.85, "med": 0.6, "low": 0.3}.get(consistency, 0.6)
        rubric_band = str(ml.get("rubric_band") or "adequate")
        return cls(
            raw=ml,
            normalized={
                "role": "professor",
                "rubric_band": rubric_band,
                "argument_depth": str(ml.get("argument_depth") or "med"),
                "moderation_consistency": consistency,
            },
            context_text=context_text,
            decision_source=DecisionSource.HYBRID,
            confidence_score=conf,
            predicted_label=rubric_band,
            predicted_class=rubric_band,
            predicted_band=rubric_band,
        )


class ValidationResult(BaseModel):
    """
    Outcome of a Pydantic schema validation attempt.

    ``valid=True`` means the data passed schema validation cleanly.
    ``valid=False`` means a fallback or repair was used; check ``errors``.
    """

    valid: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    stage: PipelineStage = PipelineStage.VALIDATE
    repaired: bool = False

    @classmethod
    def ok(cls, *, repaired: bool = False) -> "ValidationResult":
        return cls(valid=True, repaired=repaired)

    @classmethod
    def fail(cls, errors: List[str], *, repaired: bool = False) -> "ValidationResult":
        return cls(valid=False, errors=errors, repaired=repaired)


class RetrievalPackageResult(BaseModel):
    """
    Full output of the ``retrieval_packager`` service.

    Contains both the enriched LLM payload (with grounding fields injected)
    and the typed ``RagContext`` for prompt building.
    """

    enriched_payload: Dict[str, Any] = Field(default_factory=dict)
    rag_context: RagContext = Field(default_factory=RagContext)
    retrieval_skipped: bool = False
    skipped_reason: str = ""


class ExecutionMetadata(BaseModel):
    """
    Timing, provider, and versioning metadata for a single pipeline run.

    Stored in ``model_versions`` in the ``ai_reports`` Supabase table so that
    every report is traceable back to its chain version and model.
    """

    request_id: str = ""
    pipeline: str = "phase10_langchain"

    # Provider
    provider: str = LLMProvider.OLLAMA.value
    primary_model: str = ""
    fallback_model: str = ""
    model_used: str = ""
    fallback_used: bool = False

    # Execution
    execution_mode: ExecutionMode = ExecutionMode.NORMAL
    decision_source: DecisionSource = DecisionSource.LLM
    role: ChainRole = ChainRole.STUDENT

    # Versioning
    chain_version: str = "1.0"
    chain_name: str = ""
    student_prompt_version: str = "v1"
    professor_prompt_version: str = "v1"
    schema_version: str = "1.0"
    analysis_type: str = ""
    submission_kind: str = ""

    # Timing (ms)
    timings_ms: Dict[str, int] = Field(default_factory=dict)

    # Agreement score (0.0–1.0)
    agreement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_debug: Dict[str, Any] = Field(default_factory=dict)
    prompt_debug: Dict[str, Any] = Field(default_factory=dict)

    def to_model_versions_dict(self) -> Dict[str, Any]:
        """
        Serialise to the ``model_versions`` dict shape expected by
        ``phase7.py``-style Supabase inserts so Phase 10 rows are readable
        by the same frontend code.
        """
        return {
            "request_id": self.request_id,
            "pipeline": self.pipeline,
            "llm_primary": self.primary_model,
            "llm_fallback": self.fallback_model,
            "llm_model_used": self.model_used or "unknown",
            "timings_ms": self.timings_ms,
            "agreement": {
                "final_confidence": self.agreement_score,
            },
            "chain_name": self.chain_name,
            "chain_version": self.chain_version,
            "schema_version": self.schema_version,
            "analysis_type": self.analysis_type,
            "submission_kind": self.submission_kind,
            "retrieval_debug": self.retrieval_debug,
            "prompt_debug": self.prompt_debug,
        }


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------

class GenerationResult(BaseModel):
    """Raw output from a single LangChain chain invocation."""

    raw_text: str
    model_used: str
    provider: str
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Pipeline context — mutable carrier object
# ---------------------------------------------------------------------------

class PipelineContext(BaseModel):
    """
    Mutable carrier threaded through every stage of a pipeline run.

    Each service reads from and writes into this context so stages remain
    loosely coupled and independently testable.  All fields have safe defaults
    so that a partially-initialised context never raises AttributeError.
    """

    model_config = {"arbitrary_types_allowed": True}

    # ── Identity ──────────────────────────────────────────────────────────────
    request_id: str
    file_id: str
    submission_id: str = ""
    role: PipelineRole

    # ── Derived metadata ──────────────────────────────────────────────────────
    analysis_type: AnalysisType
    mode: PipelineMode = PipelineMode.NORMAL
    execution_mode: ExecutionMode = ExecutionMode.NORMAL
    submission_kind: str = "academic"       # "project" | "academic"

    # ── Input data ────────────────────────────────────────────────────────────
    ingestion: IngestionBundle = Field(default_factory=IngestionBundle)

    # ── Safety ────────────────────────────────────────────────────────────────
    injection_detected: bool = False
    injection_reason: str = ""

    # ── ML ────────────────────────────────────────────────────────────────────
    ml_raw: Dict[str, Any] = Field(default_factory=dict)
    ml_context_text: str = ""           # human-readable text injected into prompt
    ml_result: Optional[MLContextResult] = None

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag: RagContext = Field(default_factory=RagContext)

    # ── Generation ────────────────────────────────────────────────────────────
    prompt_text: str = ""
    prompt_meta: Dict[str, Any] = Field(default_factory=dict)
    raw_llm_output: str = ""
    model_used: str = ""
    llm_ok: bool = False
    fallback_used: bool = False

    # ── Validation ────────────────────────────────────────────────────────────
    validation_result: ValidationResult = Field(default_factory=ValidationResult)

    # ── Final outputs ─────────────────────────────────────────────────────────
    report: Dict[str, Any] = Field(default_factory=dict)
    rag_meta: Dict[str, Any] = Field(default_factory=dict)

    # ── Execution metadata ────────────────────────────────────────────────────
    execution_meta: ExecutionMetadata = Field(default_factory=ExecutionMetadata)

    # ── Timing (ms) ───────────────────────────────────────────────────────────
    timings_ms: Dict[str, int] = Field(default_factory=dict)
