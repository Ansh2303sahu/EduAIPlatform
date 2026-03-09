from __future__ import annotations

from importlib import import_module
from pydantic import Field

try:
    _pydantic_settings = import_module("pydantic_settings")
    BaseSettings = _pydantic_settings.BaseSettings
    SettingsConfigDict = _pydantic_settings.SettingsConfigDict
except ModuleNotFoundError:
    # Editor/runtime fallback when pydantic-settings is missing.
    # Install pydantic-settings for full env-based settings behavior.
    from pydantic import BaseModel

    class BaseSettings(BaseModel):
        pass

    def SettingsConfigDict(**kwargs):  # type: ignore[override]
        return dict(**kwargs)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)

    env: str = Field(default="development", alias="ENV")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")

    # ✅ Keep env as a string, then split into list[str]
    allowed_origins_raw: str = Field(default="http://localhost:3000", alias="ALLOWED_ORIGINS")

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

    # Parser
    parser_url: str = Field(default="", alias="PARSER_URL")
    parser_secret: str = Field(default="", alias="PARSER_SECRET")

    # ClamAV
    clamd_host: str = Field(default="clamav", alias="CLAMD_HOST")
    clamd_port: int = Field(default=3310, alias="CLAMD_PORT")

    # ✅ AI Service (Phase 5.5)
    ai_service_url: str = Field(default="http://ai-service:8000", alias="AI_SERVICE_URL")
    ai_service_secret: str = Field(default="", alias="AI_SERVICE_SECRET")

    # ✅ Audio limits (Phase 5.5)
    max_audio_bytes: int = Field(default=30_000_000, alias="MAX_AUDIO_BYTES")  # 30MB

   # ✅ LLM Service (Phase 7)
    llm_service_url: str = Field(default="http://llm-service:8030", alias="LLM_SERVICE_URL")
    llm_service_secret: str = Field(default="", alias="LLM_SERVICE_SECRET")

settings = Settings()
