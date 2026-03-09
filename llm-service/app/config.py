import os
from pydantic import BaseModel

class Settings(BaseModel):
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    primary_model: str = os.getenv("OLLAMA_PRIMARY_MODEL", "mistral")
    fallback_model: str = os.getenv("OLLAMA_FALLBACK_MODEL", "phi3")

    timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    max_input_chars: int = int(os.getenv("LLM_MAX_INPUT_CHARS", "50000"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "1"))

    service_secret: str = os.getenv("LLM_SERVICE_SECRET", "")

settings = Settings()