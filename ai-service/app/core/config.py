from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    class _BaseSettings(BaseModel):
        pass
else:
    try:
        _pydantic_settings = import_module("pydantic_settings")
        _BaseSettings = _pydantic_settings.BaseSettings
    except ModuleNotFoundError:
        class _BaseSettings(BaseModel):
            pass


class Settings(_BaseSettings):
    model_config: ConfigDict = cast(
        ConfigDict,
        {
            "env_file": ".env",
            "extra": "ignore",
        },
    )

    # shared secret used by backend -> ai-service calls
    ai_service_secret: str = "dev_ai_secret"
    env: str = "dev"


settings = Settings()
