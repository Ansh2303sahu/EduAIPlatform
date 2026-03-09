from importlib import import_module

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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # shared secret used by backend -> ai-service calls
    ai_service_secret: str = "dev_ai_secret"
    env: str = "dev"


settings = Settings()
