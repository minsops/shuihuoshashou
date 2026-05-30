from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    database_url: str = "sqlite:///data/shuihuo_killer.db"
    llm_provider: str = "mock"
    llm_model: str = "mimo-v2.5-pro"
    llm_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    llm_api_path: str = "/chat/completions"
    llm_api_key: str = ""
    llm_auth_header: str = "api-key"
    llm_auth_scheme: str = ""
    llm_response_content_path: str = "choices.0.message.content"
    llm_extra_body_json: str = ""
    llm_timeout_seconds: int = 30
    signal_enabled: bool = False
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: int = 120
    report_dir: Path = Field(default=Path("data/reports"))

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
