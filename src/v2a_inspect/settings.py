from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Did not include model name here because it is dynamic
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "API_KEY"),
    )
    openrouter_api_key: SecretStr | None = None
    auth_mode: Literal["disabled", "password"] = "password"
    auth_allow_self_signup: bool = True
    auth_cookie_key: SecretStr | None = None
    auth_cookie_name: str = "v2a_inspect_cookie"
    auth_cookie_expiry_days: int = Field(default=1, ge=1)
    auth_credentials_path: Path | None = None
    ui_analysis_concurrency_limit: int = Field(default=2, ge=1)
    ui_analysis_acquire_timeout_seconds: int = Field(default=120, ge=1)
    ui_temp_cleanup_max_age_seconds: int = Field(default=3600, ge=1)
    ui_cleanup_interval_seconds: int = Field(default=1800, ge=1)

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.secure"),
        env_file_encoding="utf-8",
    )

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        if self.gemini_api_key is None and self.openrouter_api_key is None:
            raise ValueError(
                "At least one of GEMINI_API_KEY or OPENROUTER_API_KEY must be set."
            )
        return self


settings = Settings()
