"""Environment-backed service configuration."""

from ipaddress import IPv4Address
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, IPvAnyAddress, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable settings for the localhost System-One service."""

    model_config = SettingsConfigDict(
        env_prefix="SYSTEMONE_",
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    r9v_base_url: AnyHttpUrl = "http://127.0.0.1:8000"
    r9v_model: str = Field("qwen3.8-flash-next", min_length=1)
    timeout_seconds: float = Field(300.0, gt=0)
    max_choices: int = Field(16, ge=2, le=64)
    api_key: SecretStr | None = None
    concurrency: int = Field(1, ge=1, le=8)
    bind_host: IPvAnyAddress = IPv4Address("127.0.0.1")
    bind_port: int = Field(8101, ge=1024, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    calibration_path: Path | None = None

    @field_validator("bind_host")
    @classmethod
    def require_loopback_bind(cls, value: IPvAnyAddress) -> IPvAnyAddress:
        if not value.is_loopback:
            raise ValueError("bind_host must be a loopback address")
        return value
