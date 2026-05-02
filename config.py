"""
Environment-based configuration management for PsySense Emotion AI.
Values can be overridden via environment variables or a .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    hf_model: str = "Hitan2004/psysense-emotion-ai"
    model_threshold: float = 0.10
    model_top_k: int = 10
    label_encoder_path: str = os.path.join(
        os.path.dirname(__file__), "model", "label_encoder.pkl"
    )

    # ── API server ─────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # ── Environment / logging ──────────────────────────────────────────────
    environment: str = "development"  # development | staging | production
    log_level: str = "INFO"

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "psysense-emotion-ai"
    mlflow_model_name: str = "psysense-distilbert"

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Convenience alias used throughout the project
settings = get_settings()
