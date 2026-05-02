"""
Pydantic request / response models for the PsySense Emotion AI API.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Request ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Input text to analyze")
    threshold: float = Field(0.10, ge=0.0, le=1.0, description="Confidence threshold for active emotions")
    top_k: int = Field(10, ge=1, le=28, description="Number of top emotions to return")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


# ── Sub-models ────────────────────────────────────────────────────────────────

class EmotionScore(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# ── Response ──────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str
    dominant_emotion: EmotionScore
    active_emotions: List[EmotionScore]
    top_emotions: List[EmotionScore]
    processing_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    environment: str
    version: str


class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    total_requests: int
    total_errors: int
    average_latency_ms: float
    model_version: str
