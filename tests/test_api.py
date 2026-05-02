"""
API integration tests for the FastAPI service.
The model bundle is mocked so no real weights are downloaded during CI.
"""
from __future__ import annotations

import pickle
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import MultiLabelBinarizer

from api.main import app

# ── Shared mock fixtures ──────────────────────────────────────────────────────

NUM_LABELS = 28

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral",
]


def _build_mlb() -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer()
    mlb.fit([EMOTION_LABELS])
    return mlb


def _make_fake_bundle():
    """Return a (model, tokenizer, mlb, device) tuple with mocked model."""
    probs = [0.05] * NUM_LABELS
    probs[17] = 0.90  # joy
    probs[7] = 0.40   # curiosity

    logits = torch.tensor([probs], dtype=torch.float32)
    mock_output = MagicMock()
    mock_output.logits = logits

    mock_model = MagicMock()
    mock_model.return_value = mock_output

    mock_tokenizer = MagicMock()
    inputs_mock = MagicMock()
    inputs_mock.__getitem__ = MagicMock(return_value=torch.zeros(1, 10))
    inputs_mock.to = MagicMock(return_value=inputs_mock)
    mock_tokenizer.return_value = inputs_mock

    mlb = _build_mlb()
    device = torch.device("cpu")
    return mock_model, mock_tokenizer, mlb, device


@pytest.fixture(autouse=True)
def patch_model_bundle():
    """Patch get_model_bundle for every test in this module."""
    with patch("api.main.get_model_bundle", return_value=_make_fake_bundle()):
        with patch("api.dependencies.get_model_bundle", return_value=_make_fake_bundle()):
            yield


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Health endpoint ───────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "environment" in data
        assert "version" in data

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


# ── Predict endpoint ──────────────────────────────────────────────────────────


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json={"text": "I feel happy today"})
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json={"text": "I feel happy today"}).json()
        assert "dominant_emotion" in data
        assert "active_emotions" in data
        assert "top_emotions" in data
        assert "processing_time_ms" in data
        assert "model_version" in data
        assert "text" in data

    def test_predict_dominant_emotion_has_label_and_confidence(self, client):
        data = client.post("/predict", json={"text": "test"}).json()
        dominant = data["dominant_emotion"]
        assert "label" in dominant
        assert "confidence" in dominant
        assert 0.0 <= dominant["confidence"] <= 1.0

    def test_predict_empty_text_returns_422(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_whitespace_text_returns_422(self, client):
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422

    def test_predict_missing_text_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_custom_threshold(self, client):
        data = client.post(
            "/predict", json={"text": "I feel great", "threshold": 0.5}
        ).json()
        for e in data["active_emotions"]:
            assert e["confidence"] >= 0.5

    def test_predict_custom_top_k(self, client):
        data = client.post(
            "/predict", json={"text": "I feel great", "top_k": 5}
        ).json()
        assert len(data["top_emotions"]) == 5

    def test_predict_text_echoed_in_response(self, client):
        text = "I feel happy today"
        data = client.post("/predict", json={"text": text}).json()
        assert data["text"] == text

    def test_predict_processing_time_is_positive(self, client):
        data = client.post("/predict", json={"text": "test"}).json()
        assert data["processing_time_ms"] >= 0


# ── Batch predict endpoint ────────────────────────────────────────────────────


class TestBatchPredictEndpoint:
    def test_batch_returns_200(self, client):
        payload = [{"text": "I am happy"}, {"text": "I am sad"}]
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_returns_correct_count(self, client):
        payload = [{"text": f"text {i}"} for i in range(5)]
        data = client.post("/predict/batch", json=payload).json()
        assert len(data) == 5

    def test_batch_exceeding_limit_returns_422(self, client):
        payload = [{"text": "test"} for _ in range(33)]
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422


# ── Docs ──────────────────────────────────────────────────────────────────────


class TestDocs:
    def test_openapi_schema_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]
