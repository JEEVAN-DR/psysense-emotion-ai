"""
Unit tests for the core inference logic in inference.py.
These tests use mocked models to avoid loading real weights during CI.
"""
from __future__ import annotations

import pickle
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from inference import (
    ADVICE,
    EMOJI_MAP,
    EXPLANATIONS,
    explain_emotion,
    get_emoji,
    give_advice,
    predict_emotions,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

NUM_LABELS = 28
LABEL_NAMES = list(EMOJI_MAP.keys())  # 28 known emotions


def _make_mock_model(probs: list[float]):
    """Return a mock DistilBERT model that always outputs the given probs."""
    logits = torch.tensor([probs], dtype=torch.float32)
    mock_output = MagicMock()
    mock_output.logits = logits
    mock_model = MagicMock()
    mock_model.return_value = mock_output
    return mock_model


def _make_mock_tokenizer():
    """Return a tokenizer mock whose call result supports .to(device) chaining."""
    mock_tok = MagicMock()
    inputs_mock = MagicMock()
    inputs_mock.to = MagicMock(return_value=inputs_mock)
    mock_tok.return_value = inputs_mock
    return mock_tok


# ── predict_emotions ──────────────────────────────────────────────────────────


class TestPredictEmotions:
    def test_empty_text_returns_error(self):
        model = _make_mock_model([0.5] * NUM_LABELS)
        tokenizer = _make_mock_tokenizer()
        result = predict_emotions(model, tokenizer, LABEL_NAMES, torch.device("cpu"), "")
        assert "error" in result

    def test_whitespace_only_returns_error(self):
        model = _make_mock_model([0.5] * NUM_LABELS)
        tokenizer = _make_mock_tokenizer()
        result = predict_emotions(model, tokenizer, LABEL_NAMES, torch.device("cpu"), "   ")
        assert "error" in result

    def test_valid_text_returns_expected_keys(self):
        probs = [0.1] * NUM_LABELS
        probs[0] = 0.9  # joy is at index 0 in EMOJI_MAP
        model = _make_mock_model(probs)
        tokenizer = _make_mock_tokenizer()
        result = predict_emotions(model, tokenizer, LABEL_NAMES, torch.device("cpu"), "I feel great today")
        assert "dominant_emotion" in result
        assert "active_emotions" in result
        assert "top_emotions" in result

    def test_dominant_emotion_has_highest_confidence(self):
        probs = [0.05] * NUM_LABELS
        probs[3] = 0.95  # force one clearly dominant label
        model = _make_mock_model(probs)
        tokenizer = _make_mock_tokenizer()
        result = predict_emotions(model, tokenizer, LABEL_NAMES, torch.device("cpu"), "test text")
        dominant_conf = result["dominant_emotion"]["confidence"]
        for e in result["active_emotions"]:
            assert e["confidence"] <= dominant_conf + 1e-6

    def test_active_emotions_respect_threshold(self):
        threshold = 0.30
        probs = [0.05] * NUM_LABELS
        probs[0] = 0.80
        probs[1] = 0.50
        model = _make_mock_model(probs)
        tokenizer = _make_mock_tokenizer()
        result = predict_emotions(
            model, tokenizer, LABEL_NAMES, torch.device("cpu"), "test", threshold=threshold
        )
        for e in result["active_emotions"]:
            assert e["confidence"] >= threshold

    def test_top_k_limits_results(self):
        probs = [0.5] * NUM_LABELS
        model = _make_mock_model(probs)
        tokenizer = _make_mock_tokenizer()
        result = predict_emotions(
            model, tokenizer, LABEL_NAMES, torch.device("cpu"), "test", top_k=5
        )
        assert len(result["top_emotions"]) == 5


# ── Helper function tests ─────────────────────────────────────────────────────


class TestHelpers:
    def test_explain_known_emotion(self):
        for emotion in EXPLANATIONS:
            explanation = explain_emotion(emotion)
            assert isinstance(explanation, str)
            assert len(explanation) > 0

    def test_explain_unknown_emotion_returns_fallback(self):
        result = explain_emotion("unknown_emotion_xyz")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_emoji_known(self):
        for emotion in EMOJI_MAP:
            emoji = get_emoji(emotion)
            assert isinstance(emoji, str)
            assert len(emoji) > 0

    def test_get_emoji_unknown_returns_fallback(self):
        result = get_emoji("unknown_xyz")
        assert isinstance(result, str)

    def test_give_advice_known_emotions(self):
        for emotion in ADVICE:
            advice = give_advice(emotion)
            assert isinstance(advice, str)
            assert len(advice) > 0

    def test_give_advice_unknown_returns_fallback(self):
        result = give_advice("unknown_xyz")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_all_28_emotions_covered(self):
        expected_emotions = {
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise",
            "neutral",
        }
        assert set(EMOJI_MAP.keys()) == expected_emotions
        assert set(EXPLANATIONS.keys()) == expected_emotions
        assert set(ADVICE.keys()) == expected_emotions
