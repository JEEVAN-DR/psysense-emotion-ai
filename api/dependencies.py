"""
Model loading and caching dependency for the FastAPI service.

The model is loaded once at startup and reused across requests via
FastAPI's dependency injection system combined with functools.lru_cache.
"""
from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from typing import Tuple

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from config import settings

logger = logging.getLogger(__name__)

# Type alias for the model bundle returned by get_model_bundle()
ModelBundle = Tuple[
    DistilBertForSequenceClassification,  # model
    DistilBertTokenizerFast,              # tokenizer
    object,                               # MultiLabelBinarizer (mlb)
    torch.device,                         # device
]


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    """
    Load and cache the DistilBERT model, tokenizer, and label encoder.

    This is called once on the first request (or at startup via the
    FastAPI lifespan hook) and the result is reused for all subsequent
    requests.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model %s onto %s …", settings.hf_model, device)

    model = DistilBertForSequenceClassification.from_pretrained(settings.hf_model)
    tokenizer = DistilBertTokenizerFast.from_pretrained(settings.hf_model)
    model.to(device)
    model.eval()

    with open(settings.label_encoder_path, "rb") as fh:
        mlb = pickle.load(fh)

    logger.info("Model loaded successfully. Labels: %d", len(mlb.classes_))
    return model, tokenizer, mlb, device
