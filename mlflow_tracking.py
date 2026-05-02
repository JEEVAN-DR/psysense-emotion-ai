"""
MLflow model registry integration for PsySense Emotion AI.

Usage
-----
    # Log a model run and register it:
    python mlflow_tracking.py

    # Open the MLflow UI:
    mlflow ui --port 5000
"""
from __future__ import annotations

import logging
import os
import pickle
import time

import mlflow
import mlflow.pyfunc
import numpy as np
import torch

from config import settings
from inference import load_model, predict_emotions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("psysense.mlflow")


# ── Wrapper class so MLflow can serve via pyfunc ──────────────────────────────

class EmotionModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapper for the PsySense emotion classifier."""

    def load_context(self, context):
        import pickle
        import torch
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )

        model_path = context.artifacts["hf_model_path"]
        label_encoder_path = context.artifacts["label_encoder_path"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        with open(label_encoder_path, "rb") as fh:
            mlb = pickle.load(fh)
        self.label_names = list(mlb.classes_)

    def predict(self, context, model_input, params=None):  # noqa: D102
        from inference import predict_emotions

        texts = (
            model_input["text"].tolist()
            if hasattr(model_input, "tolist")
            else list(model_input["text"])
        )
        results = []
        for text in texts:
            result = predict_emotions(
                self.model,
                self.tokenizer,
                self.label_names,
                self.device,
                text,
            )
            results.append(result.get("dominant_emotion", {}).get("label", "neutral"))
        return results


# ── Tracking run ──────────────────────────────────────────────────────────────

def log_model_to_registry() -> str:
    """
    Load the current model, run a smoke-test, and log it to MLflow.
    Returns the run ID.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info("Loading model for registration …")
    model, tokenizer, mlb, device = load_model()
    label_names = list(mlb.classes_)

    sample_texts = [
        "I feel extremely happy and proud today!",
        "This makes me very angry and frustrated.",
        "I'm scared and nervous about the upcoming event.",
    ]

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run ID: %s", run_id)

        # Log parameters
        mlflow.log_params(
            {
                "hf_model": settings.hf_model,
                "threshold": settings.model_threshold,
                "top_k": settings.model_top_k,
                "num_labels": len(label_names),
            }
        )

        # Benchmark latency
        latencies: list[float] = []
        for text in sample_texts:
            t0 = time.perf_counter()
            predict_emotions(model, tokenizer, label_names, device, text)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_latency = float(np.mean(latencies))
        mlflow.log_metrics(
            {
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "num_labels": len(label_names),
            }
        )
        logger.info("Average inference latency: %.1f ms", avg_latency)

        # Log the wrapper model
        artifacts = {
            "hf_model_path": settings.hf_model,  # HuggingFace ID or local path
            "label_encoder_path": settings.label_encoder_path,
        }
        mlflow.pyfunc.log_model(
            artifact_path="emotion_model",
            python_model=EmotionModelWrapper(),
            artifacts=artifacts,
            registered_model_name=settings.mlflow_model_name,
        )

        logger.info(
            "Model registered as '%s' in MLflow.", settings.mlflow_model_name
        )

    return run_id


if __name__ == "__main__":
    run_id = log_model_to_registry()
    print(f"\n✅ Model logged. Run ID: {run_id}")
    print(f"   Open: {settings.mlflow_tracking_uri}/#/experiments")
