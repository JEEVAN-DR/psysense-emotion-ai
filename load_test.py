"""
Load testing for the PsySense Emotion AI API using Locust.

Run with:
    locust -f load_test.py --host=http://localhost:8000

Or headless (CI mode):
    locust -f load_test.py --host=http://localhost:8000 \
        --headless --users 20 --spawn-rate 5 --run-time 60s \
        --csv=load_test_results
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task


SAMPLE_TEXTS = [
    "I feel extremely happy and excited about the new project!",
    "This makes me very angry and frustrated. How could they do this?",
    "I'm scared and nervous about the upcoming presentation.",
    "I love spending time with my family. It brings me so much joy.",
    "I'm disappointed that the results weren't as expected.",
    "Wow, I'm completely surprised by this unexpected news!",
    "I feel proud of what I've accomplished despite the challenges.",
    "I miss my old friends. The grief is overwhelming sometimes.",
    "I'm curious about how this new technology works.",
    "Feeling grateful for all the support I've received.",
]


class EmotionAPIUser(HttpUser):
    """Simulates a single user sending emotion classification requests."""

    wait_time = between(0.5, 2.0)  # seconds between requests per user

    @task(5)
    def predict_emotion(self):
        """Most common task — single text prediction."""
        payload = {
            "text": random.choice(SAMPLE_TEXTS),
            "threshold": 0.10,
            "top_k": 10,
        }
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Got status {response.status_code}")
            elif "dominant_emotion" not in response.json():
                response.failure("Missing dominant_emotion in response")

    @task(2)
    def predict_batch(self):
        """Batch prediction — less frequent than single."""
        payload = [
            {"text": random.choice(SAMPLE_TEXTS)}
            for _ in range(random.randint(2, 8))
        ]
        with self.client.post(
            "/predict/batch",
            json=payload,
            catch_response=True,
            name="/predict/batch",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Got status {response.status_code}")

    @task(1)
    def health_check(self):
        """Lightweight health-check probe."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")
            elif response.json().get("status") != "ok":
                response.failure("API reported degraded status")
