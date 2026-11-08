"""Inference backends: the model in this process, or the model API over HTTP."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import requests

from webapp.settings import Settings

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Verdict:
    label: str
    confidence: float
    probability_ai: float
    threshold: float
    model_version: str
    backend: str
    latency_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


def verdict_from_probability(probability: float, threshold: float) -> tuple[str, float]:
    if probability >= threshold:
        return "ai_generated", probability
    return "real", 1.0 - probability


class Predictor(Protocol):
    def predict(self, image_bytes: bytes, filename: str = "upload") -> Verdict: ...
    def status(self) -> dict: ...


class LocalPredictor:
    """Holds the model once for the life of the process."""

    def __init__(self, model_path: str, threshold: float, version: str):
        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(f"Model file not found at MODEL_PATH={model_path}")
        import keras

        started = time.perf_counter()
        self.model = keras.models.load_model(path, compile=False)
        self.image_size = int(self.model.input_shape[1])
        self.threshold, self.version = threshold, version
        self._lock = threading.Lock()
        log.info("Loaded model %s from %s in %.1fs", version, path, time.perf_counter() - started)

    def predict(self, image_bytes: bytes, filename: str = "upload") -> Verdict:
        import tensorflow as tf

        from visionguard.preprocess import bytes_to_batch

        started = time.perf_counter()
        try:
            batch = bytes_to_batch(image_bytes, self.image_size)
            with self._lock:
                probability = float(self.model(batch, training=False)[0][0])
        except tf.errors.InvalidArgumentError as error:
            raise InferenceError("The model could not decode this image") from error
        label, confidence = verdict_from_probability(probability, self.threshold)
        return Verdict(label, confidence, probability, self.threshold, self.version, "local", int((time.perf_counter() - started) * 1000))

    def status(self) -> dict:
        return {"model_loaded": True, "image_size": self.image_size}


class RemotePredictor:
    """Calls the cloud model API. The API owns the model, threshold and version."""

    def __init__(self, url: str, key: str, timeout: float):
        if not url or not key:
            raise ModelLoadError("Remote mode needs MODEL_API_URL and MODEL_API_KEY")
        self.url, self.key, self.timeout = url, key, timeout

    def predict(self, image_bytes: bytes, filename: str = "upload") -> Verdict:
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.url}/v1/predict",
                files={"image": (filename, image_bytes)},
                headers={"X-API-Key": self.key},
                timeout=(5, self.timeout),
            )
        except requests.RequestException as error:
            raise InferenceError(f"Model API unreachable: {type(error).__name__}") from error
        if response.status_code != 200:
            raise InferenceError(f"Model API returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            body = response.json()
            return Verdict(
                body["label"], float(body["confidence"]), float(body["probability_ai"]),
                float(body["threshold"]), body["model_version"], "remote",
                int((time.perf_counter() - started) * 1000),
            )
        except (ValueError, KeyError, TypeError) as error:
            raise InferenceError("Model API returned an invalid response") from error

    def status(self) -> dict:
        try:
            response = requests.get(f"{self.url}/health", timeout=(3, 5))
            return {"model_api_reachable": response.ok, "model_api": response.json() if response.ok else None}
        except (requests.RequestException, ValueError):
            return {"model_api_reachable": False}


def build_predictor(settings: Settings) -> Predictor:
    if settings.inference_mode == "remote":
        return RemotePredictor(settings.model_api_url, settings.model_api_key, settings.model_api_timeout)
    return LocalPredictor(settings.model_path, settings.model_threshold, settings.model_version)
