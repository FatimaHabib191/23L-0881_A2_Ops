"""VisionGuard model API for Vercel, using a TensorFlow Lite export of the champion."""
from __future__ import annotations

import hmac
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from validation import InMemoryRequest, UploadRejected, validate_upload

MODEL_FILE = Path(__file__).resolve().parent / "model.tflite"
API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_VERSION = os.getenv("MODEL_VERSION", "VisionGuard:unversioned")
THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.5"))
# Vercel rejects request bodies over 4.5 MB, so uploads must stay below that.
MAX_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "4")) * 1024 * 1024)
MAX_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))

# Module scope: the model loads once per server instance, not once per request.
interpreter = Interpreter(model_path=str(MODEL_FILE))
interpreter.allocate_tensors()
INPUT = interpreter.get_input_details()[0]
OUTPUT = interpreter.get_output_details()[0]
IMAGE_SIZE = int(INPUT["shape"][1])
lock = threading.Lock()

app = Flask(__name__)
app.request_class = InMemoryRequest
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 64 * 1024


def preprocess(data: bytes) -> np.ndarray:
    """Match training: RGB, bilinear square resize, values left in [0, 255]."""
    with Image.open(io.BytesIO(data)) as image:
        resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[np.newaxis, ...]


def verdict(probability: float) -> tuple[str, float]:
    if probability >= THRESHOLD:
        return "ai_generated", 0.5 + 0.5 * (probability - THRESHOLD) / max(1.0 - THRESHOLD, 1e-9)
    return "real", 0.5 + 0.5 * (THRESHOLD - probability) / max(THRESHOLD, 1e-9)


@app.get("/")
@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=True, model_version=MODEL_VERSION, image_size=IMAGE_SIZE, runtime="tflite")


@app.post("/v1/predict")
def predict():
    if not API_KEY:
        return jsonify(error="not_configured", message="The server has no API key configured."), 503
    if not hmac.compare_digest(request.headers.get("X-API-Key", "").encode(), API_KEY.encode()):
        return jsonify(error="unauthorized", message="A valid X-API-Key header is required."), 401
    file = request.files.get("image")
    if file is None:
        return jsonify(error="missing_file", message="Send the image in a form field named image."), 400
    try:
        image = validate_upload(file.filename, file.read(MAX_BYTES + 1), MAX_BYTES, MAX_PIXELS)
    except UploadRejected as rejection:
        return jsonify(error=rejection.code, message=rejection.message), rejection.status
    started = time.perf_counter()
    batch = preprocess(image.data)
    with lock:
        interpreter.set_tensor(INPUT["index"], batch)
        interpreter.invoke()
        probability = float(interpreter.get_tensor(OUTPUT["index"])[0][0])
    label, confidence = verdict(probability)
    return jsonify(
        filename=image.filename, sha256=image.sha256, label=label, confidence=confidence,
        probability_ai=probability, threshold=THRESHOLD, model_version=MODEL_VERSION,
        backend="vercel-tflite", latency_ms=int((time.perf_counter() - started) * 1000),
    )


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(error="too_large", message=f"The limit is {MAX_BYTES // (1024 * 1024)} MB."), 413


@app.errorhandler(HTTPException)
def http_error(error):
    return jsonify(error=error.name.lower().replace(" ", "_"), message=error.description), error.code
