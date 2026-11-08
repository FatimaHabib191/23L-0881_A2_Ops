"""Cloud inference service: POST an image with an API key, receive a JSON verdict."""
from __future__ import annotations

import hmac
import logging
import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from webapp.inference import InferenceError, LocalPredictor, Predictor
from webapp.settings import Settings
from webapp.validation import InMemoryRequest, UploadRejected, validate_upload

log = logging.getLogger("visionguard.api")


def create_api(settings: Settings | None = None, predictor: Predictor | None = None, api_key: str | None = None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = settings or Settings.from_env()
    api_key = api_key if api_key is not None else os.getenv("MODEL_API_KEY", "")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY must be set so the public endpoint is protected")
    try:
        predictor = predictor or LocalPredictor(settings.model_path, settings.model_threshold, settings.model_version)
    except Exception:
        log.exception("FATAL: cannot load the model. Check MODEL_PATH.")
        raise

    app = Flask(__name__)
    app.request_class = InMemoryRequest
    limit = settings.max_upload_mb * 1024 * 1024
    app.config["MAX_CONTENT_LENGTH"] = limit + 64 * 1024

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_version=settings.model_version, **predictor.status())

    @app.post("/v1/predict")
    def predict():
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied.encode(), api_key.encode()):
            return jsonify(error="unauthorized", message="A valid X-API-Key header is required."), 401
        file = request.files.get("image")
        if file is None:
            return jsonify(error="missing_file", message="Send the image in a form field named image."), 400
        try:
            image = validate_upload(file.filename, file.read(limit + 1), limit, settings.max_pixels)
            verdict = predictor.predict(image.data, image.filename)
        except UploadRejected as rejection:
            log.info("Rejected: %s", rejection.code)
            return jsonify(error=rejection.code, message=rejection.message), rejection.status
        except InferenceError as error:
            log.error("Inference failed: %s", error)
            return jsonify(error="inference_failed", message="The model could not process this image."), 500
        log.info("%s confidence=%.3f latency_ms=%s file=%s", verdict.label, verdict.confidence, verdict.latency_ms, image.filename)
        return jsonify(filename=image.filename, sha256=image.sha256, **verdict.to_dict())

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error="too_large", message=f"The limit is {settings.max_upload_mb} MB."), 413

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.name.lower().replace(" ", "_"), message=error.description), error.code

    return app
