"""VisionGuard Flask application factory."""
from __future__ import annotations

import logging
import math
import secrets

from flask import Flask, abort, current_app, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from webapp.database import Database
from webapp.inference import InferenceError, Predictor, build_predictor
from webapp.settings import Settings
from webapp.validation import InMemoryRequest, UploadRejected, too_large, validate_upload

log = logging.getLogger("visionguard")
LABELS = {"ai_generated": "AI-generated", "real": "Real photo"}
STATUS_FILTERS = {"ok", "rejected", "error"}


def _database() -> Database:
    return current_app.extensions["vg_database"]


def _predictor() -> Predictor:
    return current_app.extensions["vg_predictor"]


def _settings() -> Settings:
    return current_app.extensions["vg_settings"]


def _render_rejection(rejection: UploadRejected, filename: str | None = None):
    _database().add(status="rejected", filename=filename, error_code=rejection.code)
    log.info("Rejected upload: %s", rejection.code)
    return render_template("error.html", title=rejection.title, message=rejection.message, status=rejection.status), rejection.status


def _register_routes(app: Flask) -> None:
    @app.get("/")
    def upload():
        return render_template("upload.html", max_mb=_settings().max_upload_mb, active="upload")

    @app.post("/predict")
    def predict():
        settings = _settings()
        limit = settings.max_upload_mb * 1024 * 1024
        file = request.files.get("image")
        if file is None:
            return _render_rejection(UploadRejected("missing_file", 400, "No image received", "Please choose an image before submitting."))
        try:
            image = validate_upload(file.filename, file.read(limit + 1), limit, settings.max_pixels)
        except UploadRejected as rejection:
            from webapp.validation import clean_filename

            return _render_rejection(rejection, clean_filename(file.filename))

        try:
            verdict = _predictor().predict(image.data, image.filename)
        except InferenceError as error:
            log.error("Inference failed: %s", error)
            _database().add(
                status="error", filename=image.filename, sha256=image.sha256, size_bytes=len(image.data),
                width=image.width, height=image.height, error_code="inference_failed",
            )
            return render_template(
                "error.html", title="The model is unavailable",
                message="We could not get a verdict for this image. Please try again shortly.", status=502,
            ), 502

        record = _database().add(
            status="ok", filename=image.filename, sha256=image.sha256, size_bytes=len(image.data),
            width=image.width, height=image.height, label=verdict.label, confidence=verdict.confidence,
            probability_ai=verdict.probability_ai, threshold=verdict.threshold,
            model_version=verdict.model_version, backend=verdict.backend, latency_ms=verdict.latency_ms,
        )
        log.info("Request %s: %s confidence=%.3f model=%s", record.id, verdict.label, verdict.confidence, verdict.model_version)
        return redirect(url_for("result", record_id=record.id), code=303)

    @app.get("/result/<int:record_id>")
    def result(record_id: int):
        record = _database().get(record_id)
        if record is None or record.status != "ok":
            abort(404)
        return render_template("result.html", record=record, verdict=LABELS[record.label], active="upload")

    @app.get("/history")
    def history():
        page = max(request.args.get("page", 1, type=int), 1)
        status = request.args.get("status")
        status = status if status in STATUS_FILTERS else None
        size = _settings().history_page_size
        records, total = _database().history(page, size, status)
        return render_template(
            "history.html", records=records, total=total, page=page, status=status,
            pages=max(math.ceil(total / size), 1), labels=LABELS, active="history",
        )

    @app.get("/health")
    def health():
        database_ok = _database().ping()
        body = {
            "status": "ok" if database_ok else "degraded",
            "database": "ok" if database_ok else "unreachable",
            "inference_mode": _settings().inference_mode,
            "debug": current_app.debug,
            **_predictor().status(),
        }
        return jsonify(body), 200 if database_ok else 503


def _register_errors(app: Flask) -> None:
    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error):
        return _render_rejection(too_large(_settings().max_upload_mb))

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", title="Page not found", message="That page does not exist.", status=404), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return render_template("error.html", title="Not allowed", message="That action is not available here.", status=405), 405

    @app.errorhandler(500)
    def server_error(_error):
        return render_template("error.html", title="Something went wrong", message="Please try again.", status=500), 500


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = settings or Settings.from_env()
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    app.config.update(
        SECRET_KEY=settings.secret_key or secrets.token_hex(32),
        # Multipart overhead is allowed for so that the friendly per-file check can run first.
        MAX_CONTENT_LENGTH=(settings.max_upload_mb * 1024 * 1024) + 64 * 1024,
    )
    try:
        database = Database(settings.database_url)
        database.create_schema()
    except Exception:
        log.exception("FATAL: cannot initialise the database. Check DATABASE_URL and the password.")
        raise
    try:
        # Created once here and reused by every request.
        predictor = predictor or build_predictor(settings)
    except Exception:
        log.exception("FATAL: cannot load the model backend. Check MODEL_PATH and INFERENCE_MODE.")
        raise
    app.extensions.update(vg_settings=settings, vg_database=database, vg_predictor=predictor)
    _register_routes(app)
    _register_errors(app)
    log.info("VisionGuard ready: mode=%s", settings.inference_mode)
    return app
