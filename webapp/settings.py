"""All runtime configuration comes from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    inference_mode: str
    model_path: str
    model_version: str
    model_threshold: float
    database_url: str
    max_upload_mb: int
    max_pixels: int
    model_api_url: str
    model_api_key: str
    model_api_timeout: float
    secret_key: str
    history_page_size: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        mode = os.getenv("INFERENCE_MODE", "local").strip().lower()
        if mode not in {"local", "remote"}:
            raise ValueError("INFERENCE_MODE must be local or remote")
        return cls(
            inference_mode=mode,
            model_path=os.getenv("MODEL_PATH", "models/champion/model.keras"),
            model_version=os.getenv("MODEL_VERSION", "VisionGuard:unversioned"),
            model_threshold=float(os.getenv("MODEL_THRESHOLD", "0.5")),
            database_url=os.getenv("DATABASE_URL", "sqlite:///visionguard.db"),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "8")),
            max_pixels=int(os.getenv("MAX_IMAGE_PIXELS", "40000000")),
            model_api_url=os.getenv("MODEL_API_URL", "").rstrip("/"),
            model_api_key=os.getenv("MODEL_API_KEY", ""),
            model_api_timeout=float(os.getenv("MODEL_API_TIMEOUT", "30")),
            secret_key=os.getenv("SECRET_KEY", ""),
            history_page_size=int(os.getenv("HISTORY_PAGE_SIZE", "20")),
        )
