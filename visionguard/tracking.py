"""MLflow connection helpers and the manual per-epoch logger."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import keras
import mlflow
import yaml
from dotenv import load_dotenv

from visionguard.config import OUTPUT_DIR, PROJECT_NAME, PROJECT_ROOT


def setup_tracking(experiment: str = PROJECT_NAME, backend: str = "auto") -> str:
    """Connect to DagsHub when MLFLOW_TRACKING_URI is set, otherwise to a local database.

    Credentials come only from the environment or the git-ignored .env file:
    MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if backend == "local" or not uri:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{(OUTPUT_DIR / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dvc_data_version(pointer: Path = PROJECT_ROOT / "data" / "raw.dvc") -> str:
    """Use the directory hash in the DVC pointer as the dataset version."""
    if not pointer.exists():
        return "untracked"
    return yaml.safe_load(pointer.read_text())["outs"][0]["md5"]


def standard_tags(seed: int) -> dict[str, str]:
    return {
        "git_commit": git_commit(),
        "dvc_data_version": dvc_data_version(),
        "seed": str(seed),
        "project": PROJECT_NAME,
    }


class EpochMetricLogger(keras.callbacks.Callback):
    """Log every epoch's metrics, with a global step that continues across stages."""

    def __init__(self, step_offset: int = 0):
        super().__init__()
        self.step_offset = step_offset

    def on_epoch_end(self, epoch, logs=None):
        renamed = {}
        for key, value in (logs or {}).items():
            name = key if key.startswith("val_") or key == "learning_rate" else f"train_{key}"
            renamed[name] = float(value)
        mlflow.log_metrics(renamed, step=self.step_offset + epoch)
