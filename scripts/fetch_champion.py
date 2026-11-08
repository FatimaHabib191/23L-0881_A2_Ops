"""Copy the champion to models/champion/ so serving never contacts DagsHub per request."""
import _bootstrap  # noqa: F401
import argparse
import json
import shutil
from pathlib import Path

import keras
import mlflow
from mlflow import MlflowClient

from visionguard.config import PROJECT_ROOT, REGISTERED_MODEL
from visionguard.tracking import setup_tracking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "local"], default="auto")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "champion")
    args = parser.parse_args()
    setup_tracking(backend=args.backend)
    version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, "champion")
    model = mlflow.tensorflow.load_model(f"models:/{REGISTERED_MODEL}@champion")
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "model.tmp.keras"
    keras.models.save_model(model, temporary)
    shutil.move(temporary, args.output / "model.keras")
    source = version.tags.get("source_run", "unknown")
    metadata = {"registered_model": REGISTERED_MODEL, "version": version.version, "alias": "champion", "source_run": source}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"Set MODEL_VERSION={REGISTERED_MODEL}:v{version.version} in .env")


if __name__ == "__main__":
    main()
