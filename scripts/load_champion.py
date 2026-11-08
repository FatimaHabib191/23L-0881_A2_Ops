"""Task 3.6: load the model by alias and classify one image."""
import _bootstrap  # noqa: F401
import argparse
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from visionguard.config import REGISTERED_MODEL
from visionguard.preprocess import bytes_to_batch
from visionguard.tracking import setup_tracking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--threshold", type=float, default=float(os.getenv("MODEL_THRESHOLD", "0.5")))
    parser.add_argument("--backend", choices=["auto", "local"], default="auto")
    args = parser.parse_args()
    setup_tracking(backend=args.backend)
    version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, "champion")
    model = mlflow.tensorflow.load_model(f"models:/{REGISTERED_MODEL}@champion")
    batch = bytes_to_batch(args.image.read_bytes(), model.input_shape[1])
    probability = float(model(batch, training=False)[0][0])
    label = "ai_generated" if probability >= args.threshold else "real"
    confidence = probability if label == "ai_generated" else 1 - probability
    print(f"Loaded {REGISTERED_MODEL}@champion (version {version.version})")
    print(f"{args.image.name}: {label}, confidence {confidence:.3f}, P(AI) {probability:.3f}, threshold {args.threshold:.3f}")


if __name__ == "__main__":
    main()
