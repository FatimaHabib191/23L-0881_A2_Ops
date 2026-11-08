"""Task 3.6: register a run's model as a new version, optionally aliasing it champion."""
import _bootstrap  # noqa: F401
import argparse

import mlflow
from mlflow import MlflowClient

from visionguard.config import REGISTERED_MODEL, RUNS_DIR
from visionguard.tracking import setup_tracking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--champion", action="store_true")
    parser.add_argument("--backend", choices=["auto", "local"], default="auto")
    args = parser.parse_args()
    setup_tracking(backend=args.backend)
    run_dir = RUNS_DIR / args.run_name
    uri = (run_dir / "logged_model_uri.txt").read_text().strip()
    run_id = (run_dir / "mlflow_run_id.txt").read_text().strip()
    version = mlflow.register_model(uri, REGISTERED_MODEL, tags={"source_run": args.run_name, "source_run_id": run_id})
    print(f"Registered {REGISTERED_MODEL} version {version.version} from {args.run_name}")
    if args.champion:
        MlflowClient().set_registered_model_alias(REGISTERED_MODEL, "champion", version.version)
        print(f"Alias champion -> version {version.version}")


if __name__ == "__main__":
    main()
