"""Task 2.5: evaluate a finished run on the test set, once, at its validation threshold."""
import _bootstrap  # noqa: F401
import argparse
import json

import keras
import mlflow
import pandas as pd

from visionguard import evaluation
from visionguard.config import OUTPUT_DIR, RUNS_DIR, TrainConfig
from visionguard.data import read_split
from visionguard.tracking import setup_tracking
from visionguard.training import predict_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--backend", choices=["auto", "local"], default="auto")
    args = parser.parse_args()
    run_dir = RUNS_DIR / args.run_name
    guard = run_dir / "test_metrics.json"
    if guard.exists():
        raise SystemExit(f"{args.run_name} has already used the test set. Results: {guard}")

    setup_tracking(backend=args.backend)
    cfg = TrainConfig(**json.loads((run_dir / "config.json").read_text()))
    threshold = json.loads((run_dir / "val_result.json").read_text())["threshold"]
    model = keras.models.load_model(run_dir / "best.keras")
    labels, probabilities = predict_split(model, cfg, "test")
    metrics = evaluation.compute_metrics(labels, probabilities, threshold)
    metrics["threshold"] = threshold
    guard.write_text(json.dumps(metrics, indent=2))

    predictions = read_split("test").assign(probability_ai=probabilities)
    predictions["predicted"] = (probabilities >= threshold).astype(int)
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    evaluation.plot_confusion(labels, probabilities, threshold, run_dir / "test_confusion_matrix.png", f"Test: {args.run_name}")
    evaluation.plot_roc(labels, probabilities, run_dir / "test_roc_curve.png", f"Test ROC: {args.run_name}")

    run_id = (run_dir / "mlflow_run_id.txt").read_text().strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in metrics.items()})
        for name in ("test_confusion_matrix.png", "test_roc_curve.png", "test_metrics.json", "test_predictions.csv"):
            mlflow.log_artifact(str(run_dir / name), "test")

    table_path = OUTPUT_DIR / "test_metrics_table.csv"
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    row = pd.DataFrame([{"run": args.run_name, "model": cfg.model, **metrics}])
    pd.concat([table, row], ignore_index=True).to_csv(table_path, index=False)
    print(json.dumps({"run": args.run_name, **metrics}, indent=2))


if __name__ == "__main__":
    main()
