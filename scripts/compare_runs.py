"""Task 3.5: compare runs on validation metrics only and pick the best."""
import _bootstrap  # noqa: F401
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from visionguard.config import OUTPUT_DIR, PROJECT_NAME  # noqa: E402
from visionguard.tracking import setup_tracking  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "local"], default="auto")
    parser.add_argument("--prefix", default="E", help="Compare runs whose name starts with this")
    args = parser.parse_args()
    setup_tracking(backend=args.backend)
    runs = mlflow.search_runs(experiment_names=[PROJECT_NAME])
    named = runs["tags.mlflow.runName"].str.startswith(args.prefix)
    # Autologging creates child runs per stage; only top-level runs are compared.
    top_level = runs["tags.mlflow.parentRunId"].isna() if "tags.mlflow.parentRunId" in runs else True
    runs = runs[named & top_level]
    columns = {
        "tags.mlflow.runName": "run",
        "params.model": "model", "params.image_size": "image_size", "params.lr": "lr",
        "params.augment": "augment", "params.unfreeze": "unfreeze",
        "metrics.final_val_roc_auc": "val_auc", "metrics.final_val_recall": "val_recall",
        "metrics.final_val_false_alarm_rate": "val_false_alarm", "metrics.final_val_f1": "val_f1",
        "metrics.best_val_loss": "val_loss", "metrics.best_epoch": "best_epoch",
        "metrics.decision_threshold": "threshold", "run_id": "run_id",
    }
    table = runs[list(columns)].rename(columns=columns)
    # Priority: catch fakes at the capped false-alarm rate, then break ties on AUC and loss.
    table = table.sort_values(["val_recall", "val_auc", "val_loss"], ascending=[False, False, True])
    out = OUTPUT_DIR / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "experiment_comparison.csv", index=False)
    print(table.drop(columns="run_id").round(4).to_string(index=False))
    print(f"\nBest by validation: {table.iloc[0]['run']} ({table.iloc[0]['run_id']})")

    ordered = table.sort_values("run")
    figure, axis = plt.subplots(figsize=(9, 3.8))
    positions = range(len(ordered))
    axis.bar([p - 0.2 for p in positions], ordered["val_auc"], width=0.4, label="validation AUC")
    axis.bar([p + 0.2 for p in positions], ordered["val_recall"], width=0.4, label="AI recall at capped false alarms")
    axis.set_xticks(list(positions), ordered["run"], rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(out / "experiment_comparison.png", dpi=140)


if __name__ == "__main__":
    main()
