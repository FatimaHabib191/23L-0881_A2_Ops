"""Two-stage training with disciplined callbacks and MLflow logging."""
from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard import data as vg_data  # noqa: E402
from visionguard import evaluation, models, tracking  # noqa: E402
from visionguard.config import RUNS_DIR, SPLIT_DIR, TrainConfig  # noqa: E402


def callbacks_for(checkpoint: Path, patience: int, best_so_far: float | None) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_loss",
            save_best_only=True,
            initial_value_threshold=best_so_far,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 3), min_lr=1e-7, verbose=1
        ),
    ]


def plot_curves(history: pd.DataFrame, path: Path, title: str, best_epoch: int | None = None) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, metric in zip(axes, ("loss", "accuracy")):
        axis.plot(history["epoch"], history[metric], label=f"train {metric}")
        axis.plot(history["epoch"], history[f"val_{metric}"], label=f"val {metric}")
        if best_epoch is not None and best_epoch in set(history["epoch"]):
            axis.axvline(best_epoch, color="crimson", ls="--", label=f"best epoch ({best_epoch})")
        if history["stage"].nunique() > 1:
            boundary = history.loc[history["stage"] == 2, "epoch"].min()
            axis.axvline(boundary - 0.5, color="gray", ls=":", label="fine-tuning starts")
        axis.set_xlabel("epoch (1-based)")
        axis.set_ylabel(metric)
        axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_model_tables(model: keras.Model, run_dir: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lambda text, **_: lines.append(text))
    (run_dir / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    rows = [
        {
            "layer": layer.name,
            "type": type(layer).__name__,
            "output_shape": str(layer.output.shape) if hasattr(layer, "output") else "",
            "params": layer.count_params(),
        }
        for layer in model.layers
    ]
    pd.DataFrame(rows).to_csv(run_dir / "layer_table.csv", index=False)


def predict_split(model: keras.Model, cfg: TrainConfig, split: str):
    dataset = vg_data.make_dataset(split, cfg.image_size, cfg.batch_size)
    probabilities = model.predict(dataset, verbose=0).ravel()
    labels = vg_data.read_split(split)["label"].to_numpy()
    return labels, probabilities


def run_context(cfg: TrainConfig):
    """Autologging needs one run per fit, because parameters cannot change inside a run."""
    if cfg.logging == "autolog":
        mlflow.tensorflow.autolog(log_models=True, log_datasets=False, silent=True)
    return mlflow.start_run(run_name=cfg.run_name)


def stage_context(cfg: TrainConfig, stage: int):
    if cfg.logging == "autolog":
        return mlflow.start_run(run_name=f"{cfg.run_name}-stage{stage}", nested=True)
    return contextlib.nullcontext()


def fit_stage(model, cfg, train_ds, val_ds, checkpoint, stage, lr, epochs, offset, best_so_far, weights):
    models.compile_binary(model, lr)
    callbacks = callbacks_for(checkpoint, cfg.patience, best_so_far)
    with stage_context(cfg, stage):
        if cfg.logging == "autolog":
            mlflow.set_tags(tracking.standard_tags(cfg.seed))
        else:
            callbacks.append(tracking.EpochMetricLogger(step_offset=offset))
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, class_weight=weights,
            callbacks=callbacks, verbose=2,
        ).history
    frame = pd.DataFrame(history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1) + offset)
    frame.insert(1, "stage", stage)
    return frame


def package_list(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    path.write_text(result.stdout, encoding="utf-8")


def train(cfg: TrainConfig) -> Path:
    keras.utils.set_random_seed(cfg.seed)
    run_dir = RUNS_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.keras"
    checkpoint.unlink(missing_ok=True)

    train_ds = vg_data.make_dataset(
        "train", cfg.image_size, cfg.batch_size, training=True, augment=cfg.augment, seed=cfg.seed
    )
    val_ds = vg_data.make_dataset("val", cfg.image_size, cfg.batch_size)
    weights = vg_data.class_weights()

    model = models.build_model(cfg.model, cfg.image_size, cfg.dropout, cfg.weights)
    write_model_tables(model, run_dir)
    trainable = {"stage_1": models.trainable_layer_names(model)}

    with run_context(cfg) as run:
        (run_dir / "mlflow_run_id.txt").write_text(run.info.run_id)
        mlflow.set_tags({**tracking.standard_tags(cfg.seed), "logging_approach": cfg.logging})
        if cfg.logging == "manual":
            mlflow.log_params({**cfg.to_dict(), "train_class_weights": json.dumps(weights)})
        else:
            mlflow.log_params({"model": cfg.model, "augment": cfg.augment, "image_size": cfg.image_size})

        frames = [
            fit_stage(model, cfg, train_ds, val_ds, checkpoint, 1, cfg.lr, cfg.epochs, 0, None, weights)
        ]
        if cfg.model != "cnn" and cfg.unfreeze != "none":
            best = float(frames[0]["val_loss"].min())
            model.load_weights(checkpoint)
            trainable["stage_2"] = models.unfreeze(model, cfg.model, cfg.unfreeze)
            offset = int(frames[0]["epoch"].max())
            frames.append(
                fit_stage(
                    model, cfg, train_ds, val_ds, checkpoint, 2, cfg.fine_lr,
                    cfg.fine_epochs, offset, best, weights,
                )
            )

        history = pd.concat(frames, ignore_index=True)
        history.to_csv(run_dir / "history.csv", index=False)
        best_row = history.loc[history["val_loss"].idxmin()]
        best_epoch = int(best_row["epoch"])
        (run_dir / "trainable_layers.json").write_text(json.dumps(trainable, indent=2))

        plot_curves(history, run_dir / "curves_all.png", f"{cfg.run_name}: loss and accuracy", best_epoch)
        for stage, frame in history.groupby("stage"):
            plot_curves(frame, run_dir / f"curves_stage{stage}.png", f"{cfg.run_name}: stage {stage}", best_epoch)

        best_model = keras.models.load_model(checkpoint)
        labels, probabilities = predict_split(best_model, cfg, "val")
        chosen = evaluation.choose_threshold(labels, probabilities)
        threshold = chosen["threshold"]
        val_metrics = evaluation.compute_metrics(labels, probabilities, threshold)
        result = {"best_epoch": best_epoch, "best_val_loss": float(best_row["val_loss"]), **chosen}
        (run_dir / "val_result.json").write_text(json.dumps({**result, **val_metrics}, indent=2))
        evaluation.plot_confusion(labels, probabilities, threshold, run_dir / "val_confusion_matrix.png", "Validation")
        evaluation.plot_roc(labels, probabilities, run_dir / "val_roc_curve.png", "Validation ROC")

        mlflow.log_metrics({f"final_val_{k}": float(v) for k, v in val_metrics.items()})
        mlflow.log_metrics({"best_epoch": best_epoch, "best_val_loss": result["best_val_loss"], "decision_threshold": threshold})
        mlflow.set_tag("threshold_rule", chosen["rule"])
        package_list(run_dir / "requirements_frozen.txt")

        splits_copy = run_dir / "splits"
        shutil.copytree(SPLIT_DIR, splits_copy, dirs_exist_ok=True)
        mlflow.log_artifacts(str(splits_copy), "splits")
        for name in (
            "curves_all.png", "val_confusion_matrix.png", "val_roc_curve.png", "history.csv",
            "layer_table.csv", "model_summary.txt", "trainable_layers.json", "val_result.json",
        ):
            mlflow.log_artifact(str(run_dir / name), "report")
        mlflow.log_artifact(str(run_dir / "requirements_frozen.txt"), "environment")
        # MLflow 3 hides logged models from the Artifacts tab, so keep the raw file there too.
        mlflow.log_artifact(str(checkpoint), "keras_model")
        info = mlflow.tensorflow.log_model(best_model, name="model")
        (run_dir / "logged_model_uri.txt").write_text(info.model_uri)
        mlflow.set_tag("logged_model_uri", info.model_uri)

    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"[{cfg.run_name}] best epoch {best_epoch}, val_loss {result['best_val_loss']:.4f}, threshold {threshold:.3f}")
    return run_dir
