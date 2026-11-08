"""Tasks 3.1 and 3.2: tiny Keras model logged to local MLflow or to DagsHub.

python scripts/mlflow_toy.py --backend local  --lr 0.001
python scripts/mlflow_toy.py --backend local  --lr 0.01
python scripts/mlflow_toy.py --backend auto   --lr 0.001     (DagsHub, using .env)
"""
import _bootstrap  # noqa: F401
import argparse
import os
import tempfile
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402

from visionguard.tracking import EpochMetricLogger, setup_tracking  # noqa: E402


def load_data():
    if os.getenv("VG_SMOKE"):
        x = np.random.rand(600, 28, 28)
        return x, (x.mean(axis=(1, 2)) > 0.5).astype(int)
    (x, y), _ = keras.datasets.mnist.load_data()
    return x[:10000] / 255.0, y[:10000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--backend", choices=["auto", "local"], default="auto")
    args = parser.parse_args()
    setup_tracking("toy-mnist", backend=args.backend)
    keras.utils.set_random_seed(42)
    x, y = load_data()
    model = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    with mlflow.start_run(run_name=f"toy-lr-{args.lr}") as run:
        mlflow.log_params({"learning_rate": args.lr, "epochs": args.epochs, "units": 64})
        mlflow.set_tag("purpose", "tool-lab")
        history = model.fit(x, y, validation_split=0.2, epochs=args.epochs, callbacks=[EpochMetricLogger()], verbose=2).history
        figure, axis = plt.subplots(figsize=(5, 3))
        axis.plot(history["loss"], label="train")
        axis.plot(history["val_loss"], label="validation")
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.legend()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "loss_curve.png"
            figure.savefig(path, dpi=120)
            mlflow.log_artifact(str(path), "plots")
        mlflow.tensorflow.log_model(model, name="model")
        print(f"Logged run {run.info.run_id} at lr={args.lr}, val_accuracy={history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
