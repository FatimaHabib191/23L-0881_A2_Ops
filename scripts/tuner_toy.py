"""Task 2.7 toy part: tune a tiny Fashion-MNIST network with random search."""
import _bootstrap  # noqa: F401
import argparse
import os
import shutil

import keras
import keras_tuner as kt
import numpy as np
import pandas as pd

from visionguard.config import OUTPUT_DIR


def load_data():
    if os.getenv("VG_SMOKE"):
        x = np.random.rand(600, 28, 28)
        y = np.random.randint(0, 10, 600)
        return (x, y)
    (x, y), _ = keras.datasets.fashion_mnist.load_data()
    return x[:12000] / 255.0, y[:12000]


def build(hp: kt.HyperParameters) -> keras.Model:
    model = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    out = OUTPUT_DIR / "tuner"
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "toy", ignore_errors=True)
    x, y = load_data()
    tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=args.trials, seed=42, directory=out, project_name="toy")
    tuner.search(x, y, validation_split=0.2, epochs=args.epochs, verbose=0)
    rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score} for t in tuner.oracle.get_best_trials(args.trials)]
    table = pd.DataFrame(rows)
    table.to_csv(out / "toy_results.csv", index=False)
    print(table.round(5).to_string(index=False))
    print("Best settings:", tuner.get_best_hyperparameters(1)[0].values)


if __name__ == "__main__":
    main()
