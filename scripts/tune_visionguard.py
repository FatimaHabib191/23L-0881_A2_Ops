"""Task 2.7 apply part: tune learning rate, dropout and unfrozen layers on validation AUC."""
import _bootstrap  # noqa: F401
import argparse
import shutil

import keras
import keras_tuner as kt
import numpy as np
import pandas as pd

from visionguard import data as vg_data
from visionguard import models
from visionguard.config import OUTPUT_DIR


class VisionGuardHyperModel(kt.HyperModel):
    def __init__(self, backbone, image_size, weights, warmup_epochs):
        super().__init__()
        self.backbone, self.image_size, self.weights, self.warmup_epochs = backbone, image_size, weights, warmup_epochs

    def build(self, hp):
        hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
        hp.Choice("n_unfrozen", [0, 10, 20, 40])
        dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
        return models.compile_binary(
            models.build_model(self.backbone, self.image_size, dropout, self.weights), 1e-3
        )

    def fit(self, hp, model, x, validation_data=None, callbacks=None, class_weight=None, epochs=6, **kwargs):
        # The head warms up first, so the tuned rate only controls fine-tuning.
        model.fit(x, validation_data=validation_data, epochs=self.warmup_epochs, class_weight=class_weight, verbose=0)
        models.unfreeze(model, self.backbone, int(hp.get("n_unfrozen")))
        models.compile_binary(model, hp.get("learning_rate"))
        return model.fit(x, validation_data=validation_data, epochs=epochs, class_weight=class_weight, callbacks=callbacks, verbose=0, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mobilenetv2")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--weights", default="imagenet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    keras.utils.set_random_seed(args.seed)
    out = OUTPUT_DIR / "tuner"
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "visionguard", ignore_errors=True)

    train = vg_data.make_dataset("train", args.image_size, 32, training=True, augment="light", seed=args.seed)
    val = vg_data.make_dataset("val", args.image_size, 32)
    tuner = kt.RandomSearch(
        VisionGuardHyperModel(args.backbone, args.image_size, args.weights, args.warmup_epochs),
        objective=kt.Objective("val_auc", "max"),
        max_trials=args.trials, seed=args.seed, directory=out, project_name="visionguard",
    )
    tuner.search(train, validation_data=val, epochs=args.epochs, class_weight=vg_data.class_weights())

    rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_auc": t.score} for t in tuner.oracle.get_best_trials(args.trials)]
    table = pd.DataFrame(rows)
    table.to_csv(out / "visionguard_results.csv", index=False)
    print(table.round(5).to_string(index=False))

    trial_inputs = table.assign(log10_lr=np.log10(table["learning_rate"]))[["log10_lr", "dropout", "n_unfrozen"]]
    influence = trial_inputs.corrwith(table["val_auc"], method="spearman").abs().sort_values(ascending=False)
    influence.rename("abs_spearman_with_val_auc").to_csv(out / "visionguard_influence.csv")
    print("\nInfluence on validation AUC (absolute Spearman correlation):")
    print(influence.round(3).to_string())
    print("\nBest settings:", tuner.get_best_hyperparameters(1)[0].values)


if __name__ == "__main__":
    main()
