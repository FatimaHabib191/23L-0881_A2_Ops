"""Bonus: Grad-CAM for 4 correct and 4 wrong test predictions."""
import _bootstrap  # noqa: F401
import argparse

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from visionguard.config import CLASS_NAMES, PROJECT_ROOT, RUNS_DIR  # noqa: E402
from visionguard.models import BACKBONE_LAYER  # noqa: E402
from visionguard.preprocess import bytes_to_batch  # noqa: E402


def split_layers(model: keras.Model) -> tuple[list, list]:
    """Split the model at its last spatial feature map."""
    names = [layer.name for layer in model.layers]
    if BACKBONE_LAYER in names:
        index = names.index(BACKBONE_LAYER)
    else:
        index = max(i for i, layer in enumerate(model.layers) if isinstance(layer, keras.layers.Conv2D))
    return model.layers[1:index + 1], model.layers[index + 1:]


def grad_cam(model: keras.Model, batch) -> tuple[np.ndarray, float]:
    """Highlight the regions that pushed the model towards its predicted class."""
    before, after = split_layers(model)
    features = batch
    for layer in before:
        features = layer(features, training=False)
    with tf.GradientTape() as tape:
        tape.watch(features)
        output = features
        for layer in after:
            output = layer(output, training=False)
        probability = output[:, 0]
        score = probability if float(probability[0]) >= 0.5 else 1.0 - probability
    weights = tf.reduce_mean(tape.gradient(score, features), axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8), float(probability[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_dir = RUNS_DIR / args.run_name
    model = keras.models.load_model(run_dir / "best.keras")
    size = model.input_shape[1]
    frame = pd.read_csv(run_dir / "test_predictions.csv")
    hit = frame["label"] == frame["predicted"]
    correct = frame[hit].sample(min(4, int(hit.sum())), random_state=args.seed)
    wrong = frame[~hit].sample(min(4, int((~hit).sum())), random_state=args.seed)
    selected = pd.concat([correct.assign(group="correct"), wrong.assign(group="wrong")])

    figure, axes = plt.subplots(2, 4, figsize=(13, 7))
    for axis in axes.ravel():
        axis.axis("off")
    for axis, record in zip(axes.ravel(), selected.itertuples()):
        batch = bytes_to_batch((PROJECT_ROOT / record.path).read_bytes(), size)
        cam, score = grad_cam(model, batch)
        axis.imshow(batch[0].numpy().astype("uint8"))
        axis.imshow(cam, cmap="jet", alpha=0.42, extent=(0, size, size, 0), interpolation="bilinear")
        axis.set_title(f"{record.group}: true {CLASS_NAMES[record.label]}\nP(AI)={score:.2f}", fontsize=10)
    figure.tight_layout()
    figure.savefig(run_dir / "gradcam_grid.png", dpi=130)
    print(f"Saved {run_dir / 'gradcam_grid.png'}")


if __name__ == "__main__":
    main()
