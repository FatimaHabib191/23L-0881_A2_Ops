"""tf.data input pipeline driven by the split CSV files."""
from __future__ import annotations

from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

from visionguard.config import PROJECT_ROOT, SPLIT_DIR
from visionguard.preprocess import decode_and_resize

AUTOTUNE = tf.data.AUTOTUNE


def build_augmenter(level: str, seed: int) -> keras.Sequential | None:
    """Random label-preserving transforms. Pixel-level cues matter here, so 'light' is the default."""
    if level == "none":
        return None
    steps: list[keras.layers.Layer] = [keras.layers.RandomFlip("horizontal", seed=seed)]
    if level == "light":
        steps += [keras.layers.RandomZoom(0.1, seed=seed + 1)]
    elif level == "strong":
        steps += [
            keras.layers.RandomRotation(0.08, seed=seed + 1),
            keras.layers.RandomZoom(0.2, seed=seed + 2),
            keras.layers.RandomContrast(0.2, value_range=(0, 255), seed=seed + 3),
            keras.layers.RandomBrightness(0.2, value_range=(0, 255), seed=seed + 4),
        ]
    else:
        raise ValueError(f"Unknown augmentation level: {level}")
    return keras.Sequential(steps, name=f"augment_{level}")


def read_split(split: str) -> pd.DataFrame:
    return pd.read_csv(SPLIT_DIR / f"{split}.csv")


def make_dataset(
    split: str,
    image_size: int,
    batch_size: int,
    training: bool = False,
    augment: str = "none",
    seed: int = 42,
    root: Path = PROJECT_ROOT,
) -> tf.data.Dataset:
    frame = read_split(split)
    paths = [str(root / p) for p in frame["path"]]
    labels = frame["label"].astype("float32").to_numpy()

    def load(path, label):
        return decode_and_resize(tf.io.read_file(path), image_size), tf.expand_dims(label, 0)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load, num_parallel_calls=AUTOTUNE, deterministic=True).cache()
    if training:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    augmenter = build_augmenter(augment, seed) if training else None
    if augmenter is not None:
        dataset = dataset.map(
            lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE
        )
    return dataset.prefetch(AUTOTUNE)


def class_weights(split: str = "train") -> dict[int, float]:
    """Balanced weights so a mild class imbalance does not bias the decision."""
    counts = read_split(split)["label"].value_counts()
    total = counts.sum()
    return {int(k): float(total / (2 * v)) for k, v in counts.items()}
