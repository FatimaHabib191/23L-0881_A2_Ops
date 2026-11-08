"""Task 1.3: sample grids and per-class histograms. Task 2.1: one augmented batch."""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from visionguard.config import CLASS_NAMES, OUTPUT_DIR, PROJECT_ROOT, SPLIT_DIR  # noqa: E402


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def sample_grid(table: pd.DataFrame, out: Path, seed: int) -> None:
    columns = 8
    figure, axes = plt.subplots(2, columns, figsize=(columns * 1.9, 4.4))
    for row, name in enumerate(CLASS_NAMES):
        subset = table[table["class"] == name]
        sample = subset.sample(min(columns, len(subset)), random_state=seed)
        for col in range(columns):
            axis = axes[row, col]
            axis.axis("off")
            if col < len(sample):
                with Image.open(resolve(sample.iloc[col]["path"])) as image:
                    axis.imshow(image.convert("RGB"))
        axes[row, 0].set_title(name, loc="left", fontsize=11)
    figure.tight_layout()
    figure.savefig(out / "sample_grid.png", dpi=130)
    plt.close(figure)


def histograms(table: pd.DataFrame, out: Path) -> None:
    table = table.assign(file_size_kb=table["file_size_bytes"] / 1024)
    fields = [("width", "Width (px)", False), ("height", "Height (px)", False), ("file_size_kb", "File size (KB, log scale)", True)]
    for field, label, log_x in fields:
        figure, axis = plt.subplots(figsize=(6, 3.6))
        values = table[field].dropna()
        if log_x:
            bins = np.logspace(np.log10(max(values.min(), 0.1)), np.log10(values.max()), 40)
            axis.set_xscale("log")
        else:
            bins = np.linspace(values.min(), np.percentile(values, 99.5) + 1, 40)
        for name in CLASS_NAMES:
            axis.hist(table.loc[table["class"] == name, field], bins=bins, alpha=0.55, label=name)
        axis.set_xlabel(label)
        axis.set_ylabel("Images")
        axis.legend()
        figure.tight_layout()
        figure.savefig(out / f"hist_{field}.png", dpi=130)
        plt.close(figure)


def augmented_batch(out: Path, augment: str, seed: int) -> None:
    from visionguard.data import make_dataset

    images, labels = next(iter(make_dataset("train", 224, 12, training=True, augment=augment, seed=seed)))
    figure, axes = plt.subplots(2, 6, figsize=(12, 4.3))
    for index, axis in enumerate(axes.ravel()):
        axis.imshow(np.clip(images[index].numpy(), 0, 255).astype("uint8"))
        axis.set_title(CLASS_NAMES[int(labels[index][0])], fontsize=9)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(out / f"augmented_batch_{augment}.png", dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", default="light")
    args = parser.parse_args()
    out = OUTPUT_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(OUTPUT_DIR / "audit" / "audit.csv")
    table = table[table["readable"]]
    sample_grid(table, out, args.seed)
    histograms(table, out)
    if (SPLIT_DIR / "train.csv").exists():
        augmented_batch(out, args.augment, args.seed)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
