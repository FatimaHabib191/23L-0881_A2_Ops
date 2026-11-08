"""Task 1.2: walk the dataset and build a per-image audit table."""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import warnings
from pathlib import Path

import pandas as pd
import tensorflow as tf
from PIL import Image

from visionguard.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT, class_from_path

Image.MAX_IMAGE_PIXELS = None  # The audit should measure huge files, not reject them.


def audit_file(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    row = {
        "path": path.relative_to(PROJECT_ROOT).as_posix() if PROJECT_ROOT in path.parents else str(path),
        "class": class_from_path(path.relative_to(root)),
        "file_size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "width": None, "height": None, "mode": None, "format": None,
        "readable": False, "tf_decodable": False, "error": "",
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                row.update(width=image.width, height=image.height, mode=image.mode, format=image.format)
        row["readable"] = True
    except Exception as error:  # Any decoder failure means the file is corrupt or not an image.
        row["error"] = f"PIL: {type(error).__name__}: {error}"[:200]
    try:
        tf.io.decode_image(raw, channels=3, expand_animations=False)
        row["tf_decodable"] = True
    except Exception as error:
        row["error"] = (row["error"] + f" TF: {type(error).__name__}")[:240]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    root = args.data_dir.resolve()
    out = OUTPUT_DIR / "audit"
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    table = pd.DataFrame([audit_file(p, root) for p in files])
    unlabelled = table["class"].isna().sum()
    table.to_csv(out / "audit.csv", index=False)

    counts = table.groupby("class").agg(
        files=("path", "size"), readable=("readable", "sum"), tf_decodable=("tf_decodable", "sum")
    )
    counts.to_csv(out / "counts_per_class.csv")

    good = table[table["readable"]]
    stats = good.groupby("class")[["width", "height", "file_size_bytes"]].agg(["min", "median", "max"])
    stats.to_csv(out / "size_stats_per_class.csv")

    duplicated = table[table.duplicated("sha256", keep=False)].sort_values("sha256")
    duplicated.to_csv(out / "duplicates.csv", index=False)
    extra_copies = int(table.duplicated("sha256").sum())
    cross_class = int((table.groupby("sha256")["class"].nunique() > 1).sum())
    formats = good.groupby(["class", "format"]).size().unstack(fill_value=0)
    modes = good.groupby(["class", "mode"]).size().unstack(fill_value=0)
    formats.to_csv(out / "formats_per_class.csv")
    modes.to_csv(out / "modes_per_class.csv")

    report = [
        f"Dataset root: {root}", f"Files scanned: {len(table)}", f"Files without a class folder: {unlabelled}", "",
        "Counts per class:", counts.to_string(), "",
        "Width, height and file size (min, median, max):", stats.to_string(), "",
        f"Corrupt or unreadable (PIL): {int((~table['readable']).sum())}",
        f"Not decodable by TensorFlow: {int((~table['tf_decodable']).sum())}",
        f"Exact duplicates: {extra_copies} extra copies in {table[table.duplicated('sha256', keep=False)]['sha256'].nunique()} groups",
        f"Hashes appearing under both classes: {cross_class}", "",
        "Formats:", formats.to_string(), "", "Colour modes:", modes.to_string(),
    ]
    text = "\n".join(report)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
