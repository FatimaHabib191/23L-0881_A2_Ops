"""Task 2.6: show the most confident test errors with measured starting hypotheses.

The hypotheses come from simple image measurements. Look at each image and
correct them; only a person can spot things such as watermarks or painting style.
"""
import _bootstrap  # noqa: F401
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from visionguard.config import CLASS_NAMES, PROJECT_ROOT, RUNS_DIR  # noqa: E402


def measure(image: Image.Image) -> dict:
    rgb = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
    grey = rgb.mean(axis=2)
    laplacian = -4 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
    saturation = (rgb.max(axis=2) - rgb.min(axis=2)) / np.clip(rgb.max(axis=2), 1, None)
    return {"sharpness": float(laplacian.var()), "saturation": float(saturation.mean())}


def hypothesis(row, stats: dict, true_name: str) -> str:
    reasons = []
    if stats["sharpness"] < 60:
        reasons.append("soft or blurred, so little fine detail")
    if stats["saturation"] > 0.6:
        reasons.append("strong saturated colours, painting-like")
    if min(row.width, row.height) < 256:
        reasons.append(f"small source ({row.width}x{row.height}), upscaled")
    if max(row.width, row.height) / max(min(row.width, row.height), 1) > 1.6:
        reasons.append("elongated image distorted by square resize")
    if not reasons:
        reasons.append("no measured anomaly; inspect content" if true_name == "real" else "photorealistic with no measured anomaly")
    return "; ".join(reasons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    run_dir = RUNS_DIR / args.run_name
    frame = pd.read_csv(run_dir / "test_predictions.csv")
    wrong = frame[frame["label"] != frame["predicted"]].copy()
    wrong["error_size"] = (wrong["label"] - wrong["probability_ai"]).abs()
    wrong = wrong.sort_values("error_size", ascending=False).head(10)
    if wrong.empty:
        raise SystemExit("No misclassified test images.")

    rows = []
    figure, axes = plt.subplots(2, 5, figsize=(16, 7.4))
    for axis in axes.ravel():
        axis.axis("off")
    for axis, record in zip(axes.ravel(), wrong.itertuples()):
        with Image.open(PROJECT_ROOT / record.path) as image:
            stats = measure(image)
            record_size = type("Size", (), {"width": image.width, "height": image.height})
            text = hypothesis(record_size, stats, CLASS_NAMES[record.label])
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"true {CLASS_NAMES[record.label]}\nP(AI)={record.probability_ai:.2f}", fontsize=10)
        axis.text(0, -0.04, text, transform=axis.transAxes, va="top", fontsize=8, wrap=True)
        rows.append({"path": record.path, "true": CLASS_NAMES[record.label], "p_ai": record.probability_ai, **stats, "hypothesis": text})
    figure.tight_layout(h_pad=3)
    figure.savefig(run_dir / "misclassified_grid.png", dpi=130)
    pd.DataFrame(rows).to_csv(run_dir / "misclassified.csv", index=False)
    print(pd.DataFrame(rows)[["true", "p_ai", "hypothesis"]].to_string(index=False))


if __name__ == "__main__":
    main()
