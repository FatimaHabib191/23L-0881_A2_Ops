"""Generate meaningless synthetic images for offline pipeline tests only.

Results obtained on this data say nothing about the Kaggle dataset.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=60)
    args = parser.parse_args()
    random = np.random.default_rng(0)
    for folder, smooth in (("AiArtData/AiArtData", True), ("RealArt/RealArt", False)):
        target = args.out / folder
        target.mkdir(parents=True, exist_ok=True)
        for index in range(args.per_class):
            size = (int(random.integers(200, 420)), int(random.integers(200, 420)))
            pixels = random.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            if smooth:
                image = image.filter(ImageFilter.GaussianBlur(4))
            extension = "png" if index % 3 == 0 else "jpg"
            image.save(target / f"img_{index:03d}.{extension}")
        (target / "corrupt.jpg").write_bytes(b"this is not an image")
        (target / "copy_of_0.jpg").write_bytes((target / "img_001.jpg").read_bytes())


if __name__ == "__main__":
    main()
