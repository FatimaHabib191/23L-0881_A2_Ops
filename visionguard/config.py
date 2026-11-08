"""Project-wide constants and the training configuration."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_NAME = "23L-0881_A2_Ops"
REGISTERED_MODEL = "VisionGuard"

DATA_DIR = Path(os.getenv("VG_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"

# Label index 1 is the positive class, so a false positive is a real photo called fake.
CLASS_NAMES = ["real", "ai_generated"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
MAX_FALSE_ALARM_RATE = 0.10


def class_from_path(relative_path: Path) -> str | None:
    """The label of an image is the top-level folder it sits in."""
    top = relative_path.parts[0].lower()
    if "real" in top:
        return "real"
    if top.startswith("ai") or "fake" in top or "generated" in top:
        return "ai_generated"
    return None


@dataclass
class TrainConfig:
    run_name: str
    model: str = "mobilenetv2"  # cnn | mobilenetv2 | efficientnetb0
    image_size: int = 224
    batch_size: int = 32
    lr: float = 1e-3
    fine_lr: float = 1e-5
    dropout: float = 0.3
    augment: str = "light"  # none | light | strong
    epochs: int = 30
    fine_epochs: int = 15
    unfreeze: str = "block"  # block | none | <integer number of layers>
    patience: int = 6
    seed: int = 42
    logging: str = "manual"  # manual | autolog
    weights: str = "imagenet"  # imagenet | none (none is for offline smoke tests)

    def to_dict(self) -> dict:
        return asdict(self)
