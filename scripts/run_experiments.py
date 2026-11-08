"""Task 3.5: runs that each differ from E1 in exactly one meaningful setting."""
import _bootstrap  # noqa: F401
import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = {
    "E1-base": [],
    "E2-lr-1e-4": ["--lr", "1e-4"],
    "E3-no-augment": ["--augment", "none"],
    "E4-strong-augment": ["--augment", "strong"],
    "E5-frozen-backbone": ["--unfreeze", "none"],
    "E6-image-160": ["--image-size", "160"],
    "E7-efficientnetb0": ["--model", "efficientnetb0"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Arguments added to every run")
    args = parser.parse_args()
    script = Path(__file__).with_name("train.py")
    for name, change in EXPERIMENTS.items():
        print(f"\n===== {name} =====", flush=True)
        subprocess.run([sys.executable, str(script), "--run-name", name, *change, *args.extra], check=True)


if __name__ == "__main__":
    main()
