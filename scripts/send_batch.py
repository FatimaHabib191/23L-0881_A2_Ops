"""Task 8: send a folder of images through the Flask site and print each outcome."""
import argparse
import re
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--url", default="http://localhost:5000")
    args = parser.parse_args()
    for path in sorted(p for p in args.folder.iterdir() if p.is_file()):
        with path.open("rb") as handle:
            response = requests.post(f"{args.url}/predict", files={"image": (path.name, handle)}, timeout=120)
        verdict = re.search(r'class="verdict">([^<]+)<', response.text)
        confidence = re.search(r'class="confidence">([^<]+)<', response.text)
        outcome = f"{verdict[1]}, {confidence[1]}" if verdict and confidence else "no verdict"
        print(f"{path.name:<45} HTTP {response.status_code}  {outcome}")


if __name__ == "__main__":
    main()
