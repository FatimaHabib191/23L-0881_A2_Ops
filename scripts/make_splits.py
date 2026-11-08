"""Create leak-free stratified 70/15/15 train, validation and test lists."""
import _bootstrap  # noqa: F401
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from visionguard.config import LABEL_TO_INDEX, OUTPUT_DIR, SPLIT_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    table = pd.read_csv(OUTPUT_DIR / "audit" / "audit.csv")
    start = len(table)
    table = table[table["readable"] & table["tf_decodable"] & table["class"].notna()]

    # A hash under both classes has no trustworthy label, so all its copies are removed.
    per_hash = table.groupby("sha256")["class"].nunique()
    table = table[~table["sha256"].isin(per_hash[per_hash > 1].index)]
    # Keeping one copy per hash stops identical images crossing split boundaries.
    table = table.drop_duplicates("sha256").copy()
    table["label"] = table["class"].map(LABEL_TO_INDEX)
    columns = ["path", "label", "class", "sha256"]

    train, holdout = train_test_split(table[columns], test_size=0.30, stratify=table["label"], random_state=args.seed)
    val, test = train_test_split(holdout, test_size=0.50, stratify=holdout["label"], random_state=args.seed)
    assert not (set(train.sha256) & set(val.sha256) or set(train.sha256) & set(test.sha256) or set(val.sha256) & set(test.sha256))

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.sort_values("path").to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        print(f"{name:<6}{len(frame):>6} images  {frame['class'].value_counts().to_dict()}")
    print(f"Removed {start - len(table)} of {start} files (corrupt, undecodable, unlabelled, duplicates).")


if __name__ == "__main__":
    main()
