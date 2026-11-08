"""Threshold selection, metrics and evaluation figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import metrics as sk  # noqa: E402

from visionguard.config import CLASS_NAMES, MAX_FALSE_ALARM_RATE  # noqa: E402


def choose_threshold(y_true, y_prob, max_fpr: float = MAX_FALSE_ALARM_RATE) -> dict:
    """Maximise AI recall while real photos are wrongly flagged at most max_fpr of the time.

    Falls back to the best F1 threshold if no threshold meets the constraint.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    fpr, tpr, thresholds = sk.roc_curve(y_true, y_prob)
    finite = np.isfinite(thresholds) & (thresholds <= 1.0)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    allowed = np.where(fpr <= max_fpr)[0]
    if len(allowed) and tpr[allowed].max() > 0:
        best = allowed[np.argmax(tpr[allowed])]
        return {
            "threshold": float(thresholds[best]),
            "rule": f"max recall with FPR <= {max_fpr:.2f}",
            "val_fpr": float(fpr[best]),
            "val_recall": float(tpr[best]),
        }
    precision, recall, pr_thresholds = sk.precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    best = int(np.argmax(f1))
    threshold = float(pr_thresholds[best])
    predicted = y_prob >= threshold
    negatives = max(int((y_true == 0).sum()), 1)
    return {
        "threshold": threshold,
        "rule": "max F1 (FPR constraint not reachable)",
        "val_fpr": float((predicted & (y_true == 0)).sum() / negatives),
        "val_recall": float(recall[best]),
    }


def compute_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = sk.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(sk.accuracy_score(y_true, y_pred)),
        "precision": float(sk.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(sk.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(sk.f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(sk.roc_auc_score(y_true, y_prob)),
        "false_alarm_rate": float(fp / max(fp + tn, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_confusion(y_true, y_prob, threshold: float, path: Path, title: str) -> None:
    matrix = sk.confusion_matrix(y_true, np.asarray(y_prob) >= threshold, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(4.4, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], CLASS_NAMES)
    axis.set_yticks([0, 1], CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"{title}\nthreshold={threshold:.3f}")
    for (row, col), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() / 2 else "black"
        axis.text(col, row, str(value), ha="center", va="center", color=color, fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_roc(y_true, y_prob, path: Path, title: str) -> None:
    fpr, tpr, _ = sk.roc_curve(y_true, y_prob)
    figure, axis = plt.subplots(figsize=(4.6, 4))
    axis.plot(fpr, tpr, label=f"AUC = {sk.auc(fpr, tpr):.3f}")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    axis.set_xlabel("False positive rate (real called AI)")
    axis.set_ylabel("True positive rate (AI caught)")
    axis.set_title(title)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
