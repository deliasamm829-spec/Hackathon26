"""Model evaluation and report generation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute common binary-classification metrics."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
    if y_probability is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_probability))
    return metrics


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Save metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_confusion_matrix(matrix: list[list[int]], output_path: Path, title: str) -> None:
    """Save a confusion matrix figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks([0, 1], labels=["Legitimate", "Phishing"])
    axis.set_yticks([0, 1], labels=["Legitimate", "Phishing"])
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            axis.text(col_index, row_index, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_probability: np.ndarray, output_path: Path) -> None:
    """Save a ROC curve when both classes are present."""
    if len(set(y_true)) < 2:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_probability)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, label="ROC curve")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
