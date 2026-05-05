from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES


def predict_loader(model, loader, device):
    import torch

    model.eval()
    y_true, y_pred, y_prob, paths = [], [], [], []
    with torch.no_grad():
        for images, labels, batch_paths in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.tolist())
            y_prob.extend(probabilities.tolist())
            paths.extend(batch_paths)
    return np.array(y_true), np.array(y_pred), np.array(y_prob), list(paths)


def classification_metrics(y_true, y_pred, class_names: list[str] | None = None):
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    class_names = class_names or CLASS_NAMES
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_metrics_json(metrics: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_history(history: list[dict], output_path: str | Path | None = None):
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names: list[str] | None = None, output_path: str | Path | None = None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    class_names = class_names or CLASS_NAMES
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Greens", xticks_rotation=30, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig
