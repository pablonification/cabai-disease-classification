from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    import torch

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, labels, _paths in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_seen += images.size(0)
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }


def evaluate_epoch(model, loader, criterion, device):
    import torch

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total_seen += images.size(0)
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }


def fit_model(
    model,
    loaders,
    device,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    checkpoint_path: str | Path | None = None,
):
    import torch
    from tqdm.auto import tqdm

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    best_val_accuracy = -1.0
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    model.to(device)

    for epoch in tqdm(range(1, epochs + 1), desc="training"):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate_epoch(model, loaders.get("val", loaders["train"]), criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        if checkpoint_path and row["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = row["val_accuracy"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )
    return history
