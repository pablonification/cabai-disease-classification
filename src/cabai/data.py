from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .config import (
    CLASS_NAMES,
    EXCLUDED_CLASSES,
    IMAGE_EXTENSIONS,
    INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_ALIASES,
    RANDOM_SEED,
    SPLIT_ALIASES,
)


def normalize_label(value: str) -> str:
    """Normalize folder/file labels into the five-class CabAI label space."""
    normalized = re.sub(r"[_\-]+", " ", value.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return LABEL_ALIASES.get(normalized, normalized)


def normalize_split(value: str) -> str | None:
    normalized = normalize_label(value)
    return SPLIT_ALIASES.get(normalized)


def find_image_files(root: Path | str) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def infer_label_and_split(image_path: Path, dataset_root: Path) -> tuple[str | None, str | None]:
    rel_parts = image_path.relative_to(dataset_root).parts[:-1]
    label = None
    split = None
    for part in rel_parts:
        maybe_label = normalize_label(part)
        maybe_split = normalize_split(part)
        if maybe_split:
            split = maybe_split
        if maybe_label in CLASS_NAMES or maybe_label in EXCLUDED_CLASSES:
            label = maybe_label
    return label, split


def build_manifest(
    dataset_root: Path | str,
    dataset_name: str = "dataset",
    allowed_classes: Iterable[str] = CLASS_NAMES,
    include_excluded: bool = False,
) -> list[dict[str, str]]:
    dataset_root = Path(dataset_root)
    allowed = set(allowed_classes)
    excluded = set(EXCLUDED_CLASSES)
    rows: list[dict[str, str]] = []
    for image_path in find_image_files(dataset_root):
        label, split = infer_label_and_split(image_path, dataset_root)
        if label is None:
            continue
        if label in excluded and not include_excluded:
            continue
        if label not in allowed and label not in excluded:
            continue
        rows.append(
            {
                "dataset": dataset_name,
                "split": split or "",
                "label": label,
                "path": str(image_path),
            }
        )
    return rows


def assign_missing_splits(
    manifest: list[dict[str, str]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> list[dict[str, str]]:
    """Assign train/val/test only for rows without split labels."""
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows = [dict(row) for row in manifest]
    for row in rows:
        if not row.get("split"):
            grouped[row["label"]].append(row)
    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        n_total = len(label_rows)
        n_test = max(1, round(n_total * test_ratio)) if n_total >= 3 else 0
        n_val = max(1, round(n_total * val_ratio)) if n_total >= 3 else 0
        for idx, row in enumerate(label_rows):
            if idx < n_test:
                row["split"] = "test"
            elif idx < n_test + n_val:
                row["split"] = "val"
            else:
                row["split"] = "train"
    return rows


def summarize_manifest(manifest: list[dict[str, str]]) -> dict[str, Counter]:
    by_label = Counter(row["label"] for row in manifest)
    by_split = Counter(row.get("split", "") or "unsplit" for row in manifest)
    by_split_label = Counter((row.get("split", "") or "unsplit", row["label"]) for row in manifest)
    return {
        "by_label": by_label,
        "by_split": by_split,
        "by_split_label": by_split_label,
    }


def write_manifest_csv(manifest: list[dict[str, str]], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "label", "path"])
        writer.writeheader()
        writer.writerows(manifest)


def read_manifest_csv(input_path: Path | str) -> list[dict[str, str]]:
    with Path(input_path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def get_transforms(train: bool = False, input_size: int = INPUT_SIZE):
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.Resize((input_size + 32, input_size + 32)),
                transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ImageManifestDataset:
    def __init__(self, manifest: list[dict[str, str]], transform=None, class_names: list[str] | None = None):
        from PIL import Image

        self.Image = Image
        self.manifest = list(manifest)
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest[idx]
        image = self.Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label, row["path"]


def make_dataloaders(
    manifest: list[dict[str, str]],
    batch_size: int = 16,
    num_workers: int = 2,
    class_names: list[str] | None = None,
):
    import torch
    from torch.utils.data import DataLoader

    class_names = class_names or CLASS_NAMES
    splits = {split: [row for row in manifest if row.get("split") == split] for split in ["train", "val", "test"]}
    pin_memory = torch.cuda.is_available()
    loaders = {}
    for split, rows in splits.items():
        if not rows:
            continue
        dataset = ImageManifestDataset(
            rows,
            transform=get_transforms(train=split == "train"),
            class_names=class_names,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders
