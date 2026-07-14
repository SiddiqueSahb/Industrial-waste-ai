"""WaRP-C dataset — handles the nested folder layout on disk.

Layout:
    <root>/train_crops/<superclass>/<class>/*.jpg
    <root>/test_crops/<superclass>/<class>/*.jpg

Class indices are assigned by sorting the 28 leaf folder names alphabetically,
which matches `dataset.classes` in configs/classification.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_classes(split_dir: Path) -> List[str]:
    """Return the sorted leaf class folder names under a split directory."""
    leaves = sorted(
        d.name
        for super_dir in split_dir.iterdir() if super_dir.is_dir()
        for d in super_dir.iterdir() if d.is_dir()
    )
    if not leaves:
        raise FileNotFoundError(f"No class folders found under {split_dir}")
    return leaves


class WarpCDataset(Dataset):
    """Image-classification dataset over the nested WaRP-C crop folders."""

    def __init__(
        self,
        split_dir: str | Path,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
        limit_per_class: Optional[int] = None,
    ):
        self.split_dir = Path(split_dir)
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.classes = class_names or discover_classes(self.split_dir)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        for super_dir in sorted(self.split_dir.iterdir()):
            if not super_dir.is_dir():
                continue
            for class_dir in sorted(super_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                idx = self.class_to_idx.get(class_dir.name)
                if idx is None:
                    raise KeyError(
                        f"Folder {class_dir.name!r} not in the configured class list"
                    )
                files = sorted(
                    p for p in class_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                )
                if limit_per_class is not None:
                    files = files[:limit_per_class]
                self.samples.extend((p, idx) for p in files)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
