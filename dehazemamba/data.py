"""MRSHaze paired image dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MRSHazeDataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root).expanduser().resolve()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        self.gt_dir = self.root / split / "GT"
        self.hazy_dir = self.root / split / "hazy"
        self.sar_dir = self.root / split / "SAR"
        if not all(path.is_dir() for path in (self.gt_dir, self.hazy_dir, self.sar_dir)):
            raise FileNotFoundError(
                f"Expected {split}/GT, {split}/hazy and {split}/SAR below {self.root}"
            )

        self.names = sorted(path.name for path in self.gt_dir.glob("*.png"))
        if not self.names:
            raise FileNotFoundError(f"No PNG ground truth files in {self.gt_dir}")
        missing = [
            name
            for name in self.names
            if not (self.hazy_dir / name).is_file() or not (self.sar_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing paired hazy/SAR files, e.g. {missing[:5]}")

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _read(path: Path, mode: str) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert(mode), dtype=np.float32).copy() / 255.0
        if array.ndim == 2:
            array = array[:, :, None]
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.names[index]
        return {
            "hazy": self._read(self.hazy_dir / name, "RGB"),
            "sar": self._read(self.sar_dir / name, "L"),
            "target": self._read(self.gt_dir / name, "RGB"),
            "name": name,
        }
