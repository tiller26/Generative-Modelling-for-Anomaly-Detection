"""PyTorch Dataset for MVTec AD.

Directory layout expected under `data/<category>/` (the layout produced by
`scripts/download_mvtec.py` / the official MVTec AD archives):

    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*_mask.png

`good` test images have no ground-truth mask (label = 0, mask = all zeros).
Defective test images are labelled 1 and paired with a binary mask.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_transform(image_size: int = 256, normalize: bool = True) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if normalize:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


def default_mask_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])


class MVTecTrainDataset(Dataset):
    """Normal-only images for training (`train/good`)."""

    def __init__(self, root: str | Path, category: str, image_size: int = 256, transform=None):
        self.root = Path(root) / category / "train" / "good"
        if not self.root.exists():
            raise FileNotFoundError(f"No such directory: {self.root}")
        self.paths = sorted(self.root.glob("*.png"))
        if not self.paths:
            raise RuntimeError(f"No training images found under {self.root}")
        self.transform = transform or default_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


class MVTecTestDataset(Dataset):
    """Test images spanning `good` + every defect subfolder, with masks."""

    def __init__(
        self,
        root: str | Path,
        category: str,
        image_size: int = 256,
        transform=None,
        mask_transform=None,
    ):
        self.category_root = Path(root) / category
        test_root = self.category_root / "test"
        gt_root = self.category_root / "ground_truth"
        if not test_root.exists():
            raise FileNotFoundError(f"No such directory: {test_root}")

        self.image_size = image_size
        self.transform = transform or default_transform(image_size)
        self.mask_transform = mask_transform or default_mask_transform(image_size)

        self.samples: list[tuple[Path, Path | None, int, str]] = []
        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            is_good = defect_type == "good"
            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if not is_good:
                    mask_path = gt_root / defect_type / f"{img_path.stem}_mask.png"
                    if not mask_path.exists():
                        raise FileNotFoundError(f"Missing mask for {img_path}: expected {mask_path}")
                self.samples.append((img_path, mask_path, int(not is_good), defect_type))

        if not self.samples:
            raise RuntimeError(f"No test images found under {test_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path, label, defect_type = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image_t = self.transform(image)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask_t = (self.mask_transform(mask) > 0.5).float()
        else:
            mask_t = torch.zeros(1, self.image_size, self.image_size)

        return {
            "image": image_t,
            "mask": mask_t,
            "label": label,
            "defect_type": defect_type,
            "path": str(img_path),
        }


def train_val_split(dataset: MVTecTrainDataset, val_fraction: float = 0.1, seed: int = 42):
    """Split `train/good` into train/val subsets (both still anomaly-free)."""
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)
