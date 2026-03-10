"""
data/kitti_dataset.py

PyTorch Dataset for KITTI object detection.

Paths come from utils/paths.py (which reads .env) — no arguments needed.
The train/val split is handled internally; no split directories on disk.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.kitti_parser import parse_kitti_label_file, CLASS_NAMES
from utils.paths import KITTI_RAW_IMAGES, KITTI_RAW_LABELS, KITTI_YOLO_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=0, fill=(114, 114, 114)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
    ))


def get_val_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=0, fill=(114, 114, 114)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Split helper — in memory, no disk writes
# ─────────────────────────────────────────────────────────────────────────────

def get_split_ids(
    images_dir: Path,
    split: str = "train",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> List[str]:
    """
    Shuffle all image IDs with a fixed seed and return the train or val slice.
    No files are written to disk.
    """
    all_ids = sorted(p.stem for p in images_dir.glob("*.png"))
    if not all_ids:
        raise FileNotFoundError(f"No .png images found in {images_dir}")

    rng = random.Random(seed)
    rng.shuffle(all_ids)

    n_val   = int(len(all_ids) * val_ratio)
    ids     = all_ids[:n_val] if split == "val" else all_ids[n_val:]
    print(f"[KITTIDataset] '{split}': {len(ids)} / {len(all_ids)} images")
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KITTIDataset(Dataset):
    """
    KITTI 2D object detection dataset.

    Reads paths from utils/paths.py — no arguments needed in most cases.

    Example:
        train_ds = KITTIDataset(split="train")
        val_ds   = KITTIDataset(split="val")

    Override paths only if needed:
        ds = KITTIDataset(split="train",
                          images_dir="/custom/path/image_2",
                          labels_dir="/custom/path/label_2")
    """

    def __init__(
        self,
        split: str = "train",
        images_dir: Optional[Path] = None,
        labels_dir: Optional[Path] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        img_size: int = 640,
        transforms: Optional[A.Compose] = None,
        min_bbox_area: float = 100.0,
        max_truncation: float = 0.8,
        max_occlusion: int = 2,
    ):
        super().__init__()
        # Fall back to paths.py if not explicitly given
        self.images_dir     = Path(images_dir) if images_dir else KITTI_RAW_IMAGES
        self.labels_dir     = Path(labels_dir) if labels_dir else KITTI_RAW_LABELS
        self.split          = split
        self.img_size       = img_size
        self.min_bbox_area  = min_bbox_area
        self.max_truncation = max_truncation
        self.max_occlusion  = max_occlusion

        self.ids = get_split_ids(self.images_dir, split, val_ratio, seed)

        self.transforms = transforms or (
            get_train_transforms(img_size) if split == "train"
            else get_val_transforms(img_size)
        )

    def __len__(self) -> int:
        return len(self.ids)

    def _load_objects(self, image_id: str) -> Tuple[List, List]:
        label_path = self.labels_dir / f"{image_id}.txt"
        if not label_path.exists():
            return [], []
        bboxes, class_ids = [], []
        for obj in parse_kitti_label_file(str(label_path)):
            if not obj.is_valid:
                continue
            if obj.truncated > self.max_truncation:
                continue
            if obj.occluded > self.max_occlusion:
                continue
            x1, y1, x2, y2 = obj.bbox
            if (x2 - x1) * (y2 - y1) < self.min_bbox_area:
                continue
            bboxes.append(list(obj.bbox))
            class_ids.append(obj.class_id)
        return bboxes, class_ids

    def __getitem__(self, idx: int) -> Dict:
        image_id   = self.ids[idx]
        image_path = self.images_dir / f"{image_id}.png"
        image      = np.array(Image.open(image_path).convert("RGB"))
        h, w       = image.shape[:2]

        bboxes, class_ids = self._load_objects(image_id)
        aug = self.transforms(
            image=image,
            bboxes=bboxes or [],
            class_labels=class_ids or [],
        )

        return {
            "image":      aug["image"],
            "boxes":      torch.tensor(list(aug["bboxes"]),    dtype=torch.float32),
            "labels":     torch.tensor(list(aug["class_labels"]), dtype=torch.long),
            "image_id":   image_id,
            "image_path": str(image_path),
            "orig_size":  (h, w),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DETR-compatible Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KITTIDatasetDETR(KITTIDataset):
    """KITTIDataset returning COCO-style cxcywh normalised targets for DETR."""

    def __getitem__(self, idx: int) -> Tuple:
        base  = super().__getitem__(idx)
        h, w  = base["orig_size"]
        boxes = base["boxes"]

        if len(boxes) > 0:
            s      = float(self.img_size)
            cx     = (boxes[:, 0] + boxes[:, 2]) / 2 / s
            cy     = (boxes[:, 1] + boxes[:, 3]) / 2 / s
            bw     = (boxes[:, 2] - boxes[:, 0]) / s
            bh     = (boxes[:, 3] - boxes[:, 1]) / s
            cxcywh = torch.stack([cx, cy, bw, bh], dim=1).clamp(0, 1)
        else:
            cxcywh = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "image_id":     torch.tensor([idx]),
            "boxes":        cxcywh,
            "class_labels": base["labels"],
            "orig_size":    torch.tensor([h, w]),
            "size":         torch.tensor([self.img_size, self.img_size]),
        }
        return base["image"], target


# ─────────────────────────────────────────────────────────────────────────────
# YOLO format converter
# ─────────────────────────────────────────────────────────────────────────────

def convert_kitti_to_yolo(split: str = "train", val_ratio: float = 0.2, seed: int = 42) -> None:
    """
    Convert KITTI annotations to YOLO format on disk.
    Reads/writes paths from paths.py — no arguments needed.

    Usage:
        python -c "from data.kitti_dataset import convert_kitti_to_yolo; convert_kitti_to_yolo('train')"
        python -c "from data.kitti_dataset import convert_kitti_to_yolo; convert_kitti_to_yolo('val')"
    """
    import shutil

    ids        = get_split_ids(KITTI_RAW_IMAGES, split, val_ratio, seed)
    images_out = KITTI_YOLO_DIR / "images" / split
    labels_out = KITTI_YOLO_DIR / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for image_id in ids:
        img_path = KITTI_RAW_IMAGES / f"{image_id}.png"
        lbl_path = KITTI_RAW_LABELS / f"{image_id}.txt"

        from PIL import Image as PILImage
        img_w, img_h = PILImage.open(img_path).size

        objects = parse_kitti_label_file(str(lbl_path)) if lbl_path.exists() else []
        lines   = [obj.to_yolo_format(img_w, img_h) for obj in objects
                   if obj.is_valid and obj.to_yolo_format(img_w, img_h)]

        with open(labels_out / f"{image_id}.txt", "w") as f:
            f.write("\n".join(lines))

        dst = images_out / img_path.name
        if not dst.exists():
            try:
                os.symlink(img_path.resolve(), dst)
            except OSError:
                shutil.copy2(str(img_path), str(dst))

    print(f"[convert_kitti_to_yolo] '{split}': {len(ids)} samples → {KITTI_YOLO_DIR}")


def collate_fn_detr(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test  —  python data/kitti_dataset.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.paths import print_paths
    print_paths()

    train_ds = KITTIDataset(split="train")
    val_ds   = KITTIDataset(split="val")

    assert len(set(train_ds.ids) & set(val_ds.ids)) == 0, "LEAK: train and val share IDs!"
    print(f"  No overlap. Train={len(train_ds)}  Val={len(val_ds)}")

    sample = train_ds[0]
    print(f"    image shape : {sample['image'].shape}")
    print(f"    boxes       : {sample['boxes'].shape}")
    print(f"    labels      : {sample['labels']}")