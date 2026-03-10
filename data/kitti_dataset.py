"""
data/kitti_dataset.py

PyTorch Dataset classes for KITTI object detection.
Supports both YOLOv8 (via Ultralytics) and DETR (via HuggingFace Transformers).
"""

import os
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.kitti_parser import (
    parse_kitti_split, KITTI_CLASSES, CLASS_NAMES
)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 640) -> A.Compose:
    """Albumentations pipeline for training with bbox-aware augmentations."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=0, value=(114, 114, 114)
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, p=0.1),  # Driving realism
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ))


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """Validation pipeline — resize + normalize only."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=0, value=(114, 114, 114)
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Base KITTI Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KITTIDataset(Dataset):
    """
    Base PyTorch Dataset for KITTI object detection.

    Args:
        images_dir:   Path to KITTI image directory (e.g., training/image_2).
        labels_dir:   Path to KITTI label directory (e.g., training/label_2).
        split:        'train' or 'val'
        img_size:     Target image size for resizing.
        transforms:   Optional Albumentations Compose pipeline.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        split: str = "train",
        img_size: int = 640,
        transforms: Optional[A.Compose] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.split = split

        # Parse all samples
        self.samples = parse_kitti_split(
            images_dir=images_dir,
            labels_dir=labels_dir,
            min_bbox_area=100.0,
            max_truncation=0.8,
            max_occlusion=2,
        )
        print(f"[KITTIDataset] {split}: {len(self.samples)} images loaded.")

        # Default transforms
        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = get_train_transforms(img_size)
        else:
            self.transforms = get_val_transforms(img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = np.array(Image.open(sample["image_path"]).convert("RGB"))
        h, w = image.shape[:2]

        bboxes = [list(obj.bbox) for obj in sample["objects"]]   # [x1,y1,x2,y2]
        labels = [obj.class_id for obj in sample["objects"]]

        # Apply transforms
        if bboxes:
            aug = self.transforms(
                image=image, bboxes=bboxes, class_labels=labels
            )
            image = aug["image"]
            bboxes = list(aug["bboxes"])
            labels = list(aug["class_labels"])
        else:
            aug = self.transforms(image=image, bboxes=[], class_labels=[])
            image = aug["image"]

        return {
            "image": image,                                    # Tensor [C, H, W]
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": sample["image_id"],
            "image_path": sample["image_path"],
            "orig_size": (h, w),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DETR-compatible Dataset (HuggingFace format)
# ─────────────────────────────────────────────────────────────────────────────

class KITTIDatasetDETR(KITTIDataset):
    """
    KITTI dataset formatted for HuggingFace DETR.
    Returns targets in COCO-style format expected by DetrForObjectDetection.
    """

    def __getitem__(self, idx: int) -> Dict:
        base = super().__getitem__(idx)
        h, w = base["orig_size"]
        boxes = base["boxes"]  # [N, 4] in pixel (x1,y1,x2,y2) after resize

        # DETR expects COCO format: [cx, cy, w, h] normalized
        if len(boxes) > 0:
            new_h, new_w = self.img_size, self.img_size
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / new_w
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / new_h
            bw = (boxes[:, 2] - boxes[:, 0]) / new_w
            bh = (boxes[:, 3] - boxes[:, 1]) / new_h
            cxcywh = torch.stack([cx, cy, bw, bh], dim=1).clamp(0, 1)
        else:
            cxcywh = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "image_id": torch.tensor([idx]),
            "boxes": cxcywh,
            "class_labels": base["labels"],
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([self.img_size, self.img_size]),
        }
        return base["image"], target


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Dataset Converter (writes files to disk)
# ─────────────────────────────────────────────────────────────────────────────

def convert_kitti_to_yolo(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    split: str = "train",
) -> None:
    """
    Convert KITTI annotations to YOLO format and write to output_dir.

    Creates:
        output_dir/images/{split}/*.png  (symlinks or copies)
        output_dir/labels/{split}/*.txt  (YOLO format labels)
    """
    from PIL import Image as PILImage
    import shutil

    images_out = Path(output_dir) / "images" / split
    labels_out = Path(output_dir) / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    samples = parse_kitti_split(images_dir, labels_dir)
    skipped = 0

    for sample in samples:
        img_path = Path(sample["image_path"])
        img = PILImage.open(img_path)
        img_w, img_h = img.size

        # Write label file
        label_lines = []
        for obj in sample["objects"]:
            line = obj.to_yolo_format(img_w, img_h)
            if line:
                label_lines.append(line)

        label_out_path = labels_out / f"{sample['image_id']}.txt"
        with open(label_out_path, "w") as f:
            f.write("\n".join(label_lines))

        # Symlink image (saves disk space)
        img_out_path = images_out / img_path.name
        if not img_out_path.exists():
            try:
                os.symlink(img_path.resolve(), img_out_path)
            except Exception:
                shutil.copy2(str(img_path), str(img_out_path))

    print(f"[convert_kitti_to_yolo] Converted {len(samples)} samples → {output_dir}")
    if skipped:
        print(f"  Skipped {skipped} samples with no valid labels.")


def collate_fn_detr(batch):
    """Custom collate for variable-size DETR targets."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        convert_kitti_to_yolo(
            images_dir=sys.argv[1],
            labels_dir=sys.argv[2],
            output_dir=sys.argv[3],
            split=sys.argv[4] if len(sys.argv) > 4 else "train",
        )
    else:
        print("Usage: python kitti_dataset.py <images_dir> <labels_dir> <output_dir> [split]")