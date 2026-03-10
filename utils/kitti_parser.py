"""
utils/kitti_parser.py

Parses KITTI object detection annotations into a unified format.

KITTI label format (per line):
  type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y [score]

Classes we care about: Car, Pedestrian, Cyclist, Van, Truck
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


# KITTI class mapping → YOLO class indices
KITTI_CLASSES = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Van": 3,
    "Truck": 4,
    "Misc": 5,
    "DontCare": -1,   # Ignored during training
    "Person_sitting": 1,  # Map to Pedestrian
    "Tram": 4,             # Map to Truck
}

CLASS_COLORS = {
    0: (255, 100, 0),    # Car — orange
    1: (0, 220, 50),     # Pedestrian — green
    2: (255, 220, 0),    # Cyclist — yellow
    3: (180, 0, 255),    # Van — purple
    4: (0, 120, 255),    # Truck — blue
    5: (150, 150, 150),  # Misc — gray
}

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist", "Van", "Truck", "Misc"]


@dataclass
class KITTIObject:
    """Single annotated object in a KITTI frame."""
    type: str
    truncated: float      # 0 (not truncated) to 1 (fully truncated)
    occluded: int         # 0=fully visible, 1=partly, 2=largely, 3=unknown
    alpha: float          # Observation angle [-pi, pi]
    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2) in pixels
    dimensions: Tuple[float, float, float]     # (height, width, length) in meters
    location: Tuple[float, float, float]       # (x, y, z) in camera coords
    rotation_y: float     # Rotation around Y-axis [-pi, pi]
    score: Optional[float] = None  # Detection confidence (predictions only)

    @property
    def class_id(self) -> int:
        return KITTI_CLASSES.get(self.type, -1)

    @property
    def is_valid(self) -> bool:
        """Returns True if object should be used for training (not DontCare)."""
        return self.class_id >= 0

    @property
    def bbox_width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def to_yolo_format(self, img_width: int, img_height: int) -> Optional[str]:
        """Convert to YOLO normalized format: class cx cy w h"""
        if not self.is_valid:
            return None
        x1, y1, x2, y2 = self.bbox
        cx = ((x1 + x2) / 2) / img_width
        cy = ((y1 + y2) / 2) / img_height
        w  = (x2 - x1) / img_width
        h  = (y2 - y1) / img_height
        # Clamp to [0, 1]
        cx, cy, w, h = [max(0.0, min(1.0, v)) for v in [cx, cy, w, h]]
        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def parse_kitti_label_file(label_path: str) -> List[KITTIObject]:
    """
    Parse a single KITTI label .txt file.

    Args:
        label_path: Path to the .txt annotation file.

    Returns:
        List of KITTIObject instances.
    """
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 15:
                continue

            obj = KITTIObject(
                type=parts[0],
                truncated=float(parts[1]),
                occluded=int(parts[2]),
                alpha=float(parts[3]),
                bbox=(float(parts[4]), float(parts[5]),
                      float(parts[6]), float(parts[7])),
                dimensions=(float(parts[8]), float(parts[9]), float(parts[10])),
                location=(float(parts[11]), float(parts[12]), float(parts[13])),
                rotation_y=float(parts[14]),
                score=float(parts[15]) if len(parts) > 15 else None,
            )
            objects.append(obj)
    return objects


def parse_kitti_split(
    images_dir: str,
    labels_dir: str,
    filter_classes: Optional[List[str]] = None,
    min_bbox_area: float = 100.0,
    max_truncation: float = 0.8,
    max_occlusion: int = 2,
) -> List[dict]:
    """
    Parse an entire KITTI split (train or val) into a list of samples.

    Args:
        images_dir:      Directory containing .png images.
        labels_dir:      Directory containing .txt label files.
        filter_classes:  If set, only include these class names.
        min_bbox_area:   Minimum bbox area in pixels to include.
        max_truncation:  Skip objects more truncated than this.
        max_occlusion:   Skip objects more occluded than this.

    Returns:
        List of dicts with keys: image_path, objects, image_id
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    filter_set = set(filter_classes) if filter_classes else None

    samples = []
    label_files = sorted(labels_dir.glob("*.txt"))

    for label_file in label_files:
        image_id = label_file.stem
        image_path = images_dir / f"{image_id}.png"
        if not image_path.exists():
            image_path = images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            continue

        objects = parse_kitti_label_file(str(label_file))

        # Apply filters
        filtered = []
        for obj in objects:
            if not obj.is_valid:
                continue
            if filter_set and obj.type not in filter_set:
                continue
            if obj.truncated > max_truncation:
                continue
            if obj.occluded > max_occlusion:
                continue
            area = obj.bbox_width * obj.bbox_height
            if area < min_bbox_area:
                continue
            filtered.append(obj)

        samples.append({
            "image_id": image_id,
            "image_path": str(image_path),
            "objects": filtered,
        })

    return samples


def get_class_distribution(samples: List[dict]) -> dict:
    """Count instances per class across all samples."""
    counts = {name: 0 for name in CLASS_NAMES}
    for sample in samples:
        for obj in sample["objects"]:
            name = obj.type
            if name in counts:
                counts[name] += 1
    return counts


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) < 3:
        print("Usage: python kitti_parser.py <images_dir> <labels_dir>")
        sys.exit(1)

    samples = parse_kitti_split(sys.argv[1], sys.argv[2])
    dist = get_class_distribution(samples)
    print(f"Parsed {len(samples)} samples")
    print("Class distribution:", dist)