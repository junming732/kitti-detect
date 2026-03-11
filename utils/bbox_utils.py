"""
utils/bbox_utils.py

Bounding box utilities:
  - Format conversions  (xyxy ? xywh ? cxcywh)
  - Clipping / area / filtering
  - NMS (pure NumPy, no torch dependency)
  - Drawing helpers (thin wrappers used by visualize_video.py)
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Format conversions
# -----------------------------------------------------------------------------

def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] ? [x1,y1,w,h]"""
    out = boxes.copy().astype(float)
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """[x1,y1,w,h] ? [x1,y1,x2,y2]"""
    out = boxes.copy().astype(float)
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] ? [cx,cy,w,h]"""
    out = np.empty_like(boxes, dtype=float)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """[cx,cy,w,h] ? [x1,y1,x2,y2]"""
    out = np.empty_like(boxes, dtype=float)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def normalize_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """[x1,y1,x2,y2] pixels ? [x1,y1,x2,y2] normalized [0,1]."""
    out = boxes.astype(float)
    out[:, [0, 2]] /= img_w
    out[:, [1, 3]] /= img_h
    return np.clip(out, 0.0, 1.0)


def denormalize_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Normalized [0,1] ? pixel coordinates."""
    out = boxes.astype(float)
    out[:, [0, 2]] *= img_w
    out[:, [1, 3]] *= img_h
    return out


# -----------------------------------------------------------------------------
# Box properties
# -----------------------------------------------------------------------------

def box_area(boxes: np.ndarray) -> np.ndarray:
    """Area of [x1,y1,x2,y2] boxes."""
    return np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])


def clip_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Clip boxes to image boundaries."""
    out = boxes.copy().astype(float)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, img_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, img_h)
    return out


def filter_small_boxes(boxes: np.ndarray, min_area: float = 100.0) -> np.ndarray:
    """Return boolean mask for boxes with area >= min_area."""
    return box_area(boxes) >= min_area


# -----------------------------------------------------------------------------
# NMS (pure NumPy ? useful when torch is unavailable)
# -----------------------------------------------------------------------------

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Non-Maximum Suppression.

    Args:
        boxes:         (N, 4) [x1,y1,x2,y2]
        scores:        (N,)   confidence scores
        iou_threshold: overlap threshold

    Returns:
        Indices of kept boxes (sorted by descending score).
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    order = np.argsort(scores)[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=int)


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-class NMS with confidence filtering.

    Returns:
        (kept_boxes, kept_scores, kept_class_ids)
    """
    mask = scores >= conf_threshold
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
    if len(boxes) == 0:
        return boxes, scores, class_ids

    keep_all = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        kept = nms(boxes[idx], scores[idx], iou_threshold)
        keep_all.extend(idx[kept].tolist())

    keep_all = np.array(keep_all, dtype=int)
    return boxes[keep_all], scores[keep_all], class_ids[keep_all]


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------

# Default color palette (BGR) ? can be overridden
_DEFAULT_COLORS = [
    (255, 100,   0),   # 0 Car        ? orange
    (  0, 220,  50),   # 1 Pedestrian ? green
    (255, 220,   0),   # 2 Cyclist    ? yellow
    (180,   0, 255),   # 3 Van        ? purple
    (  0, 120, 255),   # 4 Truck      ? blue
    (150, 150, 150),   # 5 Misc       ? gray
]


def get_color(class_id: int, palette: Optional[List[Tuple]] = None) -> Tuple[int, int, int]:
    p = palette or _DEFAULT_COLORS
    return p[class_id % len(p)]


def draw_single_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
    font_scale: float = 0.52,
    alpha: float = 0.35,        # label background transparency
) -> np.ndarray:
    """
    Draw one bounding box with a semi-transparent label background.

    Args:
        frame:      BGR image (modified in-place)
        x1,y1,x2,y2: box corners
        label:      text to display (e.g. "Car 0.87")
        color:      BGR tuple
        thickness:  box border thickness
        font_scale: label font size
        alpha:      label background alpha

    Returns:
        Annotated frame
    """
    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    # Label size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    lx1 = x1
    ly1 = max(y1 - th - baseline - 4, 0)
    lx2 = x1 + tw + 6
    ly2 = y1

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Text
    cv2.putText(
        frame, label,
        (lx1 + 3, ly2 - baseline - 1),
        font, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def draw_detections(
    frame: np.ndarray,
    detections: List[dict],
    class_names: Optional[List[str]] = None,
    palette: Optional[List[Tuple]] = None,
    conf_threshold: float = 0.0,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw all detections on a frame.

    Each detection dict must have: x1, y1, x2, y2, conf, class_id.
    Optionally: track_id.
    """
    for det in detections:
        if det["conf"] < conf_threshold:
            continue
        cid   = det["class_id"]
        color = get_color(cid, palette)
        name  = (class_names[cid] if class_names and cid < len(class_names)
                 else f"cls{cid}")
        label = f"#{det['track_id']} " if "track_id" in det else ""
        label += f"{name} {det['conf']:.2f}"
        draw_single_box(
            frame,
            int(det["x1"]), int(det["y1"]),
            int(det["x2"]), int(det["y2"]),
            label, color, thickness,
        )
    return frame


def draw_gt_boxes(
    frame: np.ndarray,
    gt_objects: List,          # list of KITTIObject
    class_names: Optional[List[str]] = None,
    palette: Optional[List[Tuple]] = None,
    thickness: int = 1,
    dashed: bool = True,
) -> np.ndarray:
    """
    Overlay ground-truth boxes (dashed / thinner to distinguish from predictions).
    """
    for obj in gt_objects:
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        color = get_color(obj.class_id, palette)
        name  = (class_names[obj.class_id]
                 if class_names and obj.class_id < len(class_names)
                 else obj.type)
        if dashed:
            _draw_dashed_rect(frame, x1, y1, x2, y2, color, thickness)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"GT:{name}", (x1 + 2, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame


def _draw_dashed_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    """Draw a dashed rectangle (used for GT boxes)."""
    pts = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in pts:
        length = int(np.hypot(ex - sx, ey - sy))
        if length == 0:
            continue
        dx, dy = (ex - sx) / length, (ey - sy) / length
        pos = 0
        drawing = True
        while pos < length:
            seg = dash_len if drawing else gap_len
            end = min(pos + seg, length)
            if drawing:
                p1 = (int(sx + dx * pos), int(sy + dy * pos))
                p2 = (int(sx + dx * end), int(sy + dy * end))
                cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
            pos = end
            drawing = not drawing
