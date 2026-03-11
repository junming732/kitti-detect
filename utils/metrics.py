"""
utils/metrics.py

Full evaluation metrics for object detection:
  - Vectorized IoU
  - Per-class AP (101-point interpolation, VOC/COCO style)
  - mAP@0.5 and mAP@0.5:0.95
  - Precision / Recall / F1
  - Confusion matrix
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def box_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-9)


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    return float(box_iou_batch(box1[None], box2[None])[0, 0])


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    thresholds = np.linspace(0, 1, 101)
    ap = np.mean([np.max(mpre[mrec >= t]) if np.any(mrec >= t) else 0.0 for t in thresholds])
    return float(ap)


def compute_per_class_ap(predictions, ground_truths, class_id, iou_threshold=0.5):
    all_preds = []
    for img_id, preds in predictions.items():
        for p in preds:
            if p["class_id"] == class_id:
                all_preds.append({"img_id": img_id, **p})
    all_preds.sort(key=lambda x: x["conf"], reverse=True)

    n_gt = sum(sum(1 for gt in gts if gt["class_id"] == class_id) for gts in ground_truths.values())
    if n_gt == 0:
        return 0.0, np.array([]), np.array([])
    if not all_preds:
        return 0.0, np.array([0.0]), np.array([0.0])

    matched = defaultdict(set)
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for i, pred in enumerate(all_preds):
        img_id   = pred["img_id"]
        pred_box = np.array(pred["box"], dtype=float)
        gt_list  = [gt for gt in ground_truths.get(img_id, []) if gt["class_id"] == class_id]
        best_iou, best_j = iou_threshold - 1e-9, -1
        for j, gt in enumerate(gt_list):
            if j in matched[img_id]:
                continue
            iou = box_iou(pred_box, np.array(gt["box"], dtype=float))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            tp[i] = 1
            matched[img_id].add(best_j)
        else:
            fp[i] = 1

    cum_tp    = np.cumsum(tp)
    cum_fp    = np.cumsum(fp)
    recall    = cum_tp / (n_gt + 1e-9)
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)
    return _interpolated_ap(recall, precision), recall, precision


def compute_map(predictions, ground_truths, num_classes, iou_threshold=0.5, class_names=None):
    per_class = {}
    for c in range(num_classes):
        name = class_names[c] if class_names and c < len(class_names) else f"cls{c}"
        ap, _, _ = compute_per_class_ap(predictions, ground_truths, c, iou_threshold)
        per_class[name] = ap
    return float(np.mean(list(per_class.values()))), per_class


def compute_map_range(predictions, ground_truths, num_classes, iou_thresholds=None, class_names=None):
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    per_class_aps = defaultdict(list)
    for thr in iou_thresholds:
        _, cls_ap = compute_map(predictions, ground_truths, num_classes, thr, class_names)
        for name, ap in cls_ap.items():
            per_class_aps[name].append(ap)
    per_class_mean = {name: float(np.mean(aps)) for name, aps in per_class_aps.items()}
    return float(np.mean(list(per_class_mean.values()))), per_class_mean


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return float(precision), float(recall), float(f1)


def build_confusion_matrix(predictions, ground_truths, num_classes, iou_threshold=0.5, conf_threshold=0.25):
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    for img_id, gt_list in ground_truths.items():
        pred_list = [p for p in predictions.get(img_id, []) if p["conf"] >= conf_threshold]
        gt_boxes  = np.array([gt["box"] for gt in gt_list], dtype=float) if gt_list else np.zeros((0, 4))
        gt_labels = np.array([gt["class_id"] for gt in gt_list], dtype=int)
        pd_boxes  = np.array([p["box"] for p in pred_list], dtype=float) if pred_list else np.zeros((0, 4))
        pd_labels = np.array([p["class_id"] for p in pred_list], dtype=int)
        if len(gt_boxes) == 0 and len(pd_boxes) == 0:
            continue
        if len(gt_boxes) > 0 and len(pd_boxes) > 0:
            iou_mat = box_iou_batch(pd_boxes, gt_boxes)
            matched_gt = set()
            for pi in range(len(pd_boxes)):
                best_gt = int(np.argmax(iou_mat[pi]))
                if iou_mat[pi, best_gt] >= iou_threshold and best_gt not in matched_gt:
                    matrix[pd_labels[pi], gt_labels[best_gt]] += 1
                    matched_gt.add(best_gt)
                else:
                    matrix[pd_labels[pi], num_classes] += 1
            for gi in range(len(gt_boxes)):
                if gi not in matched_gt:
                    matrix[num_classes, gt_labels[gi]] += 1
        elif len(gt_boxes) > 0:
            for gl in gt_labels:
                matrix[num_classes, gl] += 1
        else:
            for pl in pd_labels:
                matrix[pl, num_classes] += 1
    return matrix