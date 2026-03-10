"""
scripts/evaluate.py

Compute mAP, per-class AP, precision, recall for YOLO or DETR on KITTI val set.

Usage:
    python scripts/evaluate.py \
        --model yolo \
        --weights runs/yolo/kitti/weights/best.pt \
        --images-val data/kitti/training/image_2 \
        --labels-val data/kitti/training/label_2 \
        --conf 0.25 \
        --iou 0.5
"""

import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.kitti_parser import CLASS_NAMES, parse_kitti_split


# ─────────────────────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────────────────────

def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# AP computation (11-point interpolation and area under PR curve)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision (area under P-R curve via 101-point interpolation)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # 101-point interpolation
    thresholds = np.linspace(0, 1, 101)
    ap = np.mean([np.max(mpre[mrec >= t]) if any(mrec >= t) else 0.0 for t in thresholds])
    return float(ap)


def compute_per_class_ap(
    predictions: Dict[str, List[dict]],
    ground_truths: Dict[str, List[dict]],
    class_id: int,
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute AP for a single class.

    Args:
        predictions:   {image_id: [{box, conf, class_id}]}
        ground_truths: {image_id: [{box, class_id}]}
        class_id:      Class to evaluate
        iou_threshold: IoU threshold for TP/FP

    Returns:
        AP score for this class
    """
    # Collect all predictions for this class, sorted by confidence
    all_preds = []
    for img_id, preds in predictions.items():
        for p in preds:
            if p["class_id"] == class_id:
                all_preds.append({"img_id": img_id, **p})
    all_preds.sort(key=lambda x: x["conf"], reverse=True)

    # Count total ground truth instances for this class
    n_gt = sum(
        sum(1 for gt in gts if gt["class_id"] == class_id)
        for gts in ground_truths.values()
    )
    if n_gt == 0:
        return 0.0

    # Track which GTs have been matched
    matched = defaultdict(set)  # img_id -> set of matched gt indices
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for i, pred in enumerate(all_preds):
        img_id = pred["img_id"]
        pred_box = np.array([pred["x1"], pred["y1"], pred["x2"], pred["y2"]])
        gt_list = [gt for gt in ground_truths.get(img_id, []) if gt["class_id"] == class_id]

        best_iou = iou_threshold - 1e-9
        best_idx = -1
        for j, gt in enumerate(gt_list):
            if j in matched[img_id]:
                continue
            gt_box = np.array(gt["box"])
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_idx >= 0:
            tp[i] = 1
            matched[img_id].add(best_idx)
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall    = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)

    return compute_ap(recall, precision)


def compute_map(
    predictions: Dict[str, List[dict]],
    ground_truths: Dict[str, List[dict]],
    iou_threshold: float = 0.5,
    num_classes: int = len(CLASS_NAMES),
) -> Tuple[float, Dict[str, float]]:
    """Compute mAP and per-class AP."""
    per_class = {}
    for c in range(num_classes):
        ap = compute_per_class_ap(predictions, ground_truths, c, iou_threshold)
        per_class[CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls{c}"] = ap
    map_score = float(np.mean(list(per_class.values())))
    return map_score, per_class


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class KITTIEvaluator:
    def __init__(self, images_dir: str, labels_dir: str, conf: float = 0.25, iou: float = 0.5):
        self.conf = conf
        self.iou_threshold = iou
        self.samples = parse_kitti_split(images_dir, labels_dir)

        # Build GT dictionary
        self.ground_truths = {}
        for s in self.samples:
            self.ground_truths[s["image_id"]] = [
                {"box": list(obj.bbox), "class_id": obj.class_id}
                for obj in s["objects"]
            ]
        print(f"[Evaluator] {len(self.samples)} validation images loaded.")

    def run_yolo(self, weights: str, device: str = "0") -> dict:
        from ultralytics import YOLO
        model = YOLO(weights)
        predictions = {}

        for sample in tqdm(self.samples, desc="YOLO inference"):
            results = model.predict(
                sample["image_path"],
                conf=self.conf, iou=self.iou_threshold,
                device=device, verbose=False,
            )
            preds = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    preds.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                    })
            predictions[sample["image_id"]] = preds

        return predictions

    def run_detr(self, weights: str, device: str = "cuda") -> dict:
        from transformers import DetrForObjectDetection, DetrImageProcessor
        from PIL import Image as PILImage

        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        checkpoint = torch.load(weights, map_location=dev)
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=len(CLASS_NAMES),
            ignore_mismatched_sizes=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(dev).eval()

        predictions = {}
        with torch.no_grad():
            for sample in tqdm(self.samples, desc="DETR inference"):
                img = PILImage.open(sample["image_path"]).convert("RGB")
                h, w = img.size[1], img.size[0]
                inputs = {k: v.to(dev) for k, v in
                          processor(images=img, return_tensors="pt").items()}
                outputs = model(**inputs)
                results = processor.post_process_object_detection(
                    outputs, threshold=self.conf, target_sizes=[(h, w)]
                )[0]
                preds = []
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    preds.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(score),
                        "class_id": int(label),
                    })
                predictions[sample["image_id"]] = preds
        return predictions

    def evaluate(self, predictions: dict) -> dict:
        map_score, per_class = compute_map(
            predictions, self.ground_truths, self.iou_threshold
        )

        print(f"\n{'='*52}")
        print(f"  KITTI Evaluation — IoU@{self.iou_threshold}")
        print(f"{'='*52}")
        for cls_name, ap in per_class.items():
            print(f"  {cls_name:<20}: AP = {ap:.4f}")
        print(f"{'─'*52}")
        print(f"  {'mAP':<20}: {map_score:.4f}")
        print(f"{'='*52}\n")

        return {"mAP": map_score, "per_class_AP": per_class}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KITTI object detection models")
    parser.add_argument("--model",       type=str, required=True, choices=["yolo", "detr"])
    parser.add_argument("--weights",     type=str, required=True)
    parser.add_argument("--images-val",  type=str, required=True)
    parser.add_argument("--labels-val",  type=str, required=True)
    parser.add_argument("--conf",        type=float, default=0.25)
    parser.add_argument("--iou",         type=float, default=0.5)
    parser.add_argument("--device",      type=str, default="0")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = KITTIEvaluator(
        args.images_val, args.labels_val,
        conf=args.conf, iou=args.iou,
    )

    if args.model == "yolo":
        preds = evaluator.run_yolo(args.weights, args.device)
    else:
        device = "cuda" if args.device != "cpu" else "cpu"
        preds = evaluator.run_detr(args.weights, device)

    results = evaluator.evaluate(preds)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_json}")