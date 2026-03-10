"""
scripts/inference.py

Quick inference on a single image or video clip.
Prints detections and optionally saves annotated output.

Usage:
    python scripts/inference.py \
        --model yolo \
        --weights runs/yolo/kitti/weights/best.pt \
        --input path/to/image.png \
        --show
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.visualize_video import YOLODetector, DETRDetector, draw_bbox, draw_legend, CLASS_NAMES


def run_image_inference(detector, image_path: str, output_path: str = None, show: bool = False):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detections = detector.predict(frame)
    print(f"\nDetections in {Path(image_path).name}:")
    for i, det in enumerate(detections):
        name = CLASS_NAMES[det['class_id']] if det['class_id'] < len(CLASS_NAMES) else f"cls{det['class_id']}"
        print(f"  [{i+1}] {name:15} conf={det['conf']:.3f}  "
              f"box=({det['x1']},{det['y1']}) → ({det['x2']},{det['y2']})")

    for det in detections:
        draw_bbox(frame, det['x1'], det['y1'], det['x2'], det['y2'],
                  det['class_id'], det['conf'])
    draw_legend(frame)

    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"\nSaved to: {output_path}")

    if show:
        cv2.imshow("KITTI Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, required=True, choices=["yolo", "detr"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input",   type=str, required=True)
    parser.add_argument("--output",  type=str, default=None)
    parser.add_argument("--conf",    type=float, default=0.4)
    parser.add_argument("--device",  type=str, default="0")
    parser.add_argument("--show",    action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model == "yolo":
        detector = YOLODetector(args.weights, conf=args.conf, device=args.device)
    else:
        device = "cuda" if args.device != "cpu" else "cpu"
        detector = DETRDetector(args.weights, conf=args.conf, device=device)

    run_image_inference(detector, args.input, args.output, args.show)