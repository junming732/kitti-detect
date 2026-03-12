"""
scripts/visualize_video.py

Draw bounding box predictions on video frames (or image sequences).
Supports both YOLOv8 and DETR models.

Features:
    - Color-coded boxes per class
    - Confidence scores
    - FPS counter
    - Optional: ByteTrack multi-object tracking
    - Saves to MP4 or displays live

Usage:
    # YOLOv8 on video file
    python scripts/visualize_video.py \
        --model yolo \
        --weights runs/yolo/kitti/weights/best.pt \
        --input /path/to/video_or_image_folder \
        --output results/yolo_output.mp4 \
        --conf 0.4

    # Minimal (uses all defaults)
    python scripts/visualize_video.py --model yolo --input /path/to/input

    # DETR on image folder
    python scripts/visualize_video.py \
        --model detr \
        --weights runs/detr/kitti/best.pt \
        --input data/kitti_frames/ \
        --output results/detr_output.mp4 \
        --conf 0.5

    # Live webcam (device=0)
    python scripts/visualize_video.py \
        --model yolo \
        --weights runs/yolo/kitti/weights/best.pt \
        --input 0 \
        --live
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.kitti_parser import CLASS_COLORS, CLASS_NAMES


# -----------------------------------------------------------------------------
# Drawing Utilities
# -----------------------------------------------------------------------------

def draw_bbox(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    class_id: int,
    conf: float,
    track_id: Optional[int] = None,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw a single bounding box with label on a frame.

    Args:
        frame:     BGR image (H, W, 3)
        x1,y1:    Top-left corner
        x2,y2:    Bottom-right corner
        class_id: Integer class index
        conf:     Detection confidence [0, 1]
        track_id: Optional track ID for multi-object tracking
        thickness: Box line thickness
        font_scale: Label font size

    Returns:
        Annotated frame (in-place modification + return)
    """
    color = CLASS_COLORS.get(class_id, (200, 200, 200))  # BGR
    name  = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"

    # Build label string
    label = f"{name} {conf:.2f}"
    if track_id is not None:
        label = f"#{track_id} {label}"

    # Draw filled rectangle for background
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_y = max(y1, text_h + baseline + 4)
    cv2.rectangle(
        frame,
        (x1, label_y - text_h - baseline - 4),
        (x1 + text_w + 4, label_y),
        color, -1  # Filled
    )

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw label text (white on colored background)
    cv2.putText(
        frame, label,
        (x1 + 2, label_y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),  # White text
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return frame


def draw_fps(frame: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)):
    """Overlay FPS counter on top-left corner."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
    )
    return frame


def draw_frame_info(frame: np.ndarray, frame_idx: int, total_frames: int, n_detections: int):
    """Draw frame counter and detection count at bottom."""
    h, w = frame.shape[:2]
    text = f"Frame {frame_idx}/{total_frames}  |  Detections: {n_detections}"
    cv2.putText(
        frame, text, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA
    )
    return frame


def draw_legend(frame: np.ndarray):
    """Draw class color legend in top-right corner."""
    h, w = frame.shape[:2]
    x_start = w - 160
    y_start = 15
    for i, name in enumerate(CLASS_NAMES):
        color = CLASS_COLORS.get(i, (200, 200, 200))
        y = y_start + i * 22
        cv2.rectangle(frame, (x_start, y), (x_start + 16, y + 14), color, -1)
        cv2.putText(
            frame, name, (x_start + 22, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return frame


# -----------------------------------------------------------------------------
# Model Wrappers
# -----------------------------------------------------------------------------

class YOLODetector:
    """YOLOv8 inference wrapper."""

    def __init__(self, weights: str, conf: float = 0.4, iou: float = 0.5, device: str = "0"):
        from ultralytics import YOLO
        print(f"[YOLO] Loading weights: {weights}")
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device

    def predict(self, frame: np.ndarray) -> List[dict]:
        """
        Run inference on a BGR frame.
        Returns list of dicts: {x1, y1, x2, y2, conf, class_id}
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                })
        return detections


class DETRDetector:
    """DETR inference wrapper using HuggingFace Transformers."""

    def __init__(self, weights: str, conf: float = 0.5, device: str = "cuda"):
        from transformers import DetrForObjectDetection, DetrImageProcessor
        print(f"[DETR] Loading weights: {weights}")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf = conf

        checkpoint = torch.load(weights, map_location=self.device)
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=len(CLASS_NAMES),
            ignore_mismatched_sizes=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> List[dict]:
        """Run DETR inference on a BGR frame."""
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        h, w = frame.shape[:2]

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.conf, target_sizes=[(h, w)]
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": float(score),
                "class_id": int(label),
            })
        return detections


# -----------------------------------------------------------------------------
# Video Pipeline
# -----------------------------------------------------------------------------

def get_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def process_video(
    detector,
    input_path: str,
    output_path: Optional[str] = None,
    live: bool = False,
    max_frames: Optional[int] = None,
    show_legend: bool = True,
) -> dict:
    """
    Main video processing loop.

    Args:
        detector:     YOLODetector or DETRDetector instance
        input_path:   Path to input video, image folder, or webcam index
        output_path:  Path to save output video (None = no save)
        live:         Show live window during processing
        max_frames:   Limit number of frames (for testing)
        show_legend:  Draw class color legend

    Returns:
        dict with stats: total_frames, avg_fps, total_detections
    """
    # -- Open video source -----------------------------------------------------
    is_folder = Path(input_path).is_dir() if not input_path.isdigit() else False

    if is_folder:
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        frames_paths = sorted([
            str(p) for p in Path(input_path).iterdir()
            if p.suffix.lower() in exts
        ])
        total_frames = len(frames_paths) if not max_frames else min(len(frames_paths), max_frames)
        fps_in = 10.0  # Default for image folders
        frame_iter = (cv2.imread(p) for p in frames_paths[:total_frames])
    else:
        cap = cv2.VideoCapture(int(input_path) if input_path.isdigit() else input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)
        fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_iter = iter(lambda: cap.read()[1], None)

    # -- Setup output writer ---------------------------------------------------
    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Get dimensions from first frame
        first_frame = next(frame_iter) if is_folder else None
        if first_frame is not None:
            h, w = first_frame.shape[:2]
        else:
            w, h = width, height
        writer = get_video_writer(output_path, fps_in, w, h)
        print(f"[Visualize] Writing to: {output_path}")

    # -- Processing loop -------------------------------------------------------
    frame_count = 0
    total_detections = 0
    fps_times = []

    pbar = tqdm(total=total_frames, desc="Processing frames")

    def process_frame(frame):
        nonlocal frame_count, total_detections

        t_start = time.perf_counter()

        # Run detector
        detections = detector.predict(frame)

        # Annotate frame
        for det in detections:
            draw_bbox(
                frame,
                det["x1"], det["y1"], det["x2"], det["y2"],
                det["class_id"], det["conf"],
            )

        # Overlays
        t_end = time.perf_counter()
        inference_fps = 1.0 / max(t_end - t_start, 1e-6)
        fps_times.append(inference_fps)
        avg_fps = sum(fps_times[-30:]) / len(fps_times[-30:])  # Rolling 30-frame avg

        draw_fps(frame, avg_fps)
        draw_frame_info(frame, frame_count, total_frames, len(detections))
        if show_legend:
            draw_legend(frame)

        total_detections += len(detections)
        frame_count += 1
        pbar.update(1)
        return frame

    # Handle first frame for folders (already consumed)
    if is_folder:
        frame_paths = sorted([
            str(p) for p in Path(input_path).iterdir()
            if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ])[:total_frames]
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            annotated = process_frame(frame)
            if writer:
                writer.write(annotated)
            if live:
                cv2.imshow("KITTI Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    else:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            annotated = process_frame(frame)
            if writer:
                writer.write(annotated)
            if live:
                cv2.imshow("KITTI Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()

    pbar.close()
    if writer:
        writer.release()
    if live:
        cv2.destroyAllWindows()

    avg_fps = sum(fps_times) / len(fps_times) if fps_times else 0.0
    stats = {
        "total_frames": frame_count,
        "avg_fps": avg_fps,
        "total_detections": total_detections,
        "avg_detections_per_frame": total_detections / max(frame_count, 1),
    }
    print(f"\n-- Stats ------------------------------------")
    print(f"  Frames processed: {stats['total_frames']}")
    print(f"  Avg FPS:          {stats['avg_fps']:.1f}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Avg per frame:    {stats['avg_detections_per_frame']:.2f}")
    print(f"---------------------------------------------")
    return stats


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize KITTI object detection on video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",   type=str, required=True, choices=["yolo", "detr"],
                        help="Model type: yolo or detr")
    parser.add_argument("--weights", type=str,
                        default="runs/yolo/kitti/weights/best.pt",
                        help="Path to trained weights")
    parser.add_argument("--input",   type=str, required=True,
                        help="Input: video path, image folder, or webcam index (0)")
    parser.add_argument("--output",  type=str, default="results/demo.mp4",
                        help="Output video path")
    parser.add_argument("--conf",    type=float, default=0.4,
                        help="Confidence threshold [0-1]")
    parser.add_argument("--iou",     type=float, default=0.5,
                        help="NMS IoU threshold (YOLO only)")
    parser.add_argument("--device",  type=str, default="0",
                        help="Device: 0 (GPU), cpu")
    parser.add_argument("--live",    action="store_true",
                        help="Show live display window")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames (for testing)")
    parser.add_argument("--no-legend", action="store_true",
                        help="Hide class color legend")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load detector
    if args.model == "yolo":
        detector = YOLODetector(args.weights, conf=args.conf, iou=args.iou, device=args.device)
    elif args.model == "detr":
        device = "cuda" if args.device != "cpu" else "cpu"
        detector = DETRDetector(args.weights, conf=args.conf, device=device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Run
    stats = process_video(
        detector=detector,
        input_path=args.input,
        output_path=args.output,
        live=args.live,
        max_frames=args.max_frames,
        show_legend=not args.no_legend,
    )

    if args.output:
        print(f"\n[OK] Output saved to: {args.output}")