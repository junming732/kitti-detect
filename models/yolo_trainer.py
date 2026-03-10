"""
models/yolo_trainer.py

Fine-tune YOLOv8 on the KITTI dataset.

Usage:
    python models/yolo_trainer.py --epochs 50 --batch 16 --model yolov8m.pt

Requirements:
    pip install ultralytics wandb
"""

import argparse
import os
import sys
from pathlib import Path

# ── Optional: WandB experiment tracking ───────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on KITTI")
    parser.add_argument("--model",      type=str, default="yolov8m.pt",
                        help="Model variant: yolov8n/s/m/l/x.pt OR path to checkpoint")
    parser.add_argument("--data",       type=str, default="configs/yolo_kitti.yaml",
                        help="Path to dataset YAML")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch",      type=int, default=16)
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--device",     type=str, default="0",
                        help="CUDA device(s) or 'cpu'")
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--lr0",        type=float, default=0.001)
    parser.add_argument("--project",    type=str, default="runs/yolo")
    parser.add_argument("--name",       type=str, default="kitti")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--no-wandb",   action="store_true",
                        help="Disable WandB logging")
    return parser.parse_args()


def train(args):
    # ── WandB Setup ───────────────────────────────────────────────────────────
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project="kitti-object-detection",
            name=f"yolov8-{args.name}",
            config=vars(args),
        )
        print("[INFO] WandB logging enabled.")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # ── Load Model ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"[INFO] Starting training for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Augmentation tuned for driving
        fliplr=0.5,
        flipud=0.0,        # No vertical flip for road scenes
        mosaic=1.0,
        mixup=0.1,
        degrees=0.0,       # No rotation
        # Training quality
        amp=True,
        patience=15,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    return results


def evaluate(args, weights_path: str):
    """Run validation and print metrics."""
    print(f"\n[INFO] Evaluating model: {weights_path}")
    model = YOLO(weights_path)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)

    print("\n── Validation Metrics ──────────────────────────────")
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")

    per_class = metrics.box.ap_class_index
    if per_class is not None:
        from ultralytics.utils.metrics import ap_per_class
        print("\n  Per-class AP@50:")
        for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            from data.kitti_dataset import CLASS_NAMES
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
            print(f"    {name:<15}: {ap:.4f}")
    print("─" * 50)
    return metrics


def export_model(weights_path: str, format: str = "onnx"):
    """Export trained model to ONNX or TensorRT for deployment."""
    print(f"[INFO] Exporting to {format.upper()}...")
    model = YOLO(weights_path)
    export_path = model.export(format=format, imgsz=640, dynamic=True)
    print(f"[INFO] Exported to: {export_path}")
    return export_path


if __name__ == "__main__":
    args = parse_args()

    # Ensure we run from project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Train
    results = train(args)

    # Auto-evaluate with best weights
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        evaluate(args, str(best_weights))

        # Export to ONNX for deployment
        export_model(str(best_weights), format="onnx")

    print("\n Training complete!")
    print(f"   Best weights: {best_weights}")
    print(f"   Results:      {Path(args.project) / args.name}")