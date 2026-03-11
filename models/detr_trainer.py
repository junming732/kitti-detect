"""
models/detr_trainer.py

Fine-tune Facebook's DETR (DEtection TRansformer) on KITTI.
Uses HuggingFace Transformers for easy loading of pre-trained weights.

Usage:
    python models/detr_trainer.py --epochs 30 --batch 8 --lr 1e-4

Reference:
    "End-to-End Object Detection with Transformers" (Carion et al., 2020)
    https://arxiv.org/abs/2005.12872
"""

import argparse
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
)
from tqdm import tqdm
import sys as _sys
tqdm = lambda *a, **kw: __import__("tqdm").tqdm(*a, **{**kw, "file": _sys.stderr})

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.kitti_dataset import KITTIDatasetDETR, collate_fn_detr, CLASS_NAMES
from utils.metrics import compute_map


# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    """Load a YAML config file and return as a flat dict."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Flatten class_names list to a joined string for display only
    cfg.pop("class_names", None)
    cfg.pop("notes", None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DETR on KITTI")
    parser.add_argument("--config", type=str,
                        default="configs/detr_kitti.yaml",
                        help="Path to YAML config. CLI args override YAML values.")
    parser.add_argument("--model-name",   type=str,   default=None)
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch",        type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--lr-backbone",  type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--imgsz",        type=int,   default=None)
    parser.add_argument("--workers",      type=int,   default=None)
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--output-dir",   type=str,   default=None)
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Load YAML defaults
    import os
    cfg = {}
    if args.config and os.path.exists(args.config):
        cfg = load_yaml_config(args.config)
        print(f"[DETR] Loaded config: {args.config}")
    else:
        print(f"[DETR] Config not found: {args.config} -- using defaults")

    # YAML key -> argparse attr mapping
    yaml_to_arg = {
        "model_name":    "model_name",
        "epochs":        "epochs",
        "batch":         "batch",
        "lr":            "lr",
        "lr_backbone":   "lr_backbone",
        "weight_decay":  "weight_decay",
        "imgsz":         "imgsz",
        "num_workers":   "workers",
        "device":        "device",
        "output_dir":    "output_dir",
        "amp":           "amp",
        "grad_clip":     "grad_clip",
        "conf_threshold":"conf_threshold",
        "iou_threshold": "iou_threshold",
        "save_every":    "save_every",
        "warmup_epochs": "warmup_epochs",
    }

    # Apply YAML values as defaults (CLI args take priority)
    for yaml_key, arg_key in yaml_to_arg.items():
        if yaml_key in cfg:
            if getattr(args, arg_key, None) is None:
                setattr(args, arg_key, cfg[yaml_key])

    # Hard defaults for anything still None
    defaults = {
        "model_name":    "facebook/detr-resnet-50",
        "epochs":        30,
        "batch":         8,
        "lr":            1e-4,
        "lr_backbone":   1e-5,
        "weight_decay":  1e-4,
        "imgsz":         640,
        "workers":       4,
        "device":        "cuda",
        "output_dir":    "runs/detr/kitti",
        "amp":           True,
        "grad_clip":     0.1,
        "conf_threshold":0.5,
        "iou_threshold": 0.5,
        "save_every":    5,
        "warmup_epochs": 2,
    }
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    return args


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

class DETRTrainer:
    def __init__(self, args):
        self.args = args
        # Normalise device string: "0" -> "cuda:0", "cuda" -> "cuda"
        device_str = args.device
        if device_str.isdigit():
            device_str = f"cuda:{device_str}"
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DETR] Using device: {self.device}")
        self._build_model()
        self._build_datasets()
        self._build_optimizers()

    def _build_model(self):
        args = self.args
        print(f"[DETR] Loading pre-trained model: {args.model_name}")

        # Load image processor
        self.processor = DetrImageProcessor.from_pretrained(args.model_name)

        # Load model ? we REPLACE the classification head for KITTI classes
        self.model = DetrForObjectDetection.from_pretrained(
            args.model_name,
            num_labels=len(CLASS_NAMES),
            ignore_mismatched_sizes=True,   # Allows new head size
        )
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[DETR] Parameters: {total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    def _build_datasets(self):
        args = self.args
        print("[DETR] Building datasets...")

        self.train_dataset = KITTIDatasetDETR(
            split="train",
            img_size=args.imgsz,
        )
        self.val_dataset = KITTIDatasetDETR(
            split="val",
            img_size=args.imgsz,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_fn_detr,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn_detr,
            pin_memory=True,
        )
        print(f"[DETR] Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def _build_optimizers(self):
        args = self.args

        # Separate LRs for backbone vs transformer head (standard DETR practice)
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if "backbone" in n and p.requires_grad
        ]
        other_params = [
            p for n, p in self.model.named_parameters()
            if "backbone" not in n and p.requires_grad
        ]

        self.optimizer = AdamW([
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": other_params,    "lr": args.lr},
        ], weight_decay=args.weight_decay)

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6
        )
        self.scaler = torch.amp.GradScaler("cuda")  # Mixed precision

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_bbox = 0.0
        total_loss_giou = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for step, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast():
                outputs = self.model(
                    pixel_values=images,
                    labels=targets,
                )
                loss = outputs.loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping (important for transformers)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            loss_dict = outputs.loss_dict
            total_loss_ce   += loss_dict.get("loss_ce", 0)
            total_loss_bbox += loss_dict.get("loss_bbox", 0)
            total_loss_giou += loss_dict.get("loss_giou", 0)

            # Update tqdm every 10 steps to avoid WandB stdout conflict under SLURM
            if step % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{loss_dict.get('loss_ce', 0):.3f}",
                    "bbox": f"{loss_dict.get('loss_bbox', 0):.3f}",
                    "giou": f"{loss_dict.get('loss_giou', 0):.3f}",
                })

        n = len(self.train_loader)
        return {
            "train/loss": total_loss / n,
            "train/loss_ce": total_loss_ce / n,
            "train/loss_bbox": total_loss_bbox / n,
            "train/loss_giou": total_loss_giou / n,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []

        for images, targets in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            targets_gpu = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast():
                outputs = self.model(pixel_values=images, labels=targets_gpu)
                val_loss += outputs.loss.item()

            # Post-process predictions for mAP calculation
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=0.5,
                target_sizes=[(self.args.imgsz, self.args.imgsz)] * len(images),
            )
            all_predictions.extend(results)
            all_targets.extend(targets)

        # mAP calculation
        try:
            # Convert DETR list format ? dict format expected by compute_map
            # predictions: {img_id: [{"box": [x1,y1,x2,y2], "conf": float, "class_id": int}]}
            # ground_truths: {img_id: [{"box": [x1,y1,x2,y2], "class_id": int}]}
            pred_dict = {}
            gt_dict   = {}
            for img_id, (pred, target) in enumerate(zip(all_predictions, all_targets)):
                pred_dict[img_id] = [
                    {
                        "box":      [float(box[0]), float(box[1]),
                                     float(box[2]), float(box[3])],
                        "conf":     float(score),
                        "class_id": int(label),
                    }
                    for box, score, label in zip(
                        pred["boxes"], pred["scores"], pred["labels"]
                    )
                ]
                boxes  = target["boxes"]   # (N, 4) tensor in cxcywh normalised
                labels = target["class_labels"]
                H = W  = self.args.imgsz   # convert cxcywh norm -> xyxy pixels
                gt_dict[img_id] = []
                for box, label in zip(boxes, labels):
                    cx, cy, w, h = box.tolist()
                    x1 = (cx - w / 2) * W
                    y1 = (cy - h / 2) * H
                    x2 = (cx + w / 2) * W
                    y2 = (cy + h / 2) * H
                    gt_dict[img_id].append({
                        "box":      [x1, y1, x2, y2],
                        "class_id": int(label),
                    })
            map_score, _ = compute_map(pred_dict, gt_dict, num_classes=len(CLASS_NAMES))
        except Exception as e:
            print(f"[DETR] mAP computation failed: {e}")
            map_score = 0.0

        return {
            "val/loss": val_loss / len(self.val_loader),
            "val/mAP50": map_score,
        }

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "args": vars(self.args),
        }
        path = self.output_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"  [OK] New best model saved ? {best_path}")

        return path

    def train(self):
        args = self.args
        best_map = 0.0
        history = []

        print(f"\n[DETR] Starting training: {args.epochs} epochs\n{'-'*50}")

        for epoch in range(args.epochs):
            t0 = time.time()

            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Scheduler step
            self.scheduler.step()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics["epoch"] = epoch + 1
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            metrics["time"] = time.time() - t0
            history.append(metrics)

            # Print summary
            print(f"\nEpoch {epoch+1:3d}/{args.epochs}")
            print(f"  Train Loss: {metrics['train/loss']:.4f}")
            print(f"  Val Loss:   {metrics['val/loss']:.4f}")
            print(f"  Val mAP@50: {metrics['val/mAP50']:.4f}")
            print(f"  Time:       {metrics['time']:.1f}s")

            # Save checkpoint
            is_best = metrics["val/mAP50"] > best_map
            if is_best:
                best_map = metrics["val/mAP50"]
            self.save_checkpoint(epoch, metrics, is_best=is_best)

        # Save full history to JSON (used by plot_training.py)
        import json

        def to_python(obj):
            """Recursively convert tensors/numpy to plain Python types."""
            import numpy as np
            if isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_python(v) for v in obj]
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            return obj

        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(to_python(history), f, indent=2)
        print(f"   History saved: {history_path}")

        print(f"\n{'='*50}")
        print(f"[OK] Training complete! Best mAP@50: {best_map:.4f}")
        print(f"   Best weights: {self.output_dir / 'best.pt'}")

        # Auto-generate training plots
        print("\n[INFO] Generating training plots...")
        try:
            project_root = Path(__file__).resolve().parent.parent
            scripts_dir  = str(project_root / "scripts")
            print(f"[INFO] Adding to path: {scripts_dir}")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from plot_training import (
                load_detr_history,
                plot_training_dashboard_detr,
                plot_loss_curves_detr,
                plot_map_curves_detr,
            )
            h = load_detr_history(str(self.output_dir))
            plot_training_dashboard_detr(h)
            plot_loss_curves_detr(h)
            plot_map_curves_detr(h)
            print("[INFO] Plots saved to results/plots/")
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

        return history


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    trainer = DETRTrainer(args)
    trainer.train()