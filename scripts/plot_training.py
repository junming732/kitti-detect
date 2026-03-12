"""
scripts/plot_training.py

Generate all training visualisation plots for kitti-detect.
Reads from:
  - YOLOv8  results CSV  (runs/yolo/<name>/results.csv)
  - DETR    history JSON (runs/detr/<name>/history.json)

Produces (saved to results/plots/):
  1. loss_curves.png          ? train + val losses per epoch
  2. map_curves.png           ? mAP@50 and mAP@50-95 over training
  3. pr_curves.png            ? Precision-Recall curves per class
  4. confusion_matrix.png     ? Normalised confusion matrix heat-map
  5. model_comparison.png     ? Side-by-side mAP / speed bar chart
  6. bbox_stats.png           ? GT bounding-box size & aspect-ratio distributions
  7. training_summary.png     ? Single-page dashboard (all panels combined)

Usage:
    # After YOLO training:
    python scripts/plot_training.py --model yolo --run-dir runs/yolo/kitti

    # After DETR training:
    python scripts/plot_training.py --model detr --run-dir runs/detr/kitti

    # Both + comparison (pass both --run-dir flags):
    python scripts/plot_training.py \
        --model both \
        --yolo-run-dir runs/yolo/kitti \
        --detr-run-dir runs/detr/kitti

    # Full evaluation plots (needs val data for PR + confusion):
    python scripts/plot_training.py \
        --model yolo \
        --run-dir runs/yolo/kitti \
        --images-val /proj/.../kitti/splits/val/image_2 \
        --labels-val /proj/.../kitti/splits/val/label_2 \
        --eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for HPC)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.kitti_parser import CLASS_NAMES, CLASS_COLORS, parse_kitti_folder
from utils.paths import RESULTS_DIR

# -- Style ---------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Class colour map (RGB 0-1 for matplotlib)
CLS_COLORS_MPL = {
    name: tuple(c / 255 for c in CLASS_COLORS.get(i, (150, 150, 150)))
    for i, name in enumerate(CLASS_NAMES)
}


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------

def load_yolo_results(run_dir: str) -> pd.DataFrame:
    """
    Load YOLOv8 results.csv produced by Ultralytics.
    Ultralytics auto-increments run dirs (kitti, kitti2, kitti3 ...).
    If results.csv is not found in run_dir directly, search the parent
    for the most recently modified matching directory.
    """
    run_path = Path(run_dir)
    csv_path = run_path / "results.csv"

    if not csv_path.exists():
        # Search parent dir for latest run with same base name
        parent   = run_path.parent
        base     = run_path.name
        candidates = sorted(
            [d for d in parent.iterdir()
             if d.is_dir() and d.name.startswith(base) and (d / "results.csv").exists()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            csv_path = candidates[0] / "results.csv"
            print(f"[plots] Auto-detected run dir: {candidates[0]}")
        else:
            raise FileNotFoundError(
                f"YOLO results CSV not found in {run_path} or any {base}* sibling. "
                f"Make sure training completed and the run dir is correct."
            )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"[plots] Loaded YOLO results: {len(df)} epochs from {csv_path}")
    return df


def load_detr_history(run_dir: str) -> List[dict]:
    """Load DETR training history JSON saved by detr_trainer.py."""
    json_path = Path(run_dir) / "history.json"
    if not json_path.exists():
        raise FileNotFoundError(f"DETR history JSON not found: {json_path}")
    with open(json_path) as f:
        history = json.load(f)
    print(f"[plots] Loaded DETR history: {len(history)} epochs from {json_path}")
    return history


# -----------------------------------------------------------------------------
# 1. Loss curves
# -----------------------------------------------------------------------------

def plot_loss_curves_yolo(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot YOLO train/val box, cls, dfl losses."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = df.get("epoch", range(len(df)))

    loss_pairs = [
        ("train/box_loss", "val/box_loss",  "Box Loss"),
        ("train/cls_loss", "val/cls_loss",  "Classification Loss"),
        ("train/dfl_loss", "val/dfl_loss",  "DFL Loss"),
    ]
    for ax, (train_col, val_col, title) in zip(axes, loss_pairs):
        if train_col in df:
            ax.plot(epochs, df[train_col], color="#E05C3A", lw=2, label="Train")
        if val_col in df:
            ax.plot(epochs, df[val_col],   color="#3A8BE0", lw=2, label="Val", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle("YOLOv8 ? Training & Validation Losses", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_path or str(PLOTS_DIR / "yolo_loss_curves.png"))
    return fig


def plot_loss_curves_detr(history: List[dict], save_path: Optional[str] = None) -> plt.Figure:
    """Plot DETR train losses (ce, bbox, giou) and val loss."""
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    epochs = df["epoch"]

    panels = [
        ("train/loss",       "val/loss",  "Total Loss",            "#E05C3A", "#3A8BE0"),
        ("train/loss_ce",    None,        "Classification (CE)",   "#E05C3A", None),
        ("train/loss_bbox",  None,        "BBox Regression (L1)",  "#9B59B6", None),
        ("train/loss_giou",  None,        "GIoU Loss",             "#27AE60", None),
    ]
    for ax, (train_col, val_col, title, tc, vc) in zip(axes, panels):
        if train_col in df:
            ax.plot(epochs, df[train_col], color=tc, lw=2, label="Train")
        if val_col and val_col in df:
            ax.plot(epochs, df[val_col], color=vc, lw=2, linestyle="--", label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle("DETR - Training Losses", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_path or str(PLOTS_DIR / "detr_loss_curves.png"))
    return fig


# -----------------------------------------------------------------------------
# 2. mAP curves
# -----------------------------------------------------------------------------

def plot_map_curves_yolo(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot mAP@50, mAP@50-95, precision, recall over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = df.get("epoch", range(len(df)))

    # mAP
    ax = axes[0]
    if "metrics/mAP50(B)" in df:
        ax.plot(epochs, df["metrics/mAP50(B)"],    color="#3A8BE0", lw=2.5, label="mAP@50")
    if "metrics/mAP50-95(B)" in df:
        ax.plot(epochs, df["metrics/mAP50-95(B)"], color="#E05C3A", lw=2.5, label="mAP@50-95", linestyle="--")
    ax.set_title("mAP over Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1)
    ax.legend()

    # Best epoch annotation
    if "metrics/mAP50(B)" in df:
        best_epoch = df["metrics/mAP50(B)"].idxmax()
        best_val   = df["metrics/mAP50(B)"].iloc[best_epoch]
        ax.annotate(
            f"Best: {best_val:.3f}",
            xy=(epochs.iloc[best_epoch], best_val),
            xytext=(10, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

    # Precision & Recall
    ax2 = axes[1]
    if "metrics/precision(B)" in df:
        ax2.plot(epochs, df["metrics/precision(B)"], color="#27AE60", lw=2.5, label="Precision")
    if "metrics/recall(B)" in df:
        ax2.plot(epochs, df["metrics/recall(B)"],    color="#9B59B6", lw=2.5, label="Recall", linestyle="--")
    ax2.set_title("Precision & Recall over Training")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1)
    ax2.legend()

    fig.suptitle("YOLOv8 ? Validation Metrics", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_path or str(PLOTS_DIR / "yolo_map_curves.png"))
    return fig


def plot_map_curves_detr(history: List[dict], save_path: Optional[str] = None) -> plt.Figure:
    df    = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = df["epoch"]

    ax = axes[0]
    if "val/mAP50" in df:
        ax.plot(epochs, df["val/mAP50"], color="#3A8BE0", lw=2.5, label="mAP@50")
        best_epoch = df["val/mAP50"].idxmax()
        best_val   = df["val/mAP50"].iloc[best_epoch]
        ax.annotate(
            f"Best: {best_val:.3f}",
            xy=(epochs.iloc[best_epoch], best_val),
            xytext=(10, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )
    ax.set_title("Validation mAP@50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1)
    ax.legend()

    ax2 = axes[1]
    ax2.plot(epochs, df["train/loss"], color="#E05C3A", lw=2, label="Train Loss")
    ax2.plot(epochs, df["val/loss"],   color="#3A8BE0", lw=2, linestyle="--", label="Val Loss")
    ax2.set_title("Loss vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig.suptitle("DETR - Validation Metrics", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_path or str(PLOTS_DIR / "detr_map_curves.png"))
    return fig


# -----------------------------------------------------------------------------
# 3. Precision-Recall curves (per class, requires eval run)
# -----------------------------------------------------------------------------

def plot_pr_curves(
    predictions: Dict,
    ground_truths: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Per-class PR curves + mAP@50 in legend."""
    from utils.metrics import compute_per_class_ap

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, cls_name in enumerate(CLASS_NAMES):
        ax = axes[i]
        ap, recall, precision = compute_per_class_ap(predictions, ground_truths, i, 0.5)
        color = CLS_COLORS_MPL.get(cls_name, (0.5, 0.5, 0.5))

        if len(recall) > 0:
            ax.plot(recall, precision, color=color, lw=2.5)
            ax.fill_between(recall, precision, alpha=0.15, color=color)
        else:
            ax.text(0.5, 0.5, "No instances", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(f"{cls_name}  (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.plot([0, 1], [1, 0], "k--", lw=0.8, alpha=0.4)  # random baseline

    fig.suptitle("Precision-Recall Curves per Class (IoU@0.50)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path or str(PLOTS_DIR / "pr_curves.png"))
    return fig


# -----------------------------------------------------------------------------
# 4. Confusion matrix
# -----------------------------------------------------------------------------

def plot_confusion_matrix(
    predictions: Dict,
    ground_truths: Dict,
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Normalised confusion matrix heat-map."""
    from utils.metrics import build_confusion_matrix

    matrix = build_confusion_matrix(predictions, ground_truths, len(CLASS_NAMES))
    labels = CLASS_NAMES + ["Background"]

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True).astype(float)
        matrix_plot = np.where(row_sums > 0, matrix / row_sums, 0.0)
        fmt, vmax = ".2f", 1.0
    else:
        matrix_plot = matrix.astype(float)
        fmt, vmax = ".0f", None

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix_plot,
        annot=True, fmt=fmt,
        xticklabels=labels, yticklabels=labels,
        cmap="Blues", vmin=0, vmax=vmax,
        linewidths=0.5, linecolor="white",
        ax=ax, annot_kws={"size": 9},
    )
    ax.set_xlabel("Ground Truth", fontsize=12)
    ax.set_ylabel("Predicted",    fontsize=12)
    title = "Confusion Matrix (Normalised)" if normalize else "Confusion Matrix"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    _save(fig, save_path or str(PLOTS_DIR / "confusion_matrix.png"))
    return fig


# -----------------------------------------------------------------------------
# 5. Model comparison (bar chart)
# -----------------------------------------------------------------------------

def plot_model_comparison(
    results: Optional[Dict] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing YOLO variants and DETR.
    Pass custom 'results' dict or use default expected benchmark values.
    """
    if results is None:
        results = {
            "Model":       ["YOLOv8n", "YOLOv8s", "YOLOv8m", "DETR-R50"],
            "mAP@50":      [0.72,       0.77,       0.81,       0.76],
            "mAP@50-95":   [0.48,       0.52,       0.57,       0.53],
            "Car AP":      [0.85,       0.88,       0.91,       0.88],
            "Ped AP":      [0.65,       0.70,       0.74,       0.69],
            "Cyc AP":      [0.62,       0.67,       0.71,       0.65],
            "FPS (A100)":  [120,        90,         60,         28],
            "Params (M)":  [3.2,        11.2,       25.9,       41.3],
        }

    df = pd.DataFrame(results)
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    models = df["Model"].tolist()
    x      = np.arange(len(models))
    bar_colors = ["#3A8BE0", "#E05C3A", "#27AE60", "#9B59B6"][:len(models)]

    # -- mAP@50 ---------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, df["mAP@50"], color=bar_colors, edgecolor="white", lw=0.8)
    ax1.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=20, ha="right")
    ax1.set_title("mAP@50"); ax1.set_ylim(0, 1)

    # -- mAP@50-95 ------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(x, df["mAP@50-95"], color=bar_colors, edgecolor="white", lw=0.8)
    ax2.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=20, ha="right")
    ax2.set_title("mAP@50-95"); ax2.set_ylim(0, 1)

    # -- Per-class AP ----------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    w = 0.25
    ax3.bar(x - w,  df["Car AP"], w, label="Car",        color="#FF6400")
    ax3.bar(x,      df["Ped AP"], w, label="Pedestrian", color="#00DC32")
    ax3.bar(x + w,  df["Cyc AP"], w, label="Cyclist",    color="#FFDC00")
    ax3.set_xticks(x); ax3.set_xticklabels(models, rotation=20, ha="right")
    ax3.set_title("Per-Class AP@50"); ax3.set_ylim(0, 1)
    ax3.legend(fontsize=8)

    # -- Speed vs Accuracy scatter ---------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0:2])
    sc = ax4.scatter(
        df["FPS (A100)"], df["mAP@50"],
        s=[p * 12 for p in df["Params (M)"]],
        c=bar_colors, alpha=0.85, edgecolors="white", linewidths=1.5, zorder=3,
    )
    for _, row in df.iterrows():
        ax4.annotate(
            row["Model"],
            (row["FPS (A100)"], row["mAP@50"]),
            xytext=(8, 4), textcoords="offset points", fontsize=9,
        )
    ax4.set_xlabel("Inference Speed (FPS on A100)")
    ax4.set_ylabel("mAP@50")
    ax4.set_title("Speed?Accuracy Tradeoff  (bubble ? #params)")

    # -- Parameter count bar ---------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    bars5 = ax5.barh(models, df["Params (M)"], color=bar_colors, edgecolor="white", lw=0.8)
    ax5.bar_label(bars5, fmt="%.1fM", padding=3, fontsize=9)
    ax5.set_xlabel("Parameters (M)")
    ax5.set_title("Model Size")
    ax5.invert_yaxis()

    fig.suptitle("Model Comparison ? KITTI Object Detection", fontsize=15, fontweight="bold")
    _save(fig, save_path or str(PLOTS_DIR / "model_comparison.png"))
    return fig


# -----------------------------------------------------------------------------
# 6. BBox size / aspect-ratio distributions (EDA)
# -----------------------------------------------------------------------------

def plot_bbox_stats(
    images_dir: str,
    labels_dir: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Violin plots of bbox width, height, area, aspect ratio per class."""
    samples = parse_kitti_folder(images_dir, labels_dir)
    rows = []
    for s in samples:
        for obj in s["objects"]:
            x1, y1, x2, y2 = obj.bbox
            w, h = x2 - x1, y2 - y1
            rows.append({
                "Class": obj.type,
                "Width (px)": w,
                "Height (px)": h,
                "Area (px?)": w * h,
                "Aspect Ratio": w / max(h, 1),
            })
    df = pd.DataFrame(rows)
    key_classes = ["Car", "Pedestrian", "Cyclist"]
    df = df[df["Class"].isin(key_classes)]

    palette = {"Car": "#FF6400", "Pedestrian": "#00DC32", "Cyclist": "#FFDC00"}
    metrics = ["Width (px)", "Height (px)", "Area (px?)", "Aspect Ratio"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, metrics):
        sns.violinplot(
            data=df, x="Class", y=metric,
            palette=palette, ax=ax, inner="box", cut=0,
        )
        ax.set_title(metric)
        ax.set_xlabel("")

    fig.suptitle("GT Bounding Box Statistics by Class", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path or str(PLOTS_DIR / "bbox_stats.png"))
    return fig


# -----------------------------------------------------------------------------
# 7. Training summary dashboard (combined single-page figure)
# -----------------------------------------------------------------------------

def plot_training_dashboard_yolo(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    All-in-one training dashboard for YOLOv8:
      Row 1: box_loss | cls_loss | dfl_loss
      Row 2: mAP@50   | mAP@50-95 | Precision | Recall
    """
    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.32)
    epochs = df.get("epoch", pd.Series(range(len(df))))

    def _plot(ax, train_col, val_col, title, tc="#E05C3A", vc="#3A8BE0"):
        if train_col in df:
            ax.plot(epochs, df[train_col], color=tc, lw=2, label="Train")
        if val_col and val_col in df:
            ax.plot(epochs, df[val_col], color=vc, lw=2, linestyle="--", label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)

    _plot(fig.add_subplot(gs[0, 0]), "train/box_loss", "val/box_loss",   "Box Loss")
    _plot(fig.add_subplot(gs[0, 1]), "train/cls_loss", "val/cls_loss",   "Cls Loss")
    _plot(fig.add_subplot(gs[0, 2]), "train/dfl_loss", "val/dfl_loss",   "DFL Loss")

    ax_lr = fig.add_subplot(gs[0, 3])
    if "lr/pg0" in df:
        ax_lr.plot(epochs, df["lr/pg0"], color="#F39C12", lw=2, label="lr/pg0")
    if "lr/pg1" in df:
        ax_lr.plot(epochs, df["lr/pg1"], color="#8E44AD", lw=2, linestyle="--", label="lr/pg1")
    ax_lr.set_title("Learning Rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.legend(fontsize=8)

    _plot(fig.add_subplot(gs[1, 0]), "metrics/mAP50(B)",    None,                   "mAP@50",    tc="#3A8BE0")
    _plot(fig.add_subplot(gs[1, 1]), "metrics/mAP50-95(B)", None,                   "mAP@50-95", tc="#E05C3A")
    _plot(fig.add_subplot(gs[1, 2]), "metrics/precision(B)", None,                  "Precision", tc="#27AE60")
    _plot(fig.add_subplot(gs[1, 3]), "metrics/recall(B)",    None,                  "Recall",    tc="#9B59B6")

    # Add best mAP annotation to mAP@50 panel
    ax_map = fig.axes[4]
    if "metrics/mAP50(B)" in df:
        bi  = df["metrics/mAP50(B)"].idxmax()
        bv  = df["metrics/mAP50(B)"].iloc[bi]
        be  = epochs.iloc[bi]
        ax_map.annotate(
            f"  Best {bv:.4f}\n  epoch {int(be)}",
            xy=(be, bv), xytext=(8, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=8, color="black",
        )

    fig.suptitle("YOLOv8 Training Dashboard ? KITTI", fontsize=16, fontweight="bold")
    _save(fig, save_path or str(PLOTS_DIR / "training_summary_yolo.png"))
    return fig


def plot_training_dashboard_detr(
    history: List[dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """All-in-one DETR training dashboard."""
    df     = pd.DataFrame(history)
    epochs = df["epoch"]

    fig = plt.figure(figsize=(20, 9))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.32)

    def _plot(ax, col, title, color, val_col=None, val_color="#3A8BE0"):
        if col in df:
            ax.plot(epochs, df[col], color=color, lw=2, label="Train")
        if val_col and val_col in df:
            ax.plot(epochs, df[val_col], color=val_color, lw=2, linestyle="--", label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)

    _plot(fig.add_subplot(gs[0, 0]), "train/loss",      "Total Loss",   "#E05C3A", "val/loss")
    _plot(fig.add_subplot(gs[0, 1]), "train/loss_ce",   "CE Loss",      "#E05C3A")
    _plot(fig.add_subplot(gs[0, 2]), "train/loss_bbox", "BBox L1 Loss", "#9B59B6")
    _plot(fig.add_subplot(gs[0, 3]), "train/loss_giou", "GIoU Loss",    "#27AE60")
    _plot(fig.add_subplot(gs[1, 0]), "val/mAP50",       "Val mAP@50",   "#3A8BE0")

    # Best annotation
    ax_map = fig.axes[4]
    if "val/mAP50" in df:
        bi = df["val/mAP50"].idxmax()
        bv = df["val/mAP50"].iloc[bi]
        be = epochs.iloc[bi]
        ax_map.annotate(
            f"  Best {bv:.4f}\n  epoch {int(be)}",
            xy=(be, bv), xytext=(8, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=8,
        )

    ax_lr = fig.add_subplot(gs[1, 1])
    if "lr" in df:
        ax_lr.plot(epochs, df["lr"], color="#F39C12", lw=2)
        ax_lr.set_title("Learning Rate"); ax_lr.set_xlabel("Epoch")

    ax_time = fig.add_subplot(gs[1, 2])
    if "time" in df:
        ax_time.bar(epochs, df["time"], color="#BDC3C7", edgecolor="white", lw=0.5)
        ax_time.set_title("Epoch Time (s)"); ax_time.set_xlabel("Epoch")

    fig.add_subplot(gs[1, 3]).axis("off")  # Reserved for future

    fig.suptitle("DETR Training Dashboard ? KITTI", fontsize=16, fontweight="bold")
    _save(fig, save_path or str(PLOTS_DIR / "training_summary_detr.png"))
    return fig


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  [OK] Saved: {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate training plots for kitti-detect")
    p.add_argument("--model",         choices=["yolo", "detr", "both"], required=True)
    p.add_argument("--run-dir",       type=str, default=None,
                   help="Run directory (YOLO or DETR, for --model yolo/detr)")
    p.add_argument("--yolo-run-dir",  type=str, default=None)
    p.add_argument("--detr-run-dir",  type=str, default=None)
    p.add_argument("--images-val",    type=str, default=None)
    p.add_argument("--labels-val",    type=str, default=None)
    p.add_argument("--eval",          action="store_true",
                   help="Run full eval (PR curves + confusion matrix). Needs --images-val/--labels-val")
    p.add_argument("--weights",       type=str, default=None,
                   help="Model weights for eval (required with --eval)")
    p.add_argument("--output-dir",    type=str, default=str(PLOTS_DIR))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    PLOTS_DIR = Path(args.output_dir)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[plots] Saving to: {PLOTS_DIR}\n")

    # -- YOLO plots ------------------------------------------------------------
    if args.model in ("yolo", "both"):
        run_dir = args.yolo_run_dir or args.run_dir
        if run_dir:
            df = load_yolo_results(run_dir)
            plot_training_dashboard_yolo(df)
            plot_loss_curves_yolo(df)
            plot_map_curves_yolo(df)

    # -- DETR plots ------------------------------------------------------------
    if args.model in ("detr", "both"):
        run_dir = args.detr_run_dir or args.run_dir
        if run_dir:
            history = load_detr_history(run_dir)
            plot_training_dashboard_detr(history)
            plot_loss_curves_detr(history)
            plot_map_curves_detr(history)

    # -- Model comparison ------------------------------------------------------
    if args.model == "both":
        plot_model_comparison()

    # -- Evaluation plots (PR + confusion) ------------------------------------
    if args.eval:
        if not (args.images_val and args.labels_val and args.weights):
            print("[ERROR] --eval requires --images-val, --labels-val, and --weights")
            sys.exit(1)

        model_type = "yolo" if args.model == "yolo" else "detr"
        print(f"[plots] Running {model_type.upper()} evaluation for PR curves...")

        sys.path.insert(0, str(Path(__file__).parent))
        from evaluate import KITTIEvaluator
        evaluator = KITTIEvaluator(args.images_val, args.labels_val)

        if model_type == "yolo":
            preds = evaluator.run_yolo(args.weights)
        else:
            preds = evaluator.run_detr(args.weights)

        plot_pr_curves(preds, evaluator.ground_truths)
        plot_confusion_matrix(preds, evaluator.ground_truths)

    # -- BBox stats (needs raw data) -------------------------------------------
    if args.images_val and args.labels_val:
        plot_bbox_stats(args.images_val, args.labels_val)

    print(f"\n[OK] All plots saved to: {PLOTS_DIR}")