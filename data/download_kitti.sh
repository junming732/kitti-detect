#!/usr/bin/env bash
# data/download_kitti.sh
#
# Downloads KITTI and converts to YOLO format in one go.
# Re-runnable — skips steps that are already done.
#
# Usage:
#   bash data/download_kitti.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOBACKUP="/proj/uppmax2025-2-346/nobackup/private/junming"
DATA_DIR="$NOBACKUP/kitti-detect-data/kitti"
YOLO_DIR="$NOBACKUP/kitti-detect-data/kitti_yolo"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  KITTI Dataset Setup"
echo "  Raw data : $DATA_DIR"
echo "  YOLO fmt : $YOLO_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Download ──────────────────────────────────────────────────────────
# Uncomment ONE of the options below, then re-run this script.

# Option A — Official KITTI (requires free account at cvlibs.net/datasets/kitti)
# scp your downloaded zips to Rackham first, then:
# mkdir -p "$DATA_DIR" && cd "$DATA_DIR"
# unzip data_object_image_2.zip
# unzip data_object_label_2.zip

# Option B — Kaggle mirror (ibrahimalobaid/kitte-dataset, verified structure)
# mkdir -p "$DATA_DIR"
# kaggle datasets download -d ibrahimalobaid/kitte-dataset --unzip -p "$DATA_DIR"

# ── Step 2: Verify raw data ───────────────────────────────────────────────────
if [ ! -d "$DATA_DIR/training/image_2" ] || [ ! -d "$DATA_DIR/training/label_2" ]; then
    echo ""
    echo "⏳ Raw data not found at $DATA_DIR/training/"
    echo "   Uncomment Option A or B above and re-run."
    exit 0
fi

N_IMG=$(ls "$DATA_DIR/training/image_2" | wc -l)
N_LBL=$(ls "$DATA_DIR/training/label_2" | wc -l)
echo ""
echo " Raw data: $N_IMG images, $N_LBL labels"

# ── Step 3: Convert to YOLO format ───────────────────────────────────────────
if [ -d "$YOLO_DIR/images/train" ] && [ -d "$YOLO_DIR/images/val" ]; then
    echo " YOLO format already exists at $YOLO_DIR — skipping conversion."
else
    echo ""
    echo "Converting to YOLO format..."
    source "$PROJECT_ROOT/venv/bin/activate"
    cd "$PROJECT_ROOT"
    python - << PYEOF
from data.kitti_dataset import convert_kitti_to_yolo
convert_kitti_to_yolo('train')
convert_kitti_to_yolo('val')
PYEOF
    echo " YOLO conversion done."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo " All done! Ready to train."
echo ""
echo "  YOLOv8:  python models/yolo_trainer.py --epochs 50 --batch 16"
echo "  DETR:    python models/detr_trainer.py --epochs 30 --batch 8"