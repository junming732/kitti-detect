#!/usr/bin/env bash
# data/download_kitti.sh
#
# Download and extract the KITTI 2D Object Detection dataset into nobackup.
# That's all this script does — no splitting, no symlinks.
# The train/val split is handled automatically by KITTIDataset in kitti_dataset.py.
#
# After this script you will have:
#   $NOBACKUP/kitti-detect-data/kitti/training/image_2/   (7,481 images)
#   $NOBACKUP/kitti-detect-data/kitti/training/label_2/   (7,481 label files)
#
# Usage:
#   bash data/download_kitti.sh

set -e

NOBACKUP="/proj/uppmax2025-2-346/nobackup/private/junming"
DATA_DIR="$NOBACKUP/kitti-detect-data/kitti"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  KITTI Dataset Download"
echo "  Target: $DATA_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Option A: Official download (requires free KITTI account) ─────────────────
# Register atsouthen download:
#   Left color images of object data set  (data_object_image_2.zip, ~12 GB)
#   Training labels of object data set    (data_object_label_2.zip,  ~5 MB)
# Transfer to Rackham with scp, then uncomment and run:
#
# mkdir -p "$DATA_DIR"
# cd "$DATA_DIR"
# unzip data_object_image_2.zip
# unzip data_object_label_2.zip

# ── Option B: Kaggle mirror (no account needed) ───────────────────────────────
# pip install kaggle
# Set up ~/.kaggle/kaggle.json with your API key, then uncomment:
#
#  mkdir -p "$DATA_DIR"
#  cd "$DATA_DIR"
#  kaggle datasets download -d harshitjain16/kitti-dataset
#  unzip -q kitti-dataset.zip
#  rm kitti-dataset.zip
kaggle datasets download -d ibrahimalobaid/kitte-dataset \
  --unzip \
  -p /proj/uppmax2025-2-346/nobackup/private/junming/kitti-detect-data/kitti/

# ── Verify ────────────────────────────────────────────────────────────────────
if [ -d "$DATA_DIR/training/image_2" ] && [ -d "$DATA_DIR/training/label_2" ]; then
    N_IMG=$(ls "$DATA_DIR/training/image_2" | wc -l)
    N_LBL=$(ls "$DATA_DIR/training/label_2" | wc -l)
    echo ""
    echo " Dataset ready: $N_IMG images, $N_LBL labels"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  # For DETR / custom training — dataset handles split automatically:"
    echo "  python data/kitti_dataset.py $DATA_DIR/training/image_2 \\"
    echo "                               $DATA_DIR/training/label_2"
    echo ""
    echo "  # For YOLOv8 — convert to YOLO format on disk:"
    echo "  python -c \""
    echo "  from data.kitti_dataset import convert_kitti_to_yolo"
    echo "  D = '$DATA_DIR/training'"
    echo "  OUT = '$NOBACKUP/kitti-detect-data/kitti_yolo'"
    echo "  convert_kitti_to_yolo(D+'/image_2', D+'/label_2', OUT, split='train')"
    echo "  convert_kitti_to_yolo(D+'/image_2', D+'/label_2', OUT, split='val')"
    echo "  \""
    echo ""
    echo "  # Then train:"
    echo "  python models/yolo_trainer.py --epochs 50 --batch 16"
else
    echo ""
    echo " Data not found at $DATA_DIR/training/"
    echo "   Uncomment Option A or Option B above, then re-run this script."
fi