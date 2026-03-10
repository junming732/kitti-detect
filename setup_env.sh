#!/usr/bin/env bash
# setup_env.sh
#
# Sets up the Python virtual environment for kitti-detect on UPPMAX.
#
# Usage:
#   bash setup_env.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
NOBACKUP="/proj/uppmax2025-2-346/nobackup/private/junming"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  kitti-detect — Environment Setup"
echo "  Project : $PROJECT_ROOT"
echo "  Venv    : $VENV_DIR"
echo "  Nobackup: $NOBACKUP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Load modules (uncomment what's available on your UPPMAX cluster) ───────
# module load python/3.11.8
# module load cuda/12.1
# module load cudnn/8.9

# ── 2. Create virtual environment ────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Venv already exists at $VENV_DIR"
else
    echo "[INFO] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "[INFO] Done."
fi

# ── 3. Install packages ───────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$PROJECT_ROOT/requirements.txt"
echo "[INFO] Packages installed."

# ── 4. Write .env so paths.py picks up the correct base directory ─────────────
# Data will live under $NOBACKUP/kitti-detect-data/ once you download it.
# We do NOT create those directories now — download_kitti.sh does that
# only after the actual data is present.
cat > "$PROJECT_ROOT/.env" << ENVEOF
# Auto-generated — do not commit (listed in .gitignore)
NOBACKUP=$NOBACKUP
KITTI_DATA_DIR=$NOBACKUP/kitti-detect-data/kitti
KITTI_YOLO_DIR=$NOBACKUP/kitti-detect-data/kitti_yolo
RUNS_DIR=$NOBACKUP/kitti-detect-data/runs
ENVEOF
echo "[INFO] Written: $PROJECT_ROOT/.env"

echo ""
echo " Setup complete!"
echo ""
echo "  Activate venv:       source venv/bin/activate"
echo "  Verify paths:        python -c 'from utils import print_paths; print_paths()'"
echo "  Download KITTI next: bash data/download_kitti.sh"