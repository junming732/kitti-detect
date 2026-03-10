"""
utils/paths.py

Central path configuration for kitti-detect.
Reads from environment variables set by setup_env.sh / .env,
with hardcoded fallbacks so the code is never broken without .env.

Import from here — never hardcode paths in any other script.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ── Nobackup (UPPMAX large storage, already exists as a softlink) ─────────────
NOBACKUP = Path(os.environ.get(
    "NOBACKUP",
    "/proj/uppmax2025-2-346/nobackup/private/junming"
))

# ── Raw KITTI data (single source of truth, no split subdirs) ────────────────
KITTI_DATA_DIR  = Path(os.environ.get("KITTI_DATA_DIR",  str(NOBACKUP / "kitti-detect-data/kitti")))
KITTI_RAW_IMAGES = KITTI_DATA_DIR / "training" / "image_2"
KITTI_RAW_LABELS = KITTI_DATA_DIR / "training" / "label_2"

# ── YOLO-format converted dataset (written by convert_kitti_to_yolo) ─────────
KITTI_YOLO_DIR  = Path(os.environ.get("KITTI_YOLO_DIR",  str(NOBACKUP / "kitti-detect-data/kitti_yolo")))

# ── Model checkpoints ─────────────────────────────────────────────────────────
RUNS_DIR        = Path(os.environ.get("RUNS_DIR",        str(NOBACKUP / "kitti-detect-data/runs")))
YOLO_RUNS_DIR   = RUNS_DIR / "yolo"
DETR_RUNS_DIR   = RUNS_DIR / "detr"

# ── Results (plots, annotated videos) — kept inside the repo ─────────────────
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def print_paths():
    """Run this to verify all paths on a new machine: python utils/paths.py"""
    rows = [
        ("PROJECT_ROOT",      PROJECT_ROOT),
        ("NOBACKUP",          NOBACKUP),
        ("KITTI_RAW_IMAGES",  KITTI_RAW_IMAGES),
        ("KITTI_RAW_LABELS",  KITTI_RAW_LABELS),
        ("KITTI_YOLO_DIR",    KITTI_YOLO_DIR),
        ("RUNS_DIR",          RUNS_DIR),
        ("RESULTS_DIR",       RESULTS_DIR),
    ]
    print("── kitti-detect paths ────────────────────────────────")
    for name, path in rows:
        status = "success" if path.exists() else "fail"
        print(f"  {status}  {name:<20}: {path}")
    print("──────────────────────────────────────────────────────")


if __name__ == "__main__":
    print_paths()