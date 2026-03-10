#  KITTI Object Detection — YOLOv8 + DETR

> Fine-tuning state-of-the-art object detectors on the KITTI autonomous driving dataset with real-time bounding box visualization.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange.svg)](https://ultralytics.com)
[![DETR](https://img.shields.io/badge/Model-DETR-green.svg)](https://huggingface.co/facebook/detr-resnet-50)
[![KITTI](https://img.shields.io/badge/Dataset-KITTI-red.svg)](http://www.cvlibs.net/datasets/kitti/)

---

##  Project Overview

This project fine-tunes **YOLOv8** and **DETR (DEtection TRansformer)** on the KITTI autonomous driving dataset to detect:
-  Cars
-  Pedestrians  
-  Cyclists
-  Vans, Trucks

It includes a **visualization pipeline** that overlays bounding boxes on video frames, suitable for real-time inference demos.

---

## 🗂 Project Structure

```
kitti-detection/
├── configs/
│   ├── yolo_kitti.yaml          # YOLOv8 training config
│   └── detr_kitti.yaml          # DETR training config
├── data/
│   ├── download_kitti.sh        # Dataset download script
│   └── kitti_dataset.py         # PyTorch Dataset class for KITTI
├── models/
│   ├── yolo_trainer.py          # YOLOv8 fine-tuning script
│   └── detr_trainer.py          # DETR fine-tuning with HuggingFace
├── scripts/
│   ├── visualize_video.py       # Draw bounding boxes on video frames
│   ├── evaluate.py              # Compute mAP, precision, recall
│   └── inference.py             # Single image/video inference
├── utils/
│   ├── kitti_parser.py          # Parse KITTI annotation format
│   ├── bbox_utils.py            # Bounding box utilities
│   └── metrics.py               # Evaluation metrics (mAP, IoU)
├── notebooks/
│   └── EDA_and_Results.ipynb    # Exploratory analysis + results
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download KITTI Dataset
```bash
bash data/download_kitti.sh
# OR manually: http://www.cvlibs.net/datasets/kitti/eval_object.php
```

### 3. Train YOLOv8
```bash
python models/yolo_trainer.py --epochs 50 --batch 16 --imgsz 640
```

### 4. Train DETR
```bash
python models/detr_trainer.py --epochs 30 --batch 8 --lr 1e-4
```

### 5. Visualize on Video
```bash
python scripts/visualize_video.py \
    --model yolo \
    --weights runs/yolo/best.pt \
    --input data/kitti_video.mp4 \
    --output results/output.mp4 \
    --conf 0.5
```

### 6. Evaluate Models
```bash
python scripts/evaluate.py --model yolo --weights runs/yolo/best.pt
```

---

##  KITTI Dataset

| Split | Images | Labels |
|-------|--------|--------|
| Train | 6,481  | Car, Pedestrian, Cyclist, Van, Truck |
| Val   | 1,000  | Same classes |
| Test  | 7,518  | No labels (official submission) |

**Class distribution:**
- Car: ~28,000 instances
- Pedestrian: ~4,400 instances
- Cyclist: ~1,600 instances

---

##  Model Architectures

### YOLOv8 (Ultralytics)
- Backbone: CSPDarknet + C2f modules
- Neck: PAN-FPN
- Head: Decoupled detection head
- Pre-trained: COCO → Fine-tuned on KITTI

### DETR (Facebook AI)
- Backbone: ResNet-50
- Transformer: 6 encoder + 6 decoder layers
- 100 learned object queries
- Pre-trained: COCO → Fine-tuned on KITTI

---

##  Results (Expected Benchmarks)



---

##  Visualization Features

- **Color-coded boxes** by class (Car=blue, Pedestrian=green, Cyclist=yellow)
- **Confidence scores** displayed per detection
- **FPS counter** for real-time performance monitoring
- **Track IDs** (optional, using ByteTrack)
- **Save to MP4** or display live with OpenCV

---

## Key Skills Demonstrated

- Fine-tuning pre-trained models on domain-specific data
- KITTI dataset parsing (unique `.txt` annotation format)
- Multi-class object detection evaluation (mAP, per-class AP)
- Video inference pipeline with OpenCV
- Transformer-based detection (DETR) vs. CNN-based (YOLO)
- Autonomous driving domain knowledge

---

## References

- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [YOLOv8 Docs](https://docs.ultralytics.com)
- [DETR Paper](https://arxiv.org/abs/2005.12872)
- [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)