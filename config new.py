"""
config.py
===================================================
Central configuration for the 2-Stage Drone Detection & Size
Classification pipeline (Steps 1-5).

Fill in the placeholder paths (WEIGHTS_PATH, CLASSIFIER_PATH, INFER_SOURCE)
before running step5_infer.py. Everything else has a working default.
"""

import os

# ── Directory layout ────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")          # class_names.txt, checkpoints, etc.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Stage 1 — YOLO detector ─────────────────────────────────────
# Path to the trained YOLOv12n weights (drone/bird/aircraft/helicopter detector).
WEIGHTS_PATH = "path/to/yolov12n_drone_detector.pt"

# ── Stage 2 — Size classifier ───────────────────────────────────
# Path to the trained tier classifier from Steps 1-4.
CLASSIFIER_PATH  = "path/to/size_classifier.pt"
CLASSIFIER_TYPE  = "resnet50"          # "resnet50" or "efficientnetb0"
NUM_CLASSES      = 5                   # nano, micro, small, medium, large
TIERS            = ["nano", "micro", "small", "medium", "large"]

# Image preprocessing for Stage 2 (must match training config)
IMAGE_SIZE     = 224
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# ── Stage 2 quality gates (only classify size if ALL pass) ─────
# Otherwise the drone is labeled "drone (size N/A)".
BBOX_AREA_THRESHOLD = 1600     # px^2 — bbox too small/far to reliably tell tiers apart
YOLO_CONF_GATE      = 0.55     # Stage-1 detection confidence must be at least this
LAPLACIAN_VAR_GATE  = 80.0     # blur/sharpness gate — Laplacian variance of the crop

# ── Stage 1 scoring mode ────────────────────────────────────────
# True  -> bypass Ultralytics' `.predict()` NMS and run a raw pre-NMS
#          forward pass ourselves, so the confidence panel shows a TRUE
#          4-way score across drone/bird/aircraft/helicopter per box.
#          Costs an extra forward pass per frame (roughly ~2x Stage-1
#          inference time). Falls back automatically (with a one-time
#          console warning) to single-class scoring if the raw path fails.
# False -> use the normal single-class `.predict()` confidence only
#          (faster, matches the original behavior).
USE_FULL_STAGE1_SCORES = True


# ── Inference source ─────────────────────────────────────────────
# One of:
#   0                          -> webcam
#   "path/to/video.mp4"        -> video file
#   "path/to/image_or_folder"  -> single image or a folder of images
INFER_SOURCE     = "path/to/input_video_or_folder"
INFER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "inference_results")

SAVE_INFER_OUTPUT = True    # save annotated frames/images to INFER_OUTPUT_DIR
SHOW_WINDOW        = True    # display an OpenCV window while running (set False for headless/servers)

# ── Video smoothing ──────────────────────────────────────────────
# Sliding-window size (in frames) used for the majority-vote "stable"
# prediction shown for video/webcam sources (see step5_infer.py).
SMOOTH_FRAMES = 20

# ── Misc ──────────────────────────────────────────────────────────
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def separator(title=""):
    """Pretty console section header used throughout Steps 1-5."""
    bar = "=" * 60
    if title:
        print(f"\n{bar}\n {title}\n{bar}")
    else:
        print(f"\n{bar}")
