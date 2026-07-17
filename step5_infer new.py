"""
STEP 5 — Full 2-Stage Inference Pipeline
===================================================
Matches your diagram exactly:

 Image/Video
     │
 Stage 1: YOLOv12n → drone / bird / aircraft / helicopter
     │
 class == drone?
 ├── No  → final label = bird / aircraft / helicopter  (stage-1 conf shown)
 └── Yes → quality gates:
             yolo_conf   >= YOLO_CONF_GATE
             bbox_area   >= BBOX_AREA_THRESHOLD
             sharpness   >= LAPLACIAN_VAR_GATE   (Laplacian variance, blur check)
           ├── any fail → "drone (size N/A)"
           └── all pass → Stage 2 classifier → nano/micro/small/medium/large

Notes on this revision
-----------------------
* No object tracking. Every frame is processed independently by Stage 1 / Stage 2.
* Bounding boxes always show the STAGE-1 confidence (never the stage-2 confidence).
* If USE_FULL_STAGE1_SCORES is True (config.py), Stage 1 confidence is a TRUE
  4-way softmax/sigmoid score across drone/bird/aircraft/helicopter, obtained
  by running the raw pre-NMS model output ourselves instead of relying on
  Ultralytics' `.predict()`, which only returns the single winning class per
  box after NMS. If this raw path fails for any reason (e.g. a future
  Ultralytics version changes its internal output format), it automatically
  falls back to the normal single-class `.predict()` confidence and prints a
  one-time warning — nothing crashes.
* A small on-screen panel lists the confidence for every Stage-1 class
  (drone / bird / aircraft / helicopter) for each box, and — for any drone
  that cleared the gates — the full nano/micro/small/medium/large tier
  breakdown from Stage 2.
* Images (single file or a folder): every image is predicted independently,
  no temporal smoothing.
* Video / webcam: raw per-frame predictions are shown for the first
  SMOOTH_FRAMES (default 20) frames. From then on a sliding-window majority
  vote over the last SMOOTH_FRAMES frames is also displayed as the
  "stable" prediction, and it updates (slides) every frame — so if the
  majority label changes, the displayed stable prediction changes with it.

Run (only after Steps 1-4 are complete and CLASSIFIER_PATH is set in config.py):
    python step5_infer.py
"""

import os
import sys
from pathlib import Path
from collections import Counter, deque

from config import (
    WEIGHTS_PATH, CLASSIFIER_PATH, CLASSIFIER_TYPE,
    INFER_SOURCE, INFER_OUTPUT_DIR,
    BBOX_AREA_THRESHOLD, SMOOTH_FRAMES, SHOW_WINDOW, SAVE_INFER_OUTPUT,
    OUTPUT_DIR, IMAGE_SIZE, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD,
    SUPPORTED_EXTS, TIERS,
    separator,
)

# ── Optional extras (fall back to sensible defaults if not in config.py) ──
try:
    from config import YOLO_CONF_GATE
except ImportError:
    YOLO_CONF_GATE = 0.55

try:
    from config import LAPLACIAN_VAR_GATE
except ImportError:
    LAPLACIAN_VAR_GATE = 80.0

try:
    from config import USE_FULL_STAGE1_SCORES
except ImportError:
    USE_FULL_STAGE1_SCORES = True   # set False in config.py to skip the raw-forward-pass path

# Window size for the sliding-window majority vote on video/webcam sources.
SMOOTH_WINDOW = SMOOTH_FRAMES if SMOOTH_FRAMES else 20

# Only print the raw-scores fallback warning once, not every frame.
_WARNED_RAW_FALLBACK = [False]


def load_stage2_classifier():
    """Load the trained size classifier (ResNet50 or EfficientNetB0)."""
    import torch
    import torch.nn as nn
    from torchvision import models

    if not os.path.exists(CLASSIFIER_PATH):
        print(f"ERROR: Classifier not found: {CLASSIFIER_PATH}")
        print(f"       Run Steps 1-4 first to train the classifier.")
        sys.exit(1)

    if CLASSIFIER_TYPE == "efficientnetb0":
        m = models.efficientnet_b0(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, NUM_CLASSES))
    else:
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, NUM_CLASSES))

    m.load_state_dict(torch.load(CLASSIFIER_PATH, map_location="cpu"))
    m.eval()
    print(f"  Stage 2 classifier loaded: {CLASSIFIER_TYPE}")
    return m


def get_infer_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def classify_size(classifier, crop_bgr, tf, class_names):
    """
    Run Stage 2 on a cropped drone region.
    Returns (tier_label, tier_conf, full_probs_dict) where full_probs_dict maps
    every tier name -> probability, so the caller can show the full breakdown.
    """
    import torch
    crop_rgb = crop_bgr[:, :, ::-1].copy()      # BGR → RGB for PIL
    tensor   = tf(crop_rgb).unsqueeze(0)         # (1, 3, 224, 224)
    with torch.no_grad():
        out   = classifier(tensor)
        probs = torch.softmax(out, dim=1)[0]
        idx   = probs.argmax().item()
    full_probs = {name: float(probs[i]) for i, name in enumerate(class_names)}
    return class_names[idx], float(probs[idx]), full_probs


def laplacian_variance(crop_bgr):
    """Blur/sharpness metric — higher = sharper. Used as a Stage-2 quality gate."""
    import cv2
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Stage-1 raw (pre-NMS) multi-class scoring ───────────────────────────
def init_stage1_predictor(yolo):
    """
    Warm up yolo.predictor (needed before raw forward passes). Ultralytics
    only builds `yolo.predictor` the first time `.predict()` is called, so we
    run one throwaway inference on a tiny blank frame at startup.
    """
    import numpy as np
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    yolo.predict(dummy, verbose=False)


def raw_stage1_predictions(yolo, frame, conf_thres=0.4, iou_thres=0.45):
    """
    Runs Stage-1 ourselves via a single raw forward pass, bypassing
    Ultralytics' `.predict()` NMS step — which silently discards every class
    score except the winner. This keeps the FULL per-class probability
    vector (drone/bird/aircraft/helicopter) for every surviving box.

    Returns: list of {"box": [x1,y1,x2,y2], "class_probs": {name: prob, ...}}
    in ORIGINAL frame pixel coordinates.

    Depends on the YOLOv8-family anchor-free Detect head output shape
    (batch, 4+nc, num_anchors), with class scores already sigmoid-activated —
    true for YOLOv8 through YOLOv12 in Ultralytics as of this writing. If a
    future Ultralytics release changes this, this raises and the caller (see
    _process_frame) transparently falls back to single-class scores.
    """
    import torch
    import torchvision
    from ultralytics.utils.ops import scale_boxes

    predictor = yolo.predictor
    if predictor is None:
        raise RuntimeError("Stage-1 predictor not initialized — call init_stage1_predictor(yolo) first.")

    names = yolo.names            # {0: 'drone', 1: 'bird', ...}
    nc = len(names)

    im0 = frame
    im = predictor.preprocess([im0])          # letterboxed + normalized tensor
    with torch.no_grad():
        raw = predictor.model(im)
    if isinstance(raw, (list, tuple)):
        raw = raw[0]

    raw = raw[0].transpose(0, 1)               # (num_anchors, 4+nc)
    boxes = raw[:, :4]
    cls_scores = raw[:, 4:4 + nc]              # per-class probs, pre-NMS, all classes intact

    max_scores, _ = cls_scores.max(dim=1)
    keep = max_scores >= conf_thres
    boxes, cls_scores, max_scores = boxes[keep], cls_scores[keep], max_scores[keep]

    if boxes.shape[0] == 0:
        return []

    # NMS ranked by the winning class's score — same effective ranking
    # Ultralytics uses — but we DON'T discard the other class scores.
    nms_idx = torchvision.ops.nms(boxes, max_scores, iou_thres)
    boxes, cls_scores = boxes[nms_idx], cls_scores[nms_idx]

    boxes = scale_boxes(im.shape[2:], boxes, im0.shape).round()

    out = []
    for box, scores in zip(boxes.tolist(), cls_scores.tolist()):
        out.append({
            "box": box,
            "class_probs": {names[i]: float(scores[i]) for i in range(nc)},
        })
    return out


# ── Class colors ─────────────────────────────────────────────
COLORS = {
    "drone":      (0,   0,   255),
    "bird":       (0,   255, 0  ),
    "aircraft":   (255, 0,   0  ),
    "helicopter": (0,   165, 255),
    "nano":       (0,   0,   200),
    "micro":      (0,   100, 255),
    "small":      (0,   200, 255),
    "medium":     (0,   255, 200),
    "large":      (50,  255, 50 ),
}
DEFAULT_COLOR = (200, 200, 200)


def draw_box(img, x1, y1, x2, y2, label, stage1_conf, color):
    """Draws a box whose label always shows the STAGE-1 confidence."""
    import cv2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {stage1_conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_confidence_panel(frame, panel_entries, stable_label=None):
    """
    Top-right panel listing, per detected object this frame:
      - Stage-1 scores for EVERY class (drone/bird/aircraft/helicopter) if
        raw scoring succeeded, or just the winning class if it fell back.
      - if it's a drone that cleared the gates: full tier breakdown
        (nano/micro/small/medium/large)
    """
    import cv2
    fh, fw = frame.shape[:2]
    x0 = fw - 260
    y  = 24

    cv2.putText(frame, "Confidence", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    y += 22

    for entry in panel_entries:
        for cname, cprob in entry["class_probs"].items():
            color = COLORS.get(cname, DEFAULT_COLOR)
            marker = ">" if cname == entry["stage1_label"] else " "
            cv2.putText(frame, f"{marker}{cname}: {cprob:.2f}",
                        (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 17

        if entry.get("tier_probs"):
            for tier in ("nano", "micro", "small", "medium", "large"):
                p = entry["tier_probs"].get(tier, 0.0)
                tcolor = COLORS.get(tier, DEFAULT_COLOR)
                cv2.putText(frame, f"   {tier}: {p:.2f}",
                            (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tcolor, 1)
                y += 16
        y += 8

    if stable_label is not None:
        cv2.putText(frame, f"Stable (last {SMOOTH_WINDOW}fr): {stable_label}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def _process_frame(frame, yolo, classifier, infer_tf, size_classes):
    """
    Core 2-stage logic per frame — no tracking, purely per-frame.
    Returns list of entries:
      {
        "box": [x1,y1,x2,y2],
        "final_label": str,           # what's drawn on the box
        "stage1_label": str,          # winning Stage-1 class name
        "stage1_conf": float,         # ALWAYS the Stage-1 confidence (winner)
        "class_probs": dict,          # {class_name: prob} — all 4 if raw path worked,
                                       # otherwise just {stage1_label: stage1_conf}
        "tier_probs": dict|None,      # full Stage-2 tier breakdown if computed
        "color": tuple,
      }
    """
    fh, fw = frame.shape[:2]

    raw_dets = None
    if USE_FULL_STAGE1_SCORES:
        try:
            raw_dets = raw_stage1_predictions(yolo, frame, conf_thres=0.4, iou_thres=0.45)
        except Exception as e:
            if not _WARNED_RAW_FALLBACK[0]:
                print(f"  WARNING: full 4-class Stage-1 scores unavailable ({e}); "
                      f"falling back to single-class confidence via yolo.predict().")
                _WARNED_RAW_FALLBACK[0] = True
            raw_dets = None

    if raw_dets is not None:
        detections = raw_dets   # each has box + full class_probs already
    else:
        results = yolo.predict(frame, conf=0.4, iou=0.45, verbose=False)
        r = results[0]
        detections = []
        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls_id].lower()
            detections.append({
                "box": [x1, y1, x2, y2],
                "class_probs": {label: conf},   # only the winner is known here
            })

    entries = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        class_probs = det["class_probs"]
        stage1_label = max(class_probs, key=class_probs.get)
        stage1_conf = class_probs[stage1_label]

        tier_probs = None

        # ── STAGE 1 branch ──────────────────────────────────────
        if stage1_label != "drone":
            final_label = stage1_label
            color = COLORS.get(stage1_label, DEFAULT_COLOR)

        else:
            # ── Drone detected — run quality gates before Stage 2 ────
            bbox_area = (x2 - x1) * (y2 - y1)

            cx1 = max(0, int(x1)); cy1 = max(0, int(y1))
            cx2 = min(fw, int(x2)); cy2 = min(fh, int(y2))
            crop = frame[cy1:cy2, cx1:cx2]

            passes_conf = stage1_conf >= YOLO_CONF_GATE
            passes_area = bbox_area >= BBOX_AREA_THRESHOLD
            passes_sharp = crop.size != 0 and laplacian_variance(crop) >= LAPLACIAN_VAR_GATE

            if not (passes_conf and passes_area and passes_sharp):
                final_label = "drone (size N/A)"
                color = COLORS["drone"]
            else:
                size_tier, _size_conf, tier_probs = classify_size(
                    classifier, crop, infer_tf, size_classes)
                final_label = f"drone ({size_tier})"
                color = COLORS.get(size_tier, COLORS["drone"])

        entries.append({
            "box": [x1, y1, x2, y2],
            "final_label": final_label,
            "stage1_label": stage1_label,
            "stage1_conf": stage1_conf,
            "class_probs": class_probs,
            "tier_probs": tier_probs,
            "color": color,
        })

    return entries


def _draw_frame(frame, entries, frame_idx, stable_label=None):
    import cv2
    hud_counts = Counter()
    for e in entries:
        x1, y1, x2, y2 = [int(v) for v in e["box"]]
        draw_box(frame, x1, y1, x2, y2, e["final_label"], e["stage1_conf"], e["color"])
        hud_counts[e["final_label"]] += 1

    # Left-side HUD: counts of what's on screen right now (raw, per-frame)
    hud_y = 28
    for lbl, cnt in hud_counts.items():
        base = lbl.replace("drone (", "").replace(")", "")
        color = COLORS.get(base, DEFAULT_COLOR)
        cv2.putText(frame, f"{lbl}: {cnt}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        hud_y += 30

    # Right-side confidence panel (stage1 4-class + stage2 tier breakdown)
    draw_confidence_panel(frame, entries, stable_label)

    cv2.putText(frame,
                f"Frame {frame_idx}  |  window={SMOOTH_WINDOW}fr  |  "
                f"area_thresh={BBOX_AREA_THRESHOLD}px  |  "
                f"conf_gate={YOLO_CONF_GATE}  |  sharp_gate={LAPLACIAN_VAR_GATE}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


def _frame_dominant_label(entries):
    """Majority label *within a single frame* (mode across simultaneous detections)."""
    if not entries:
        return None
    labels = [e["final_label"] for e in entries]
    return Counter(labels).most_common(1)[0][0]


def _run_step5_images(image_files, yolo, classifier, infer_tf, size_classes):
    """Process a folder of images — every image predicted independently, no smoothing."""
    import cv2
    for idx, fpath in enumerate(image_files):
        frame = cv2.imread(fpath)
        if frame is None:
            continue
        entries = _process_frame(frame, yolo, classifier, infer_tf, size_classes)
        _draw_frame(frame, entries, idx)
        if SAVE_INFER_OUTPUT:
            out = os.path.join(INFER_OUTPUT_DIR, Path(fpath).name)
            cv2.imwrite(out, frame)
        if SHOW_WINDOW:
            cv2.imshow("2-Stage Drone Pipeline", frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(image_files)} images done")
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print(f"  ✓ Done. {len(image_files)} images processed.")
    if SAVE_INFER_OUTPUT:
        print(f"  Output saved to: {INFER_OUTPUT_DIR}")


def run_step5():
    separator("STEP 5 — 2-Stage Inference Pipeline")
    import cv2
    import torch
    from ultralytics import YOLO

    # ── Validate paths
    for label, path in [("WEIGHTS_PATH", WEIGHTS_PATH),
                         ("CLASSIFIER_PATH", CLASSIFIER_PATH)]:
        if "path\\to" in path or "path/to" in path:
            print(f"ERROR: {label} still has placeholder value. Fill it in config.py.")
            sys.exit(1)

    if SAVE_INFER_OUTPUT:
        os.makedirs(INFER_OUTPUT_DIR, exist_ok=True)

    # ── Load class names for Stage 2
    class_file = os.path.join(OUTPUT_DIR, "class_names.txt")
    if os.path.exists(class_file):
        with open(class_file) as f:
            size_classes = [line.strip().split()[1] for line in f if line.strip()]
    else:
        size_classes = TIERS
    print(f"  Size classes : {size_classes}")

    # ── Load models
    print(f"  Loading Stage 1 YOLO from  : {WEIGHTS_PATH}")
    yolo = YOLO(WEIGHTS_PATH)
    classifier = load_stage2_classifier()
    infer_tf = get_infer_transform()

    if USE_FULL_STAGE1_SCORES:
        init_stage1_predictor(yolo)
        print("  Full 4-class Stage-1 scoring: ENABLED (raw pre-NMS forward pass)")
    else:
        print("  Full 4-class Stage-1 scoring: DISABLED (single-class via yolo.predict())")

    # ── Open source
    if INFER_SOURCE == 0 or str(INFER_SOURCE).isdigit():
        cap = cv2.VideoCapture(0)
        print("  Source: webcam")
    elif os.path.isfile(str(INFER_SOURCE)):
        cap = cv2.VideoCapture(str(INFER_SOURCE))
        print(f"  Source: video file — {INFER_SOURCE}")
    elif os.path.isdir(str(INFER_SOURCE)):
        # Image folder mode — process each image independently, no smoothing
        image_files = sorted([
            str(f) for f in Path(INFER_SOURCE).iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS
        ])
        print(f"  Source: image folder — {len(image_files)} images")
        _run_step5_images(image_files, yolo, classifier, infer_tf, size_classes)
        return
    else:
        print(f"ERROR: INFER_SOURCE not found: {INFER_SOURCE}"); sys.exit(1)

    # ── Video / webcam loop ─────────────────────────────────────
    # No tracking. First SMOOTH_WINDOW frames show the raw per-frame prediction.
    # From then on, a sliding-window majority vote (over the dominant label of
    # each of the last SMOOTH_WINDOW frames) is also shown as the "stable"
    # prediction, and it slides/updates every new frame.
    frame_label_history = deque(maxlen=SMOOTH_WINDOW)
    frame_idx = 0
    paused = False
    print("\n  Controls: Q=quit  S=save frame  SPACE=pause\n")
    print(f"  Raw predictions shown for first {SMOOTH_WINDOW} frames; "
          f"sliding majority vote shown after that.\n")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            entries = _process_frame(frame, yolo, classifier, infer_tf, size_classes)

            dominant = _frame_dominant_label(entries)
            if dominant is not None:
                frame_label_history.append(dominant)

            if len(frame_label_history) < SMOOTH_WINDOW:
                stable_label = dominant   # not enough history yet — show raw prediction
            else:
                stable_label = Counter(frame_label_history).most_common(1)[0][0]

            _draw_frame(frame, entries, frame_idx, stable_label=stable_label)

            if SAVE_INFER_OUTPUT:
                cv2.imwrite(f"{INFER_OUTPUT_DIR}/frame_{frame_idx:05d}.jpg", frame)

            frame_idx += 1

        if SHOW_WINDOW:
            cv2.imshow("2-Stage Drone Pipeline", frame)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                sp = f"{INFER_OUTPUT_DIR}/saved_{frame_idx:05d}.jpg"
                cv2.imwrite(sp, frame); print(f"  Saved: {sp}")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
        else:
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames...")

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print(f"\n  ✓ Done. {frame_idx} frames processed.")
    if SAVE_INFER_OUTPUT:
        print(f"  Output saved to: {INFER_OUTPUT_DIR}")


if __name__ == "__main__":
    run_step5()
