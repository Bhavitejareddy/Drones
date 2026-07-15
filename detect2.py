import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import deque, Counter

# ── Config ──────────────────────────────────────────────────────────────
WEIGHTS     = "runs/train/drone_detect_v1/weights/best.pt"
SOURCE      = "Datasets/Drone detection.v3i.yolov12/test/images"
CONF        = 0.4
IOU         = 0.45
SAVE_OUTPUT = True
OUTPUT_DIR  = "runs/detect/output"

# ── Temporal smoothing ───────────────────────────────────────────────────
SMOOTH_FRAMES   = 20      # number of frames to average over for label voting
MIN_VOTES_SHOW  = 5       # only show a tracked object after it appears in N frames
                           # (avoids showing label on very first detection)
IOU_MATCH_THRESH = 0.3    # IoU threshold to match a new box to an existing tracked box
BOX_SMOOTH_ALPHA = 0.3    # bbox smoothing: 0=fully fixed, 1=fully instant (0.3 = smooth)
# ────────────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "drone":      (0,   0,   255),
    "bird":       (0,   255, 0  ),
    "aircraft":   (255, 0,   0  ),
    "helicopter": (0,   165, 255),
}
DEFAULT_COLOR = (200, 200, 200)

# ── IoU helper ───────────────────────────────────────────────────────────
def compute_iou(boxA, boxB):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)


# ── Tracker ──────────────────────────────────────────────────────────────
class TrackedObject:
    """
    Tracks a single detected object across frames.

    - label_history : rolling window of raw per-frame labels (last SMOOTH_FRAMES)
    - smooth_box    : exponentially smoothed bounding box (no jumping)
    - stable_label  : majority-vote label over the window (what gets displayed)
    - age           : how many frames this object has been seen
    - missed        : how many consecutive frames it was NOT detected
    """
    _id_counter = 0

    def __init__(self, box, label, conf):
        TrackedObject._id_counter += 1
        self.id            = TrackedObject._id_counter
        self.smooth_box    = list(box)          # [x1,y1,x2,y2] floats
        self.label_history = deque(maxlen=SMOOTH_FRAMES)
        self.conf_history  = deque(maxlen=SMOOTH_FRAMES)
        self.label_history.append(label)
        self.conf_history.append(conf)
        self.stable_label  = label
        self.stable_conf   = conf
        self.age           = 1
        self.missed        = 0

    def update(self, box, label, conf):
        """Called when a new detection is matched to this tracker."""
        # Smooth the bounding box position (exponential moving average)
        for i in range(4):
            self.smooth_box[i] = (BOX_SMOOTH_ALPHA * box[i]
                                  + (1 - BOX_SMOOTH_ALPHA) * self.smooth_box[i])
        self.label_history.append(label)
        self.conf_history.append(conf)
        self.age    += 1
        self.missed  = 0
        self._update_stable()

    def mark_missed(self):
        """Called when no detection matched this tracker this frame."""
        self.missed += 1

    def _update_stable(self):
        """Majority vote over label history → stable_label."""
        vote          = Counter(self.label_history)
        self.stable_label = vote.most_common(1)[0][0]
        # Average confidence only for frames that voted for the winning label
        winning_confs = [
            self.conf_history[i]
            for i, lbl in enumerate(self.label_history)
            if lbl == self.stable_label
        ]
        self.stable_conf = float(np.mean(winning_confs)) if winning_confs else 0.0

    @property
    def box_int(self):
        return [int(v) for v in self.smooth_box]

    @property
    def visible(self):
        """Only show label once we've collected enough votes."""
        return self.age >= MIN_VOTES_SHOW and self.missed == 0


# ── Multi-object tracker pool ────────────────────────────────────────────
class TrackerPool:
    """
    Maintains a pool of TrackedObjects.
    Each frame: matches new detections to existing trackers via IoU,
    creates new trackers for unmatched detections,
    drops trackers that have been missed too long.
    """
    MAX_MISSED = 10   # drop tracker after N consecutive missed frames

    def __init__(self):
        self.trackers = []

    def update(self, detections):
        """
        detections: list of (box, label, conf)
                    box = [x1,y1,x2,y2] floats
        Returns list of TrackedObject (all currently active)
        """
        matched_tracker_ids = set()
        matched_det_ids     = set()

        # Match each detection to the best existing tracker by IoU
        for det_i, (box, label, conf) in enumerate(detections):
            best_iou    = IOU_MATCH_THRESH
            best_trk    = None
            for trk in self.trackers:
                if trk.id in matched_tracker_ids:
                    continue
                iou = compute_iou(box, trk.smooth_box)
                if iou > best_iou:
                    best_iou = iou
                    best_trk = trk
            if best_trk is not None:
                best_trk.update(box, label, conf)
                matched_tracker_ids.add(best_trk.id)
                matched_det_ids.add(det_i)

        # Unmatched detections → new trackers
        for det_i, (box, label, conf) in enumerate(detections):
            if det_i not in matched_det_ids:
                self.trackers.append(TrackedObject(box, label, conf))

        # Unmatched trackers → mark missed
        for trk in self.trackers:
            if trk.id not in matched_tracker_ids:
                trk.mark_missed()

        # Drop stale trackers
        self.trackers = [t for t in self.trackers if t.missed < self.MAX_MISSED]

        return self.trackers


# ── Main ─────────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device : {device}")
print(f"Smooth window: {SMOOTH_FRAMES} frames")
print(f"Min votes to show label: {MIN_VOTES_SHOW} frames\n")

model = YOLO(WEIGHTS)

if SAVE_OUTPUT:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

results = model.predict(
    source=SOURCE,
    conf=CONF,
    iou=IOU,
    save=False,
    stream=True,
    device=device,
)

pool        = TrackerPool()
paused      = False
frame_count = 0

print("Controls: 'q' quit  |  's' save frame  |  SPACE pause/resume")

for result in results:
    img   = result.orig_img.copy()
    names = result.names

    # ── Collect raw detections this frame ──────────────────────────────
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = names[cls_id]
        detections.append(([x1, y1, x2, y2], label, conf))

    # ── Update tracker pool ────────────────────────────────────────────
    active_trackers = pool.update(detections)

    # ── Draw only stable, visible trackers ────────────────────────────
    hud_counts = Counter()

    for trk in active_trackers:
        if not trk.visible:
            continue   # not enough frames yet — skip drawing

        x1, y1, x2, y2 = trk.box_int
        label  = trk.stable_label      # majority-vote label
        conf   = trk.stable_conf       # avg confidence for winning label
        color  = CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)

        hud_counts[label] += 1

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label pill
        text = f"{label} {conf:.2f}  [{len(trk.label_history)}fr]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Small tracker ID (debug — remove if you don't want it)
        cv2.putText(img, f"#{trk.id}", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ── HUD — stable class counts top-left ────────────────────────────
    hud_y = 28
    for cls_name, count in hud_counts.items():
        color    = CLASS_COLORS.get(cls_name.lower(), DEFAULT_COLOR)
        hud_text = f"{cls_name}: {count}"
        cv2.putText(img, hud_text, (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        hud_y += 30

    # Frame counter + smoothing window indicator
    cv2.putText(img, f"Frame {frame_count}  |  window={SMOOTH_FRAMES}fr",
                (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── Save ───────────────────────────────────────────────────────────
    if SAVE_OUTPUT:
        cv2.imwrite(f"{OUTPUT_DIR}/frame_{frame_count:05d}.jpg", img)

    frame_count += 1

    # ── Display ────────────────────────────────────────────────────────
    cv2.imshow("Drone Detection — Temporal Smoothing", img)
    key = cv2.waitKey(0 if paused else 1) & 0xFF

    if key == ord('q'):
        print("Quitting..."); break
    elif key == ord('s'):
        save_path = f"{OUTPUT_DIR}/saved_{frame_count:05d}.jpg"
        cv2.imwrite(save_path, img)
        print(f"Saved: {save_path}")
    elif key == ord(' '):
        paused = not paused
        print("Paused" if paused else "Resumed")

cv2.destroyAllWindows()
print(f"\n✅ Done! Processed {frame_count} frames.")
if SAVE_OUTPUT:
    print(f"Output saved to: {OUTPUT_DIR}/")
