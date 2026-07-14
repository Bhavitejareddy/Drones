import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from collections import deque, Counter

# ── Config ──────────────────────────────────────────────────────────────
WEIGHTS = "runs/train/drone_detect_v1/weights/best.pt"
SOURCE  = "Datasets/Drone detection.v3i.yolov12/test/images"
CONF    = 0.4
IOU     = 0.45
SAVE_OUTPUT = True
OUTPUT_DIR  = "runs/detect/output"

# ── Temporal smoothing config (no YOLO track IDs — plain predict() + our
#    own IoU matching to link boxes across frames) ─────────────────────
MATCH_IOU_THRESH = 0.3   # min overlap with previous frame's box to count as "same object"
VOTE_WINDOW      = 20    # min frames of evidence before a label locks
VOTE_MAXLEN      = 30    # how many recent votes we keep per object
STALE_FRAMES     = 10    # drop an object if unseen for this many frames
# ────────────────────────────────────────────────────────────────────────

# Class colors: BGR format
CLASS_COLORS = {
    "drone":      (0,   0,   255),   # Red
    "bird":       (0,   255, 0  ),   # Green
    "aircraft":   (255, 0,   0  ),   # Blue
    "helicopter": (0,   165, 255),   # Orange
}
DEFAULT_COLOR = (200, 200, 200)      # Grey for unknown classes

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model = YOLO(WEIGHTS)

# Output folder
if SAVE_OUTPUT:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TrackedObject:
    """One physical object, followed across frames purely by box overlap
    (no YOLO track ID). Holds a rolling vote of its detected classes.

    Below VOTE_WINDOW frames of evidence, reports the current majority
    vote so something sensible shows immediately, but does not lock.
    Once VOTE_WINDOW frames are collected, the majority label freezes
    ("fixed") and never changes again for this object.
    """

    def __init__(self, obj_id, box, label, frame_idx):
        self.id = obj_id
        self.box = box
        self.history = deque(maxlen=VOTE_MAXLEN)
        self.locked_label = None
        self.last_seen = frame_idx
        self.label = None
        self._vote(label)

    def _vote(self, label):
        self.history.append(label)
        current_best = Counter(self.history).most_common(1)[0][0]
        if self.locked_label is None and len(self.history) >= VOTE_WINDOW:
            self.locked_label = current_best
        self.label = self.locked_label if self.locked_label is not None else current_best

    def update(self, box, label, frame_idx):
        self.box = box
        self.last_seen = frame_idx
        if self.locked_label is None:
            self._vote(label)

    @property
    def is_locked(self):
        return self.locked_label is not None


tracked_objects = []   # list of TrackedObject, matched by IoU frame-to-frame
next_obj_id = 0

# Plain per-frame detection — no persist/track, we do our own matching below.
results = model.predict(
    source=SOURCE,
    conf=CONF,
    iou=IOU,
    save=False,
    stream=True,
    device=device,
)

print("Press 'q' to quit, 's' to save current frame, SPACE to pause")
print(f"Each object's label is averaged over its first {VOTE_WINDOW} matched "
      f"frames (matched by box overlap, no track IDs), then fixed.\n")

paused = False
frame_count = 0

for result in results:
    img   = result.orig_img.copy()
    names = result.names

    # Current frame's raw detections
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label  = names[cls_id]
            detections.append((x1, y1, x2, y2, label))

    # ── Match detections to existing tracked objects via best IoU ──────
    candidates = []
    for di, (dx1, dy1, dx2, dy2, _label) in enumerate(detections):
        for oi, obj in enumerate(tracked_objects):
            score = iou((dx1, dy1, dx2, dy2), obj.box)
            if score >= MATCH_IOU_THRESH:
                candidates.append((score, di, oi))
    candidates.sort(reverse=True, key=lambda c: c[0])

    used_dets, used_objs = set(), set()
    matches = []  # (det_idx, obj_idx)
    for score, di, oi in candidates:
        if di in used_dets or oi in used_objs:
            continue
        matches.append((di, oi))
        used_dets.add(di)
        used_objs.add(oi)

    # Update matched objects with this frame's box + class vote
    for di, oi in matches:
        x1, y1, x2, y2, label = detections[di]
        tracked_objects[oi].update((x1, y1, x2, y2), label, frame_count)

    # Any detection that didn't match an existing object becomes a new one
    for di, (x1, y1, x2, y2, label) in enumerate(detections):
        if di in used_dets:
            continue
        tracked_objects.append(TrackedObject(next_obj_id, (x1, y1, x2, y2), label, frame_count))
        next_obj_id += 1

    # Drop objects not seen recently (left the frame)
    tracked_objects = [o for o in tracked_objects
                        if frame_count - o.last_seen <= STALE_FRAMES]

    # ── Draw only objects updated THIS frame (avoid drawing stale boxes) ──
    class_counts = {}
    for obj in tracked_objects:
        if obj.last_seen != frame_count:
            continue
        x1, y1, x2, y2 = obj.box
        label = obj.label
        status_tag = " [FIXED]" if obj.is_locked else f" [{len(obj.history)}/{VOTE_WINDOW}]"
        color = CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)

        class_counts[label] = class_counts.get(label, 0) + 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"#{obj.id} {label}{status_tag}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # HUD — show counts top-left
    hud_y = 25
    for cls_name, count in class_counts.items():
        hud_text = f"{cls_name}: {count}"
        color    = CLASS_COLORS.get(cls_name.lower(), DEFAULT_COLOR)
        cv2.putText(img, hud_text, (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        hud_y += 28

    # Save frame if enabled
    if SAVE_OUTPUT:
        out_path = f"{OUTPUT_DIR}/frame_{frame_count:05d}.jpg"
        cv2.imwrite(out_path, img)

    frame_count += 1

    # Display
    cv2.imshow("Drone Detection", img)
    key = cv2.waitKey(0 if paused else 1) & 0xFF

    if key == ord('q'):
        print("Quitting...")
        break
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
