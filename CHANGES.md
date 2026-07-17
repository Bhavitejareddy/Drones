# Step 5 Inference Pipeline — Change Log

**Files affected:** `step5_infer.py`, `config.py`
**Purpose:** Document what was changed, where in the code it lives, and why it was needed.

---

## 1. Tracking removed

**Where:** `ObjectTracker`, `TrackerPool`, `LabelSmoother`, `iou()` deleted entirely.
`_process_frame()` no longer takes a `tracker` argument.

**Why:** Requirement — "don't need tracking." Every frame is now scored independently by
Stage 1 / Stage 2, with no attempt to persist an object's identity across frames.

---

## 2. Bounding boxes always show Stage-1 confidence

**Where:** `draw_box()`, and the `stage1_conf` field on each detection entry built
inside `_process_frame()`.

**Why:** Previously, a drone box's on-screen confidence came from Stage 2 (the size
classifier) once a tier was assigned. Requirement was to always show the Stage-1
detector's confidence on the box, regardless of whether Stage 2 ran.

---

## 3. Stage-2 gating — three checks instead of one

**Where:** Inside `_process_frame()`, in the `if stage1_label != "drone":` / `else:` branch.

| Gate | Config variable | Default | Catches |
|---|---|---|---|
| Detection confidence | `YOLO_CONF_GATE` | 0.55 | Weak/uncertain Stage-1 detections |
| Bounding-box area | `BBOX_AREA_THRESHOLD` | 1600 px² | Drone too small/far to tell tiers apart |
| Sharpness (Laplacian variance) | `LAPLACIAN_VAR_GATE` | 80.0 | Blurry/motion-smeared crops that would fool Stage 2 |

All three must pass before the crop is sent to Stage 2 (`classify_size()`). If any fail,
the box is labeled `"drone (size N/A)"`.

**Why:** A single area threshold wasn't catching low-confidence or blurry detections that
would otherwise get a (likely wrong) size-tier guess. `laplacian_variance()` (new helper
function, uses `cv2.Laplacian(...).var()`) was added specifically for the blur check.

---

## 4. Confidence visualization panel

**Where:** New function `draw_confidence_panel()`, called from `_draw_frame()`.
Drawn in the top-right corner of the frame/image.

**What it shows, per detected object:**
- The Stage-1 score for **every** class — drone / bird / aircraft / helicopter — not
  just the winning one (see item 5 below for how the extra scores are obtained).
- If the object is a drone that cleared the Stage-2 gates: the full tier breakdown
  (nano / micro / small / medium / large probabilities), not just the winning tier.

**Why:** Requirement was to visualize confidence across all Stage-1 classes and all
Stage-2 tiers, not just the single predicted label — useful for spotting ambiguous or
borderline detections.

---

## 5. True 4-class Stage-1 scoring (raw pre-NMS forward pass)

**Where:** New function `raw_stage1_predictions()`, plus `init_stage1_predictor()` to
warm up `yolo.predictor` once at startup (called in `run_step5()`).

**Why:** Ultralytics' normal `.predict()` call runs NMS internally, which keeps only the
*winning* class per box and discards the other three scores. To show real confidence
for all 4 classes on every box (item 4), Stage 1 is run manually:
1. `predictor.model(im)` — raw forward pass, before NMS, output shape
   `(batch, 4+nc, num_anchors)`.
2. Filter anchors by `max(class_scores) >= conf_thres`.
3. Run NMS ourselves (`torchvision.ops.nms`), ranked by the winning class's score —
   but keep the full class-score vector for every surviving box, instead of discarding it.
4. Rescale boxes back to the original frame with Ultralytics' own `scale_boxes()`.

**Toggle:** `USE_FULL_STAGE1_SCORES` in `config.py` (default `True`).
Set to `False` to skip this and fall back to the faster single-class `.predict()` path.

**Safety net:** Wrapped in `try/except` inside `_process_frame()`. If a future
Ultralytics version changes the model's internal output format, this raises once, a
warning is printed a single time (`_WARNED_RAW_FALLBACK` flag), and the pipeline falls
back to single-class scoring automatically — it does not crash.

**Cost:** One raw forward pass replaces what would otherwise be two (`.predict()` +
a second raw pass), so this is not much slower than the original per-frame cost — but
manual NMS/box-decoding in Python is less optimized than Ultralytics' internal C-level
NMS. Worth benchmarking against `USE_FULL_STAGE1_SCORES = False` on CPU/embedded targets.

---

## 6. Images vs. video/webcam handling

**Where:** `_run_step5_images()` (images) vs. the main loop in `run_step5()` (video/webcam).

- **Images / folder of images:** each image processed fully independently — no
  smoothing of any kind. (`_run_step5_images()`)
- **Video / webcam:** no per-object smoothing (tracking is gone — item 1), but a
  **global sliding-window majority vote** was added:
  - Frames 0–19: the raw, unsmoothed per-frame prediction is shown as-is.
  - Frame 20 onward: a `"Stable (last 20fr): ..."` line appears (drawn in
    `draw_confidence_panel()`), computed as the majority vote of each frame's dominant
    label (`_frame_dominant_label()`) over the last `SMOOTH_WINDOW` frames
    (`SMOOTH_FRAMES` in `config.py`, default 20). The window slides every frame, so the
    stable label updates whenever the majority changes.
  - Individual boxes still always show their own instantaneous raw label — the
    "stable" line is a separate overlay, not an override, so multi-object scenes still
    make sense.

**Why:** Requirement was explicit — images get normal independent predictions; video
should show raw predictions for the first 20 frames, then switch to a majority vote
over a 20-frame sliding window that updates as the majority changes.

---

## 7. `config.py` — new/relevant variables

| Variable | Default | Purpose |
|---|---|---|
| `YOLO_CONF_GATE` | 0.55 | Stage-2 gate — min Stage-1 confidence |
| `LAPLACIAN_VAR_GATE` | 80.0 | Stage-2 gate — min sharpness |
| `BBOX_AREA_THRESHOLD` | 1600 | Stage-2 gate — min box area (px²) |
| `USE_FULL_STAGE1_SCORES` | `True` | Toggle raw 4-class scoring vs. fast single-class |
| `SMOOTH_FRAMES` | 20 | Sliding-window size for the video "stable" prediction |

All other variables (`WEIGHTS_PATH`, `CLASSIFIER_PATH`, `INFER_SOURCE`, etc.) are
reconstructed from what `step5_infer.py` imports — `WEIGHTS_PATH`, `CLASSIFIER_PATH`,
and `INFER_SOURCE` are placeholders and need to be filled in before running.

---

## Still open / not implemented

- The 4-class Stage-1 scores are a genuine per-anchor softmax/sigmoid readout, but the
  matching of `class_probs` to a given box relies on the raw path being active
  (`USE_FULL_STAGE1_SCORES = True`); when it falls back, `class_probs` only contains the
  one known class.
- No object tracking means no persistent per-object identity, count-over-time, or
  trajectory information — by design, per the "don't need tracking" requirement.
