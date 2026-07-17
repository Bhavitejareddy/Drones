"""
STEP 0 — Offline Augmentation of Raw Drone Images
===================================================
Run this BEFORE step1_crop.py.

Reads raw images from RAW_DIR/<tier>/, generates augmented copies,
and saves them into a new AUG_DIR/<tier>/ folder (originals + augmented
combined), so the rest of the pipeline (step1_crop.py onward) can just
point RAW_DIR at AUG_DIR.

Uses only opencv-python, numpy and Pillow — all offline, no internet
needed once installed.

Run:
    python step0_augment.py
"""

import os
import sys
import random
from pathlib import Path

import cv2
import numpy as np

from config import RAW_DIR, SUPPORTED_EXTS, TIERS, separator, check_placeholders

# ═══════════════════════════════════════════════════════════
#  SETTINGS
# ═══════════════════════════════════════════════════════════

# Where augmented images will be written (originals are copied in too)
AUG_DIR = os.path.join(os.path.dirname(RAW_DIR), "raw_images_augmented")

# How many augmented variants to generate PER ORIGINAL image, per tier.
# 500 originals * 5 = 3000 total per tier (a good target range).
# Edit these numbers before each run — e.g. give harder-to-detect tiers
# (like nano) a higher multiplier than easier ones.
AUG_PER_IMAGE = {
    "nano":   5,
    "micro":  5,
    "small":  5,
    "medium": 5,
    "large":  5,
}

SEED = 42


# ═══════════════════════════════════════════════════════════
#  AUGMENTATION FUNCTIONS
#  All operate on a full raw scene (drone + background), so the
#  object must stay fully inside the frame for YOLO to detect it
#  in step1_crop.py.
# ═══════════════════════════════════════════════════════════

def rotate_image(img, max_angle=15):
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def flip_image(img):
    # horizontal flip only — vertical flip would put "sky" at the bottom,
    # which is unrealistic for drone footage
    return cv2.flip(img, 1)


def adjust_brightness_contrast(img, b_range=(0.7, 1.3), c_range=(0.7, 1.3)):
    brightness = random.uniform(*b_range)
    contrast = random.uniform(*c_range)
    img = img.astype(np.float32)
    mean = img.mean()
    img = (img - mean) * contrast + mean * brightness
    return np.clip(img, 0, 255).astype(np.uint8)

def add_gaussian_noise(img, sigma_range=(3, 12)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def slight_blur(img, k_choices=(3, 5)):
    k = random.choice(k_choices)
    return cv2.GaussianBlur(img, (k, k), 0)


def translate_image(img, max_shift_frac=0.08):
    h, w = img.shape[:2]
    tx = random.uniform(-max_shift_frac, max_shift_frac) * w
    ty = random.uniform(-max_shift_frac, max_shift_frac) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def zoom_image(img, zoom_range=(0.9, 1.15)):
    """Scale-jitter on the FULL scene (safe: tier label comes from the
    original scene proportions, and Step 1 resizes every crop to a
    fixed 224x224 regardless anyway)."""
    h, w = img.shape[:2]
    z = random.uniform(*zoom_range)
    nh, nw = int(h * z), int(w * z)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if z >= 1.0:
        y0 = (nh - h) // 2
        x0 = (nw - w) // 2
        return resized[y0:y0 + h, x0:x0 + w]
    else:
        canvas = cv2.copyMakeBorder(
            resized,
            (h - nh) // 2, h - nh - (h - nh) // 2,
            (w - nw) // 2, w - nw - (w - nw) // 2,
            borderType=cv2.BORDER_REFLECT_101,
        )
        return cv2.resize(canvas, (w, h))


def random_gamma(img, gamma_range=(0.7, 1.4)):
    gamma = random.uniform(*gamma_range)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


# Pool of augmentations. Each augmented image applies a random SUBSET
# (1-3 of these) so variants don't all look identical.
AUG_FUNCTIONS = [
    rotate_image,
    flip_image,
    adjust_brightness_contrast,
    add_gaussian_noise,
    slight_blur,
    translate_image,
    zoom_image,
    random_gamma,
]


def apply_random_augmentation(img):
    n_ops = random.randint(1, 3)
    ops = random.sample(AUG_FUNCTIONS, n_ops)
    out = img.copy()
    for op in ops:
        out = op(out)
    return out


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def run_step0():
    separator("STEP 0 — Offline Augmentation")
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"  Source : {RAW_DIR}")
    print(f"  Output : {AUG_DIR}")
    print(f"  Augmented copies per original (per tier): {AUG_PER_IMAGE}\n")

    for tier in TIERS:
        in_dir = os.path.join(RAW_DIR, tier)
        out_dir = os.path.join(AUG_DIR, tier)
        n_aug = AUG_PER_IMAGE.get(tier, 5)

        if not os.path.exists(in_dir):
            print(f"  [{tier}] SKIP — folder not found: {in_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        files = sorted([f for f in Path(in_dir).iterdir() if f.suffix.lower() in SUPPORTED_EXTS])
        print(f"  [{tier}] {len(files)} original images found  (x{n_aug} augmentations each)")

        saved = 0
        for fpath in files:
            img = cv2.imread(str(fpath))
            if img is None:
                continue

            # 1. Copy the original through untouched
            orig_out = os.path.join(out_dir, f"orig_{fpath.name}")
            cv2.imwrite(orig_out, img)
            saved += 1

            # 2. Generate N augmented variants
            for i in range(n_aug):
                aug_img = apply_random_augmentation(img)
                aug_name = f"aug{i}_{fpath.stem}.jpg"
                cv2.imwrite(os.path.join(out_dir, aug_name), aug_img)
                saved += 1

        print(f"  [{tier}] ✓ {saved} total images written to {out_dir}\n")

    print("  ✓ Step 0 complete.")
    print(f"\n  NEXT STEP: point RAW_DIR in config.py to:\n    {AUG_DIR}")
    print("  then run step1_crop.py as usual.")


if __name__ == "__main__":
    check_placeholders([RAW_DIR])
    os.makedirs(AUG_DIR, exist_ok=True)
    run_step0()
