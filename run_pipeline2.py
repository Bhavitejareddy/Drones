"""
DRONE CLASSIFICATION PIPELINE — CONFIGURE AND RUN
===================================================
Just fill in the paths below, then run:
    python run_pipeline.py

That's it. No command line arguments needed.
"""

# ═══════════════════════════════════════════════════════════
#  CONFIGURE YOUR PATHS HERE — change these to match your PC
# ═══════════════════════════════════════════════════════════

# Folder containing your raw images, one subfolder per tier
# Example: r"C:\Users\Bhavi\Desktop\drone_project\raw"
RAW_DIR = r"C:\path\to\your\raw_images"

# Where cropped 224x224 images will be saved
CROPS_DIR = r"C:\path\to\your\crops"

# Where train/val/test split will be saved
SPLIT_DIR = r"C:\path\to\your\dataset_split"

# Where trained models and results will be saved
OUTPUT_DIR = r"C:\path\to\your\output"

# Path to your YOLO weights file (best.pt)
WEIGHTS_PATH = r"C:\path\to\your\best.pt"

# Path to pretrained ResNet50 weights (resnet50_imagenet.pth)
RESNET_WEIGHTS = r"C:\path\to\your\resnet50_imagenet.pth"

# Path to pretrained EfficientNetB0 weights (efficientnetb0_imagenet.pth)
EFFICIENT_WEIGHTS = r"C:\path\to\your\efficientnetb0_imagenet.pth"

# ═══════════════════════════════════════════════════════════
#  CHOOSE WHICH STEPS TO RUN
#  Set True to run, False to skip
# ═══════════════════════════════════════════════════════════

RUN_STEP1_CROP  = True    # Crop raw images using YOLO
RUN_STEP2_SPLIT = True    # Split into train/val/test
RUN_STEP3_TRAIN = True    # Train ResNet50 + EfficientNetB0
RUN_STEP4_EVAL  = True    # Evaluate and compare both models
RUN_STEP5_INFER = False   # Run full 2-stage inference on a video/image/folder
                           # Set True only after Steps 1-4 are complete

# Which model to train: "both" / "resnet50" / "efficientnetb0"
TRAIN_MODEL = "both"

# Training epochs (reduce for a quick test run)
P1_EPOCHS = 15   # Phase 1 — frozen backbone
P2_EPOCHS = 20   # Phase 2 — fine-tuning

# ═══════════════════════════════════════════════════════════
#  STEP 5 CONFIG — 2-Stage Inference (fill when ready)
# ═══════════════════════════════════════════════════════════

# Path to your trained Stage 2 classifier (best_model.pth from Step 3/4)
# Use whichever model won in Step 4 comparison
# Example: r"C:\drone_project\output\efficientnetb0\best_model.pth"
CLASSIFIER_PATH = r"C:\path\to\output\efficientnetb0\best_model.pth"

# Which classifier architecture matches the weights above
# "efficientnetb0" or "resnet50"
CLASSIFIER_TYPE = "efficientnetb0"

# Input source for inference:
# - Path to a video file:  r"C:\videos\test.mp4"
# - Path to image folder:  r"C:\images\test_images"
# - Single image:          r"C:\images\test.jpg"
# - Webcam:                0
INFER_SOURCE = r"C:\path\to\your\video_or_images"

# Where to save annotated output frames/video
INFER_OUTPUT_DIR = r"C:\path\to\your\inference_output"

# Minimum bbox pixel area to attempt size classification
# Boxes smaller than this → labelled "drone (size N/A)"
# Tune based on your camera resolution
# 40x40 px = 1600 is a good starting point
BBOX_AREA_THRESHOLD = 1600

# Temporal smoothing — majority vote over N frames per tracked object
# Higher = more stable label, slower to update (20 recommended)
SMOOTH_FRAMES = 20

# Show output window during inference (set False for headless/server)
SHOW_WINDOW = True

# Save annotated frames to INFER_OUTPUT_DIR
SAVE_INFER_OUTPUT = True

# ═══════════════════════════════════════════════════════════
#  DO NOT EDIT BELOW THIS LINE
# ═══════════════════════════════════════════════════════════

import os
import sys
import csv
import time
import copy
import random
import shutil
from pathlib import Path
from collections import Counter

# ───────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ───────────────────────────────────────────────────────────
TIERS          = ["nano", "micro", "small", "medium", "large"]
IMAGE_SIZE     = 224
NUM_CLASSES    = 5
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def separator(title):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")

# ───────────────────────────────────────────────────────────
# STEP 1 — CROP
# ───────────────────────────────────────────────────────────
CONF_THRESH       = 0.35
MAX_BOX_AREA_FRAC = 0.15
MIN_BOX_AREA_PX   = 20 * 20
MAX_ASPECT_RATIO  = 4.0
MARGIN_FRAC       = 0.15

def is_valid_box(x1, y1, x2, y2, fw, fh):
    w, h = x2-x1, y2-y1
    area = w * h
    if area > MAX_BOX_AREA_FRAC * fw * fh: return False, "too_large"
    if area < MIN_BOX_AREA_PX:             return False, "too_small"
    if w <= 0 or h <= 0:                   return False, "degenerate"
    if max(w/h, h/w) > MAX_ASPECT_RATIO:   return False, "bad_aspect"
    if h > 0.5*fh and w < 0.4*h:          return False, "person_shape"
    return True, "ok"

def crop_and_resize(img, x1, y1, x2, y2):
    import cv2
    ih, iw = img.shape[:2]
    bw, bh = x2-x1, y2-y1
    cx1 = max(0, int(x1 - bw*MARGIN_FRAC))
    cy1 = max(0, int(y1 - bh*MARGIN_FRAC))
    cx2 = min(iw, int(x2 + bw*MARGIN_FRAC))
    cy2 = min(ih, int(y2 + bh*MARGIN_FRAC))
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0: return None
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

def build_contact_sheet(out_dir, tier):
    import cv2
    import numpy as np
    files = sorted([f for f in Path(out_dir).glob("*.jpg") if not f.name.startswith("_")])
    if not files: return
    imgs = [cv2.resize(cv2.imread(str(f)), (128,128)) for f in files if cv2.imread(str(f)) is not None]
    cols, cell = 8, 130
    rows = (len(imgs)+cols-1)//cols
    grid = np.ones((rows*cell, cols*cell, 3), dtype="uint8") * 240
    for i, im in enumerate(imgs):
        r, c = i//cols, i%cols
        grid[r*cell+1:r*cell+129, c*cell+1:c*cell+129] = im
    path = os.path.join(out_dir, f"_contact_sheet_{tier}.jpg")
    cv2.imwrite(path, grid)
    print(f"  Contact sheet → {path}")
    print(f"  ⚠ OPEN THIS FILE and delete any bad crops before running Step 2!")

def run_step1():
    separator("STEP 1 — Crop Raw Images Using YOLO")
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        print("ERROR: ultralytics or opencv not installed.")
        print("       Run: pip install ultralytics opencv-python")
        sys.exit(1)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: YOLO weights not found: {WEIGHTS_PATH}")
        sys.exit(1)

    print(f"Loading YOLO model from {WEIGHTS_PATH} ...")
    model = YOLO(WEIGHTS_PATH)
    print(f"Classes: {model.names}\n")

    for tier in TIERS:
        in_dir  = os.path.join(RAW_DIR,   tier)
        out_dir = os.path.join(CROPS_DIR, tier)

        if not os.path.exists(in_dir):
            print(f"  [{tier}] SKIP — folder not found: {in_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        files = sorted([f for f in Path(in_dir).iterdir() if f.suffix.lower() in SUPPORTED_EXTS])
        print(f"\n  [{tier}] {len(files)} images found")

        counts = Counter()
        for idx, fpath in enumerate(files):
            img = cv2.imread(str(fpath))
            if img is None:
                counts["read_error"] += 1; continue

            fh, fw = img.shape[:2]
            results = model.predict(str(fpath), conf=CONF_THRESH, verbose=False)
            r = results[0]

            if r.boxes is None or len(r.boxes) == 0:
                counts["no_detection"] += 1; continue

            confs = r.boxes.conf.tolist()
            boxes = r.boxes.xyxy.tolist()
            best  = max(range(len(confs)), key=lambda i: confs[i])
            x1,y1,x2,y2 = boxes[best]
            conf = confs[best]

            ok, reason = is_valid_box(x1,y1,x2,y2,fw,fh)
            if not ok:
                counts[f"rejected_{reason}"] += 1; continue

            crop = crop_and_resize(img, x1,y1,x2,y2)
            if crop is None:
                counts["empty_crop"] += 1; continue

            out_name = f"{tier}_{fpath.stem}.jpg"
            cv2.imwrite(os.path.join(out_dir, out_name), crop)
            counts["kept"] += 1

            if (idx+1) % 50 == 0:
                print(f"    {idx+1}/{len(files)} processed ... kept={counts['kept']}")

        print(f"\n  [{tier}] Results:")
        for status, count in counts.most_common():
            print(f"    {status:30s}: {count}")

        build_contact_sheet(out_dir, tier)

        if counts["kept"] < 300:
            print(f"  ⚠ WARNING: Only {counts['kept']} crops — consider adding more images for [{tier}]")
        else:
            print(f"  ✓ {counts['kept']} crops saved to {out_dir}")


# ───────────────────────────────────────────────────────────
# STEP 2 — SPLIT
# ───────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
SEED       = 42

def run_step2():
    separator("STEP 2 — Split into Train / Val / Test")
    random.seed(SEED)
    print(f"  Split: {TRAIN_FRAC:.0%} train / {VAL_FRAC:.0%} val / {1-TRAIN_FRAC-VAL_FRAC:.0%} test\n")

    for tier in TIERS:
        src = Path(CROPS_DIR) / tier
        if not src.exists():
            print(f"  [{tier}] SKIP — {src} not found"); continue

        files = sorted([
            f for f in src.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS and not f.name.startswith("_")
        ])
        if not files:
            print(f"  [{tier}] SKIP — no images"); continue

        random.shuffle(files)
        n       = len(files)
        n_train = int(n * TRAIN_FRAC)
        n_val   = int(n * VAL_FRAC)

        splits = {
            "train": files[:n_train],
            "val":   files[n_train:n_train+n_val],
            "test":  files[n_train+n_val:],
        }
        for split_name, split_files in splits.items():
            dest = Path(SPLIT_DIR) / split_name / tier
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dest/f.name)

        print(
            f"  {tier:8s}  total={n:4d}  "
            f"train={len(splits['train']):4d}  "
            f"val={len(splits['val']):4d}  "
            f"test={len(splits['test']):4d}"
        )
    print("\n  ✓ Split complete")


# ───────────────────────────────────────────────────────────
# STEP 3 — TRAIN
# ───────────────────────────────────────────────────────────
BATCH_TRAIN  = 16
BATCH_VAL    = 32
NUM_WORKERS  = 0
WEIGHT_DECAY = 1e-4
P1_LR        = 1e-3
P2_LR        = 1e-5

def get_transforms():
    from torchvision import transforms
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.85,1.15)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf

def load_data(train_tf, val_tf):
    from torchvision import datasets
    from torch.utils.data import DataLoader
    train_ds = datasets.ImageFolder(os.path.join(SPLIT_DIR,"train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(SPLIT_DIR,"val"),   transform=val_tf)
    tr = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=NUM_WORKERS)
    vl = DataLoader(val_ds,   batch_size=BATCH_VAL,   shuffle=False, num_workers=NUM_WORKERS)
    print(f"  Classes : {train_ds.classes}")
    print(f"  Train   : {len(train_ds)} images")
    print(f"  Val     : {len(val_ds)} images\n")
    return tr, vl, train_ds.classes

def build_resnet50():
    import torch
    from torchvision import models
    import torch.nn as nn
    m = models.resnet50(weights=None)
    if os.path.exists(RESNET_WEIGHTS):
        m.load_state_dict(torch.load(RESNET_WEIGHTS, map_location="cpu"))
        print(f"  ResNet50: loaded {RESNET_WEIGHTS}")
    else:
        print(f"  ResNet50: no pretrained weights found — training from scratch")
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, NUM_CLASSES))
    return m

def build_efficientnetb0():
    import torch
    from torchvision import models
    import torch.nn as nn
    m = models.efficientnet_b0(weights=None)
    if os.path.exists(EFFICIENT_WEIGHTS):
        m.load_state_dict(torch.load(EFFICIENT_WEIGHTS, map_location="cpu"))
        print(f"  EfficientNetB0: loaded {EFFICIENT_WEIGHTS}")
    else:
        print(f"  EfficientNetB0: no pretrained weights found — training from scratch")
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, NUM_CLASSES))
    return m

def run_epoch(model, loader, criterion, optimizer, is_train):
    import torch
    model.train() if is_train else model.eval()
    loss_sum = correct = total = 0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            out  = model(imgs)
            loss = criterion(out, labels)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += (out.argmax(1)==labels).sum().item()
            total    += imgs.size(0)
    return loss_sum/total, correct/total

def train_phase(model, tr, vl, crit, opt, sch, epochs, phase, mname, log, out_dir):
    import torch
    best_acc, best_wt = 0.0, copy.deepcopy(model.state_dict())
    best_path = os.path.join(out_dir, mname, "best_model.pth")
    print(f"\n  [{mname}] {phase} — {epochs} epochs")
    print(f"  {'─'*48}")
    for ep in range(1, epochs+1):
        t0 = time.time()
        tl, ta = run_epoch(model, tr, crit, opt, True)
        vl_, va = run_epoch(model, vl, crit, None, False)
        sch.step(vl_)
        mark = ""
        if va > best_acc:
            best_acc = va
            best_wt  = copy.deepcopy(model.state_dict())
            torch.save(best_wt, best_path)
            mark = "  ← best"
        print(f"  Ep {ep:2d}/{epochs}  loss={tl:.3f} acc={ta:.3f}  |  val_loss={vl_:.3f} val_acc={va:.3f}  ({time.time()-t0:.0f}s){mark}")
        log.append([phase, ep, round(tl,4), round(ta,4), round(vl_,4), round(va,4)])
    print(f"  [{mname}] Best val acc: {best_acc:.4f}")
    return best_wt, best_acc

def train_one_model(mname, model, freeze_fn, unfreeze_fn, tr, vl, out_dir):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    os.makedirs(os.path.join(out_dir, mname), exist_ok=True)
    log  = []
    crit = nn.CrossEntropyLoss()

    freeze_fn(model)
    opt1 = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=P1_LR, weight_decay=WEIGHT_DECAY)
    sch1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=3, factor=0.5)
    bwt, acc1 = train_phase(model, tr, vl, crit, opt1, sch1, P1_EPOCHS, "Phase1", mname, log, out_dir)

    model.load_state_dict(bwt)
    unfreeze_fn(model)
    opt2 = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=P2_LR, weight_decay=WEIGHT_DECAY)
    sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=4, factor=0.5)
    bwt2, acc2 = train_phase(model, tr, vl, crit, opt2, sch2, P2_EPOCHS, "Phase2", mname, log, out_dir)

    model.load_state_dict(bwt2)
    torch.save(model.state_dict(), os.path.join(out_dir, mname, "final_model.pth"))

    with open(os.path.join(out_dir, mname, "training_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase","epoch","train_loss","train_acc","val_loss","val_acc"])
        w.writerows(log)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p1 = [r for r in log if r[0]=="Phase1"]
        p2 = [r for r in log if r[0]=="Phase2"]
        fig,(a1,a2) = plt.subplots(1,2,figsize=(12,4))
        for rows,lbl,col in [(p1,"Phase1","steelblue"),(p2,"Phase2","coral")]:
            if not rows: continue
            ep=[r[1] for r in rows]
            a1.plot(ep,[r[2] for r in rows],"--",color=col,alpha=0.6)
            a1.plot(ep,[r[4] for r in rows],"-", color=col,label=lbl)
            a2.plot(ep,[r[3] for r in rows],"--",color=col,alpha=0.6)
            a2.plot(ep,[r[5] for r in rows],"-", color=col,label=lbl)
        a1.set_title("Loss");a1.legend();a2.set_title("Accuracy");a2.legend()
        plt.suptitle(f"{mname} — Training"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir,mname,"training_curves.png"),dpi=150); plt.close()
    except: pass

    return max(acc1, acc2)

def freeze_resnet(m):
    for n,p in m.named_parameters(): p.requires_grad=("fc" in n)

def unfreeze_resnet(m):
    for n,p in m.named_parameters():
        if any(g in n for g in ["layer4","layer3","fc"]): p.requires_grad=True

def freeze_efficient(m):
    for n,p in m.named_parameters(): p.requires_grad=("classifier" in n)

def unfreeze_efficient(m):
    for n,p in m.named_parameters():
        if "classifier" in n: p.requires_grad=True
        elif "features" in n:
            try:
                if int(n.split(".")[1]) >= 6: p.requires_grad=True
            except: pass

def run_step3():
    separator("STEP 3 — Train Classifier")
    import torch
    print(f"  PyTorch : {torch.__version__}  |  Device: CPU")
    print(f"  Model   : {TRAIN_MODEL}\n")

    train_tf, val_tf  = get_transforms()
    tr, vl, classes   = load_data(train_tf, val_tf)
    results = {}

    with open(os.path.join(OUTPUT_DIR,"class_names.txt"),"w") as f:
        [f.write(f"{i} {c}\n") for i,c in enumerate(classes)]

    if TRAIN_MODEL in ("both","resnet50"):
        print("\n  ── Training ResNet50 ──")
        m = build_resnet50()
        results["resnet50"] = train_one_model("resnet50", m, freeze_resnet, unfreeze_resnet, tr, vl, OUTPUT_DIR)

    if TRAIN_MODEL in ("both","efficientnetb0"):
        print("\n  ── Training EfficientNetB0 ──")
        m = build_efficientnetb0()
        results["efficientnetb0"] = train_one_model("efficientnetb0", m, freeze_efficient, unfreeze_efficient, tr, vl, OUTPUT_DIR)

    print(f"\n  Val Accuracy Summary:")
    winner = max(results, key=results.get)
    for name, acc in results.items():
        mark = "  ← winner" if name==winner else ""
        print(f"  {name:20s}  {acc*100:.1f}%{mark}")
    print(f"\n  ✓ Training complete. Run Step 4 to evaluate on test set.")


# ───────────────────────────────────────────────────────────
# STEP 4 — EVALUATE
# ───────────────────────────────────────────────────────────
def evaluate_model(mname, model_path, classes):
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader

    # Load model
    if "resnet" in mname:
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, NUM_CLASSES))
    else:
        m = models.efficientnet_b0(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, NUM_CLASSES))

    m.load_state_dict(torch.load(model_path, map_location="cpu"))
    m.eval()

    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds     = datasets.ImageFolder(os.path.join(SPLIT_DIR,"test"), transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_VAL, shuffle=False, num_workers=NUM_WORKERS)

    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            preds.extend(m(imgs).argmax(1).tolist())
            labels.extend(lbls.tolist())

    # Metrics
    n  = len(classes)
    cm = [[0]*n for _ in range(n)]
    pc = [0]*n; pt = [0]*n
    for p,l in zip(preds,labels):
        cm[l][p]+=1; pt[l]+=1
        if p==l: pc[l]+=1
    overall  = sum(pc)/len(labels)
    per_acc  = [pc[i]/pt[i] if pt[i]>0 else 0.0 for i in range(n)]

    # Print report
    lines = [f"\n  [{mname}]  Overall accuracy: {overall*100:.1f}%\n"]
    lines.append(f"  {'Tier':10s}  {'N':>5s}  {'Acc':>7s}  Bar")
    lines.append(f"  {'─'*45}")
    for i,cls in enumerate(classes):
        bar = "█"*int(per_acc[i]*20) + "░"*(20-int(per_acc[i]*20))
        lines.append(f"  {cls:10s}  {pt[i]:5d}  {per_acc[i]*100:6.1f}%  {bar}")
    lines.append(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    lines.append(f"  {'':12s}" + "".join(f"{c:>9s}" for c in classes))
    for i,rc in enumerate(classes):
        lines.append(f"  {rc:12s}" + "".join(f"{cm[i][j]:9d}" for j in range(n)))
    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.join(OUTPUT_DIR,mname), exist_ok=True)
    with open(os.path.join(OUTPUT_DIR,mname,"evaluation_report.txt"),"w") as f:
        f.write(report)

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        arr  = np.array(cm,dtype=float)
        norm = arr / arr.sum(axis=1,keepdims=True).clip(min=1)
        fig,ax = plt.subplots(figsize=(7,6))
        im = ax.imshow(norm,cmap="Blues",vmin=0,vmax=1)
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        ax.set_xticks(range(n)); ax.set_xticklabels(classes,rotation=45,ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{mname} — Confusion Matrix")
        for i in range(n):
            for j in range(n):
                ax.text(j,i,str(int(arr[i,j])),ha="center",va="center",
                        color="white" if norm[i,j]>0.6 else "black",fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR,mname,"confusion_matrix.png"),dpi=150)
        plt.close()
    except: pass

    return overall, per_acc

def run_step4():
    separator("STEP 4 — Evaluate & Compare")
    results = {}

    # Load class names
    class_file = os.path.join(OUTPUT_DIR,"class_names.txt")
    if os.path.exists(class_file):
        with open(class_file) as f:
            classes = [line.strip().split()[1] for line in f if line.strip()]
    else:
        classes = TIERS

    for mname in ["resnet50","efficientnetb0"]:
        mpath = os.path.join(OUTPUT_DIR, mname, "best_model.pth")
        if not os.path.exists(mpath):
            print(f"  [{mname}] SKIP — {mpath} not found"); continue
        print(f"\n  Evaluating {mname} ...")
        overall, per_acc = evaluate_model(mname, mpath, classes)
        results[mname] = {"overall": overall, "per_acc": per_acc}

    if len(results) > 1:
        winner = max(results, key=lambda x: results[x]["overall"])
        lines  = [f"\n{'═'*55}", f"  FINAL COMPARISON", f"{'═'*55}\n"]
        for name, res in results.items():
            acc  = res["overall"]
            bar  = "█"*int(acc*30)
            mark = "  ← USE THIS MODEL" if name==winner else ""
            lines.append(f"  {name:20s}  {acc*100:5.1f}%  {bar}{mark}")
        lines.append(f"\n  Best model path:")
        lines.append(f"  {os.path.join(OUTPUT_DIR, winner, 'best_model.pth')}")
        report = "\n".join(lines)
        print(report)
        with open(os.path.join(OUTPUT_DIR,"comparison_report.txt"),"w") as f:
            f.write(report)
        print(f"\n  ✓ Comparison report → {os.path.join(OUTPUT_DIR,'comparison_report.txt')}")


# ───────────────────────────────────────────────────────────
# STEP 5 — FULL 2-STAGE INFERENCE PIPELINE
# Matches your diagram exactly:
#
#  Image/Video
#      │
#  Stage 1: YOLOv12n → drone / bird / aircraft / helicopter
#      │
#  class == drone?
#  ├── No  → final label = bird / aircraft / helicopter
#  └── Yes → bbox area >= threshold?
#            ├── No  → "drone (size N/A)"
#            └── Yes → Stage 2 classifier → nano/micro/small/medium/large
# ───────────────────────────────────────────────────────────

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
    """Run Stage 2 on a cropped drone region. Returns (tier_label, confidence)."""
    import torch
    # BGR → RGB for PIL
    crop_rgb = crop_bgr[:, :, ::-1].copy()
    tensor   = tf(crop_rgb).unsqueeze(0)   # (1, 3, 224, 224)
    with torch.no_grad():
        out   = classifier(tensor)
        probs = torch.softmax(out, dim=1)[0]
        idx   = probs.argmax().item()
    return class_names[idx], float(probs[idx])

# ── IoU helper ───────────────────────────────────────────────
def iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

# ── Simple per-object label smoother ────────────────────────
class LabelSmoother:
    """Keeps rolling label history per tracked object. Returns majority vote."""
    def __init__(self, window=20):
        from collections import deque
        self.window   = window
        self.history  = deque(maxlen=window)

    def update(self, label):
        self.history.append(label)

    @property
    def stable(self):
        if not self.history: return None
        return Counter(self.history).most_common(1)[0][0]

    @property
    def count(self):
        return len(self.history)

# ── Tracker ──────────────────────────────────────────────────
class ObjectTracker:
    _next_id = 1
    IOU_THRESH  = 0.25
    MAX_MISSED  = 15
    ALPHA       = 0.35   # bbox smoothing

    def __init__(self, box, label):
        self.id      = ObjectTracker._next_id
        ObjectTracker._next_id += 1
        self.box     = list(map(float, box))
        self.smoother= LabelSmoother(SMOOTH_FRAMES)
        self.smoother.update(label)
        self.missed  = 0
        self.age     = 1

    def update(self, box, label):
        for i in range(4):
            self.box[i] = self.ALPHA*box[i] + (1-self.ALPHA)*self.box[i]
        self.smoother.update(label)
        self.missed = 0
        self.age   += 1

    def mark_missed(self):
        self.missed += 1

    @property
    def ibox(self):
        return [int(v) for v in self.box]

    @property
    def visible(self):
        return self.age >= 5 and self.missed == 0

class TrackerPool:
    def __init__(self):
        self.pool = []

    def update(self, detections):
        matched_t, matched_d = set(), set()
        for di, (box, label) in enumerate(detections):
            best_iou, best_t = ObjectTracker.IOU_THRESH, None
            for t in self.pool:
                v = iou(box, t.box)
                if v > best_iou:
                    best_iou, best_t = v, t
            if best_t:
                best_t.update(box, label)
                matched_t.add(best_t.id); matched_d.add(di)
        for di, (box, label) in enumerate(detections):
            if di not in matched_d:
                self.pool.append(ObjectTracker(box, label))
        for t in self.pool:
            if t.id not in matched_t: t.mark_missed()
        self.pool = [t for t in self.pool if t.missed < ObjectTracker.MAX_MISSED]
        return self.pool

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

def draw_box(img, x1, y1, x2, y2, label, conf, color, extra=""):
    import cv2
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    text = f"{label} {conf:.2f}{extra}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
    cv2.putText(img, text, (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def run_step5():
    separator("STEP 5 — 2-Stage Inference Pipeline")
    import cv2
    import torch
    from ultralytics import YOLO

    # ── Validate paths
    for label, path in [("WEIGHTS_PATH", WEIGHTS_PATH),
                         ("CLASSIFIER_PATH", CLASSIFIER_PATH)]:
        if "path\\to" in path or "path/to" in path:
            print(f"ERROR: {label} still has placeholder value. Fill it in at the top.")
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
    yolo       = YOLO(WEIGHTS_PATH)
    classifier = load_stage2_classifier()
    infer_tf   = get_infer_transform()

    # ── Open source
    if INFER_SOURCE == 0 or str(INFER_SOURCE).isdigit():
        cap = cv2.VideoCapture(0)
        print("  Source: webcam")
    elif os.path.isfile(str(INFER_SOURCE)):
        cap = cv2.VideoCapture(str(INFER_SOURCE))
        print(f"  Source: video file — {INFER_SOURCE}")
    elif os.path.isdir(str(INFER_SOURCE)):
        # Image folder mode — process each image
        image_files = sorted([
            str(f) for f in Path(INFER_SOURCE).iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS
        ])
        print(f"  Source: image folder — {len(image_files)} images")
        _run_step5_images(image_files, yolo, classifier, infer_tf, size_classes)
        return
    else:
        print(f"ERROR: INFER_SOURCE not found: {INFER_SOURCE}"); sys.exit(1)

    # ── Video / webcam loop
    tracker    = TrackerPool()
    frame_idx  = 0
    paused     = False
    print("\n  Controls: Q=quit  S=save frame  SPACE=pause\n")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret: break

            detections = _process_frame(frame, yolo, classifier, infer_tf,
                                        size_classes, tracker, frame_idx)
            _draw_frame(frame, detections, frame_idx)

            if SAVE_INFER_OUTPUT:
                cv2.imwrite(f"{INFER_OUTPUT_DIR}/frame_{frame_idx:05d}.jpg", frame)

            frame_idx += 1

        if SHOW_WINDOW:
            cv2.imshow("2-Stage Drone Pipeline", frame)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if   key == ord('q'): break
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
    if SHOW_WINDOW: cv2.destroyAllWindows()
    print(f"\n  ✓ Done. {frame_idx} frames processed.")
    if SAVE_INFER_OUTPUT:
        print(f"  Output saved to: {INFER_OUTPUT_DIR}")

def _process_frame(frame, yolo, classifier, infer_tf, size_classes, tracker, frame_idx):
    """
    Core 2-stage logic per frame.
    Returns list of (x1,y1,x2,y2, final_label, conf, color)
    """
    fh, fw = frame.shape[:2]
    results = yolo.predict(frame, conf=0.4, iou=0.45, verbose=False)
    r       = results[0]

    raw_detections = []   # (box, display_label) for tracker

    for box in r.boxes:
        x1,y1,x2,y2 = map(float, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = r.names[cls_id].lower()

        # ── STAGE 1 branch ──────────────────────────────────────
        if label != "drone":
            # bird / aircraft / helicopter — final label from Stage 1
            final_label = label
            final_conf  = conf
            color       = COLORS.get(label, DEFAULT_COLOR)

        else:
            # ── Drone detected — check bbox area gate ───────────
            bbox_area = (x2-x1) * (y2-y1)

            if bbox_area < BBOX_AREA_THRESHOLD:
                # Too small to classify size reliably
                final_label = "drone (size N/A)"
                final_conf  = conf
                color       = COLORS["drone"]

            else:
                # ── STAGE 2: crop → size classifier ─────────────
                cx1 = max(0, int(x1)); cy1 = max(0, int(y1))
                cx2 = min(fw, int(x2)); cy2 = min(fh, int(y2))
                crop = frame[cy1:cy2, cx1:cx2]

                if crop.size == 0:
                    final_label = "drone (size N/A)"
                    final_conf  = conf
                    color       = COLORS["drone"]
                else:
                    size_tier, size_conf = classify_size(
                        classifier, crop, infer_tf, size_classes)
                    final_label = f"drone ({size_tier})"
                    final_conf  = size_conf
                    color       = COLORS.get(size_tier, COLORS["drone"])

        raw_detections.append(([x1,y1,x2,y2], final_label))

    # ── Update tracker (temporal smoothing) ─────────────────────
    active = tracker.update(raw_detections)

    # Build final draw list from stable trackers
    draw_list = []
    for t in active:
        if t.visible:
            draw_list.append((
                t.ibox,
                t.smoother.stable,
                0.0,   # conf shown on label comes from raw detection
                COLORS.get(
                    t.smoother.stable.replace("drone (","").replace(")",""),
                    COLORS.get("drone", DEFAULT_COLOR)
                )
            ))
    return draw_list

def _draw_frame(frame, draw_list, frame_idx):
    import cv2
    hud_counts = Counter()
    for (box, label, conf, color) in draw_list:
        x1,y1,x2,y2 = box
        draw_box(frame, x1,y1,x2,y2, label, conf, color)
        hud_counts[label] += 1

    # HUD
    hud_y = 28
    for lbl, cnt in hud_counts.items():
        base = lbl.replace("drone (","").replace(")","")
        color = COLORS.get(base, DEFAULT_COLOR)
        cv2.putText(frame, f"{lbl}: {cnt}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        hud_y += 30

    cv2.putText(frame, f"Frame {frame_idx}  |  window={SMOOTH_FRAMES}fr  |  area_thresh={BBOX_AREA_THRESHOLD}px",
                (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

def _run_step5_images(image_files, yolo, classifier, infer_tf, size_classes):
    """Process a folder of images (no video loop needed)."""
    import cv2
    tracker = TrackerPool()
    for idx, fpath in enumerate(image_files):
        frame = cv2.imread(fpath)
        if frame is None: continue
        draw_list = _process_frame(frame, yolo, classifier, infer_tf,
                                   size_classes, tracker, idx)
        _draw_frame(frame, draw_list, idx)
        if SAVE_INFER_OUTPUT:
            out = os.path.join(INFER_OUTPUT_DIR, Path(fpath).name)
            cv2.imwrite(out, frame)
        if SHOW_WINDOW:
            cv2.imshow("2-Stage Drone Pipeline", frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'): break
        if (idx+1) % 20 == 0:
            print(f"  {idx+1}/{len(image_files)} images done")
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print(f"  ✓ Done. {len(image_files)} images processed.")
    if SAVE_INFER_OUTPUT:
        print(f"  Output saved to: {INFER_OUTPUT_DIR}")


# ───────────────────────────────────────────────────────────
# RUN ALL STEPS
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Validate that user has filled in paths
    placeholder_paths = [p for p in [RAW_DIR, CROPS_DIR, SPLIT_DIR, OUTPUT_DIR, WEIGHTS_PATH]
                         if "path\\to\\your" in p or "path/to/your" in p]
    if placeholder_paths:
        print("\n" + "!"*55)
        print("  ERROR: Please fill in your actual paths at the top of this file.")
        print("  These still have placeholder values:")
        for p in placeholder_paths:
            print(f"    {p}")
        print("!"*55 + "\n")
        sys.exit(1)

    os.makedirs(CROPS_DIR,  exist_ok=True)
    os.makedirs(SPLIT_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start = time.time()

    if RUN_STEP1_CROP:  run_step1()
    if RUN_STEP2_SPLIT: run_step2()
    if RUN_STEP3_TRAIN: run_step3()
    if RUN_STEP4_EVAL:  run_step4()
    if RUN_STEP5_INFER: run_step5()

    total = time.time() - start
    print(f"\n{'═'*55}")
    print(f"  ALL STEPS COMPLETE")
    print(f"  Total time: {total/3600:.1f} hours ({total/60:.0f} minutes)")
    print(f"  Results in: {OUTPUT_DIR}")
    print(f"{'═'*55}\n")
