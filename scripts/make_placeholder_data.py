"""
STAGE 2 helper — generates local test data so the pipeline can be built and
verified end-to-end BEFORE you plug in your real files.

1. data/data.csv       -> REAL Wisconsin Diagnostic Breast Cancer data (569 rows,
                          30 features), pulled from sklearn's bundled copy of the
                          same UCI dataset your original data.csv comes from.
                          You can use this file as-is, or overwrite it with your
                          own data.csv (same column names) once you re-download it.

2. data/BUSI/...        -> SYNTHETIC placeholder ultrasound images + masks, used
                          ONLY so every script below has something to run against
                          while you get the real BUSI dataset onto disk. Clearly
                          NOT real medical data. See README "Stage 2" for how to
                          swap in the real BUSI folder.
"""
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.datasets import load_breast_cancer

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Tabular data (real Wisconsin dataset via sklearn)
# ---------------------------------------------------------------------------
def make_tabular_csv(out_path="data/data.csv"):
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # sklearn target: 0 = malignant, 1 = benign -> convert to the familiar M/B coding
    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})
    df = df.drop(columns=["target"])
    # put diagnosis first, like the original Kaggle data.csv layout
    cols = ["diagnosis"] + [c for c in df.columns if c != "diagnosis"]
    df = df[cols]
    df.insert(0, "id", range(1, len(df) + 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[tabular] wrote {out_path} -> shape {df.shape}")


# ---------------------------------------------------------------------------
# 2. Synthetic BUSI-style placeholder images + masks
# ---------------------------------------------------------------------------
def _make_one_image(label, size=224):
    """Grayscale-ish speckle-noise background + a blob shaped by class,
    plus a binary ground-truth mask for the blob (mimics BUSI's *_mask.png)."""
    base = RNG.normal(loc=110, scale=18, size=(size, size)).astype(np.float32)
    base = cv2.GaussianBlur(base, (3, 3), 0)  # ultrasound-like speckle smoothing
    img = np.clip(base, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = np.zeros((size, size), dtype=np.uint8)

    cx, cy = size // 2 + RNG.integers(-15, 15), size // 2 + RNG.integers(-15, 15)
    if label == "benign":
        # rounder, more regular lesion, lower contrast
        r = RNG.integers(28, 38)
        cv2.circle(img, (cx, cy), r, (70, 70, 70), -1)
        cv2.circle(mask, (cx, cy), r, 255, -1)
    elif label == "malignant":
        # irregular, spiculated lesion shape, higher contrast
        axes = (RNG.integers(30, 45), RNG.integers(18, 30))
        angle = int(RNG.integers(0, 180))
        cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, (35, 35, 35), -1)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        # spicules
        for _ in range(6):
            a = RNG.uniform(0, 2 * np.pi)
            x2 = int(cx + np.cos(a) * RNG.integers(40, 55))
            y2 = int(cy + np.sin(a) * RNG.integers(40, 55))
            cv2.line(img, (cx, cy), (x2, y2), (45, 45, 45), 2)
    else:  # normal — no lesion, mask stays empty
        pass

    noise = RNG.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, mask


def make_busi_placeholder(root="data/BUSI", n_per_class=25, size=224):
    for label in ["benign", "malignant", "normal"]:
        d = os.path.join(root, label)
        os.makedirs(d, exist_ok=True)
        for i in range(1, n_per_class + 1):
            img, mask = _make_one_image(label, size=size)
            cv2.imwrite(os.path.join(d, f"{label} ({i}).png"), img)
            if label != "normal":
                cv2.imwrite(os.path.join(d, f"{label} ({i})_mask.png"), mask)
    print(f"[imaging] wrote synthetic BUSI-style placeholder data to {root} "
          f"({n_per_class} images/class)")


if __name__ == "__main__":
    make_tabular_csv()
    make_busi_placeholder()
