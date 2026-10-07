"""
STAGE 3 — Biomedical Image Processing
Enhancement (CLAHE) | Restoration (Wiener deconvolution) | Segmentation (Otsu +
morphology) | Compression (JPEG quality sweep with PSNR/SSIM).
"""
import os
import cv2
import numpy as np
import pandas as pd
from skimage.restoration import wiener
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# A. ENHANCEMENT — CLAHE
# ---------------------------------------------------------------------------
def enhance_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Contrast Limited Adaptive Histogram Equalization.

    Why CLAHE (not plain histogram equalization) for ultrasound:
    ultrasound images have strong LOCAL contrast variation (speckle noise,
    shadowing) and a global histogram equalization would over-amplify noisy
    regions and wash out real detail. CLAHE equalizes contrast in small tiles
    with a clipped histogram, boosting local contrast (e.g. lesion vs.
    surrounding tissue) without blowing out noise globally.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# B. RESTORATION — degrade with a known model, then invert it (Wiener)
# ---------------------------------------------------------------------------
def degrade_image(gray, blur_ksize=7, noise_sigma=8):
    """Simulate a KNOWN degradation model: out-of-focus/motion blur (Gaussian
    PSF) + additive Gaussian noise. Restoration only makes sense once you can
    state the degradation model being reversed — this is that model."""
    psf = cv2.getGaussianKernel(blur_ksize, blur_ksize / 3)
    psf_2d = psf @ psf.T
    blurred = cv2.filter2D(gray.astype(np.float64), -1, psf_2d)
    noisy = blurred + np.random.default_rng(0).normal(0, noise_sigma, gray.shape)
    degraded = np.clip(noisy, 0, 255)
    return degraded.astype(np.uint8), psf_2d


def restore_wiener(degraded, psf_2d, balance=0.02):
    """Wiener deconvolution: inverts the known blur PSF while damping noise
    amplification, using the balance term as a proxy for the noise-to-signal
    ratio (classic Wiener filter formulation).

    Limitation (be upfront about this in your writeup / viva): this assumes
    the PSF is exactly known and spatially invariant, which is an
    idealization — real ultrasound speckle is not a simple linear blur, so
    this demonstrates the *technique* correctly rather than clinically
    perfect restoration.
    """
    img_float = degraded.astype(np.float64) / 255.0
    restored = wiener(img_float, psf_2d, balance=balance, clip=False)
    restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
    return restored


# ---------------------------------------------------------------------------
# C. SEGMENTATION — Otsu + morphology + contour extraction
# ---------------------------------------------------------------------------
def segment_lesion(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, raw_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ultrasound lesions are often darker than surrounding tissue; Otsu may
    # pick either polarity depending on image statistics, so keep whichever
    # binarization yields the smaller connected foreground blob (typical for
    # a discrete lesion rather than the whole background).
    fg_frac = raw_mask.mean() / 255.0
    if fg_frac > 0.5:
        raw_mask = 255 - raw_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(overlay, [largest], -1, (0, 0, 255), 2)

    return raw_mask, cleaned, overlay


def segmentation_metrics(pred_mask, gt_mask):
    """Dice and IoU against ground-truth mask (only computable when BUSI's
    *_mask.png ground truth is available for that image)."""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    return {"dice": float(dice), "iou": float(iou)}


# ---------------------------------------------------------------------------
# D. COMPRESSION — real JPEG quality sweep with PSNR/SSIM/ratio
# ---------------------------------------------------------------------------
def compression_experiment(gray, qualities=(95, 75, 50, 25), out_dir="outputs/compressed"):
    os.makedirs(out_dir, exist_ok=True)
    original_size = gray.nbytes
    rows = []
    decoded_images = {}

    for q in qualities:
        ok, enc = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            continue
        compressed_size = enc.size  # bytes actually on disk
        out_path = os.path.join(out_dir, f"compressed_q{q}.jpg")
        with open(out_path, "wb") as f:
            f.write(enc.tobytes())

        decoded = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        decoded_images[q] = decoded

        p = psnr(gray, decoded, data_range=255)
        s = ssim(gray, decoded, data_range=255)
        ratio = original_size / compressed_size

        rows.append({
            "quality": q,
            "file_size_bytes": int(compressed_size),
            "compression_ratio": round(ratio, 3),
            "psnr_db": round(float(p), 3),
            "ssim": round(float(s), 4),
        })

    df = pd.DataFrame(rows).sort_values("quality", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "compression_table.csv"), index=False)
    return df, decoded_images
