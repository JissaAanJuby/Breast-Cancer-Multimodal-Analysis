"""
STAGE 4 — SIFT feature matching + RANSAC outlier rejection.

BUSI images alone aren't natural stereo/multi-view pairs, so (as instructed)
we create a CONTROLLED transformed version of a real image with a KNOWN
homography (rotation + scale + translation) — this lets us both demonstrate
the technique correctly AND verify RANSAC recovers a transform close to the
one we actually applied.
"""
import cv2
import numpy as np


def make_transformed_pair(gray, angle_deg=15, scale=1.1, tx=12, ty=8):
    """Apply a KNOWN affine transform to create img2 from img1, so we have a
    ground-truth transform to sanity-check RANSAC's estimate against."""
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    transformed = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return transformed, M


def sift_keypoints(gray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    vis = cv2.drawKeypoints(
        gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return kp, des, vis


def match_lowe_ratio(des1, des2, ratio=0.75):
    """BFMatcher + Lowe's ratio test to keep only distinctive matches before
    RANSAC ever sees them (RANSAC works far better on a pre-filtered set)."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return good


def ransac_homography(kp1, kp2, good_matches, reproj_thresh=4.0):
    """Estimate a homography with RANSAC and split matches into inliers vs
    outliers. RANSAC is necessary here because Lowe's ratio test still leaves
    some incorrect matches (repetitive speckle texture in ultrasound is
    especially prone to this) — RANSAC fits a global geometric model and
    rejects matches inconsistent with it, rather than trusting each match's
    local descriptor distance alone."""
    if len(good_matches) < 4:
        return None, None, [], []

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if mask is None:
        return H, mask, [], good_matches

    mask = mask.ravel().astype(bool)
    inliers = [m for m, keep in zip(good_matches, mask) if keep]
    outliers = [m for m, keep in zip(good_matches, mask) if not keep]
    return H, mask, inliers, outliers


def draw_matches(img1, kp1, img2, kp2, matches, max_draw=40):
    return cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
