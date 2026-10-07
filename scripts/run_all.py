"""
Runs STAGES 3-9 end to end and writes:
  outputs/metrics/*.csv / *.json
  outputs/figures/*.png
  models/*.joblib, models/*.pt

Run from the project root:
    python scripts/run_all.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import cv2

from src import image_processing as ip
from src import sift_ransac as sr
from src import cnn_model as cm
from src import tabular_model as tm
from src import fusion as fu
from src import visualization as viz

METRICS_DIR = "outputs/metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

t0 = time.time()
final_results = {}

# ---------------------------------------------------------------------------
print("\n=== STAGE 3: Biomedical image processing ===")
sample_path = sorted(cm.discover_busi(), key=lambda it: it["label"])[0]["path"]
# use a malignant sample for a more visually interesting demo if available
malignant_items = [it for it in cm.discover_busi() if it["label"] == "malignant"]
sample_item = malignant_items[0] if malignant_items else cm.discover_busi()[0]
gray = cv2.imread(sample_item["path"], cv2.IMREAD_GRAYSCALE)

enhanced = ip.enhance_clahe(gray)
viz.fig_before_after(gray, enhanced, ("Original", "CLAHE enhanced"), "01_enhancement.png")
viz.fig_histogram_comparison(gray, enhanced)

degraded, psf = ip.degrade_image(gray)
restored = ip.restore_wiener(degraded, psf)
viz.fig_before_after(degraded, restored, ("Degraded (blur+noise)", "Wiener restored"), "03_restoration.png")

raw_mask, cleaned_mask, overlay = ip.segment_lesion(gray)
viz.fig_segmentation(gray, raw_mask, cleaned_mask, overlay)
seg_metrics = None
if sample_item["mask"]:
    gt_mask = cv2.imread(sample_item["mask"], cv2.IMREAD_GRAYSCALE)
    seg_metrics = ip.segmentation_metrics(cleaned_mask, gt_mask)
    print("Segmentation vs ground-truth mask:", seg_metrics)
final_results["segmentation_metrics"] = seg_metrics

comp_df, _ = ip.compression_experiment(gray)
viz.fig_compression_table(comp_df)
comp_df.to_csv(os.path.join(METRICS_DIR, "compression_table.csv"), index=False)
print(comp_df)

# ---------------------------------------------------------------------------
print("\n=== STAGE 4: SIFT + RANSAC ===")
img2, M_true = sr.make_transformed_pair(gray)
kp1, des1, vis1 = sr.sift_keypoints(gray)
kp2, des2, vis2 = sr.sift_keypoints(img2)
viz.fig_sift_keypoints(vis1)

good = sr.match_lowe_ratio(des1, des2)
H, mask, inliers, outliers = sr.ransac_homography(kp1, kp2, good)
raw_vis = sr.draw_matches(gray, kp1, img2, kp2, good)
ransac_vis = sr.draw_matches(gray, kp1, img2, kp2, inliers)
viz.fig_matches(raw_vis, ransac_vis)

sift_ransac_summary = {
    "keypoints_img1": len(kp1), "keypoints_img2": len(kp2),
    "good_matches_after_ratio_test": len(good),
    "ransac_inliers": len(inliers), "ransac_outliers": len(outliers),
    "inlier_ratio": round(len(inliers) / max(len(good), 1), 3),
}
print(sift_ransac_summary)
final_results["sift_ransac"] = sift_ransac_summary

# ---------------------------------------------------------------------------
print("\n=== STAGE 5: CNN transfer learning (frozen ResNet18 backbone) ===")
items = cm.discover_busi()
viz.fig_class_distribution(items, cm.CLASSES)
train_items, val_items, test_items = cm.stratified_split(items)
print(f"train={len(train_items)} val={len(val_items)} test={len(test_items)}")

model = cm.build_model()
model, history = cm.train_model(model, train_items, val_items, epochs=6, batch_size=8)
viz.fig_training_curves(history)

y_true, y_pred, y_prob = cm.evaluate_model(model, test_items)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
cnn_metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
}
cnn_cm = confusion_matrix(y_true, y_pred)
viz.fig_confusion_matrix(cnn_cm, cm.CLASSES, "CNN confusion matrix", "09_cnn_confusion_matrix.png")
print("CNN test metrics:", cnn_metrics)
print(classification_report(y_true, y_pred, target_names=cm.CLASSES, zero_division=0))
final_results["cnn_metrics"] = cnn_metrics

# ---------------------------------------------------------------------------
print("\n=== STAGE 6: Tabular ML (Wisconsin, leakage-free) ===")
df = tm.load_and_clean()
X_train, X_test, y_train, y_test, scaler = tm.split_and_scale(df)
xgb_model = tm.train_xgb(X_train, y_train)
tab_metrics, tab_cm, (fpr, tpr), tab_pred, tab_prob = tm.evaluate(xgb_model, X_test, y_test)
tm.save_model(xgb_model, scaler)
viz.fig_confusion_matrix(tab_cm, ["Benign", "Malignant"], "Tabular confusion matrix", "10_tabular_confusion_matrix.png")
viz.fig_roc_curve(fpr, tpr, tab_metrics["roc_auc"])
viz.fig_pr_curve(y_test, tab_prob)
print("Tabular test metrics:", tab_metrics)
final_results["tabular_metrics"] = tab_metrics

# ---------------------------------------------------------------------------
print("\n=== STAGE 7: Fusion (honest, unpaired-dataset demo) ===")
demo, fusion_results = fu.simulate_paired_demo(
    y_test.reset_index(drop=True), tab_prob, y_true, y_prob, np.array(cm.CLASSES)
)
viz.fig_fusion_comparison(fusion_results)
print(pd.DataFrame(fusion_results).T)
final_results["fusion_demo_results"] = fusion_results

# ---------------------------------------------------------------------------
print("\n=== Saving final results table ===")
main_table = pd.DataFrame([
    {"component": "Tabular", "model": "XGBoost", **tab_metrics},
    {"component": "Imaging", "model": "ResNet18 (frozen backbone)",
     "accuracy": cnn_metrics["accuracy"], "precision": cnn_metrics["precision_macro"],
     "recall": cnn_metrics["recall_macro"], "f1": cnn_metrics["f1_macro"], "roc_auc": np.nan},
    {"component": "Fusion (illustrative demo)", "model": "Weighted late-fusion",
     **fusion_results["weighted_fusion (on demo pairs)"]},
    {"component": "Fusion (illustrative demo)", "model": "Stacked meta-classifier",
     **fusion_results["stacked_meta (held-out demo split)"]},
])
main_table.to_csv(os.path.join(METRICS_DIR, "final_results_table.csv"), index=False)
print(main_table)

with open(os.path.join(METRICS_DIR, "all_metrics.json"), "w") as f:
    json.dump(final_results, f, indent=2, default=float)

print(f"\nDone in {time.time()-t0:.1f}s. Figures -> outputs/figures/, metrics -> outputs/metrics/")
