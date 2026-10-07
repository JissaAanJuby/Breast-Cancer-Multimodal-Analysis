"""
STAGE 8 — T6 Visualization. Every figure is SAVED to outputs/figures/, not
just displayed transiently.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_before_after(original, processed, titles=("Original", "Processed"), name="fig.png"):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(original, cmap="gray"); ax[0].set_title(titles[0]); ax[0].axis("off")
    ax[1].imshow(processed, cmap="gray"); ax[1].set_title(titles[1]); ax[1].axis("off")
    return _save(fig, name)


def fig_histogram_comparison(original, enhanced, name="02_enhancement_histogram.png"):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    ax[0].hist(original.ravel(), bins=50, color="gray"); ax[0].set_title("Original histogram")
    ax[1].hist(enhanced.ravel(), bins=50, color="steelblue"); ax[1].set_title("CLAHE histogram")
    return _save(fig, name)


def fig_segmentation(original, mask, cleaned, overlay, name="04_segmentation.png"):
    fig, ax = plt.subplots(1, 4, figsize=(14, 4))
    for a, img, t in zip(
        ax, [original, mask, cleaned, overlay],
        ["Original", "Otsu mask", "Cleaned mask", "Overlay"],
    ):
        a.imshow(img if img.ndim == 2 else img[..., ::-1], cmap="gray" if img.ndim == 2 else None)
        a.set_title(t); a.axis("off")
    return _save(fig, name)


def fig_compression_table(df, name="05_compression_quality_vs_psnr.png"):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df["quality"], df["psnr_db"], "o-", color="tab:blue", label="PSNR (dB)")
    ax1.set_xlabel("JPEG Quality"); ax1.set_ylabel("PSNR (dB)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df["quality"], df["compression_ratio"], "s--", color="tab:red", label="Compression ratio")
    ax2.set_ylabel("Compression ratio", color="tab:red")
    fig.suptitle("Compression quality vs PSNR / ratio")
    return _save(fig, name)


def fig_sift_keypoints(vis, name="06_sift_keypoints.png"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(vis, cmap="gray"); ax.axis("off"); ax.set_title("SIFT keypoints")
    return _save(fig, name)


def fig_matches(raw_vis, ransac_vis, name="07_matches_raw_vs_ransac.png"):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].imshow(raw_vis, cmap="gray"); ax[0].set_title("Raw good matches (post ratio-test)"); ax[0].axis("off")
    ax[1].imshow(ransac_vis, cmap="gray"); ax[1].set_title("RANSAC inlier matches"); ax[1].axis("off")
    return _save(fig, name)


def fig_training_curves(history, name="08_cnn_training_curves.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train"); ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    return _save(fig, name)


def fig_confusion_matrix(cm, class_names, title, name):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    return _save(fig, name)


def fig_roc_curve(fpr, tpr, auc_val, name="11_tabular_roc_curve.png"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Tabular model ROC curve"); ax.legend()
    return _save(fig, name)


def fig_pr_curve(y_true, y_prob, name="12_tabular_pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curve")
    return _save(fig, name)


def fig_class_distribution(items, class_names, name="13_busi_class_distribution.png"):
    counts = [sum(1 for it in items if it["label"] == c) for c in class_names]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=class_names, y=counts, ax=ax)
    ax.set_title("BUSI class distribution"); ax.set_ylabel("Count")
    return _save(fig, name)


def fig_fusion_comparison(results_dict, name="14_fusion_comparison.png"):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    width = 0.2
    for i, (name_k, vals) in enumerate(results_dict.items()):
        ax.bar(x + i * width, [vals[m] for m in metrics], width, label=name_k)
    ax.set_xticks(x + width * 1.5); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title("Branch comparison (illustrative fusion demo)")
    return _save(fig, name)
