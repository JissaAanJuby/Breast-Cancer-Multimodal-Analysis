"""
STAGE 7 — Fusion.

IMPORTANT DATASET LIMITATION (state this explicitly in your report/viva):
Wisconsin (FNA cytology, numeric nuclear features) and BUSI (ultrasound
images) are two INDEPENDENT public datasets with no shared patients. There is
no ground truth linking a specific Wisconsin row to a specific BUSI image.

So this module does NOT claim per-patient fused predictions on real data.
Instead it:
  1. Evaluates each branch independently on its own held-out test set (the
     real, valid numbers — these go in your main results table).
  2. Implements the actual fusion MECHANISM (weighted late-fusion and a
     stacked meta-classifier) exactly as it would be applied if a hospital
     supplied both modalities for the same patient.
  3. Demonstrates that mechanism on an ILLUSTRATIVE simulated pairing (each
     test-set tabular case is randomly paired with a test-set image case of
     the matching label) purely to show the fusion code works end-to-end —
     clearly labelled as a methodology demo, not a clinical result.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def weighted_fusion(p_tab, p_img_malignant, w_tab=0.5, w_img=0.5):
    """P_final = w1 * P_tabular + w2 * P_imaging (both expressed as
    P(malignant)). This is the fusion rule you'd apply per patient in a
    real paired-data deployment."""
    p_final = w_tab * p_tab + w_img * p_img_malignant
    label = "Malignant" if p_final >= 0.5 else "Benign"
    return p_final, label


def simulate_paired_demo(tab_y_test, tab_prob, img_y_true, img_y_prob, img_classes,
                          n_pairs=150, w_tab=0.5, w_img=0.5, seed=42):
    """Illustrative-only: randomly pair each simulated case's tabular test
    prediction with a BUSI test prediction OF THE SAME GROUND-TRUTH CLASS
    (benign/malignant only — BUSI's 'normal' class has no Wisconsin
    equivalent and is excluded from this demo), then apply weighted fusion
    and a stacked logistic-regression meta-classifier, and compare all three
    (tabular-only, imaging-only, fused) on the SAME simulated pairs.
    """
    rng = np.random.default_rng(seed)

    mal_idx = np.where(img_classes[img_y_true] == "malignant")[0]
    ben_idx = np.where(img_classes[img_y_true] == "benign")[0]
    tab_mal_idx = np.where(tab_y_test.values == 1)[0]
    tab_ben_idx = np.where(tab_y_test.values == 0)[0]

    rows = []
    for _ in range(n_pairs // 2):
        i = rng.choice(tab_mal_idx)
        j = rng.choice(mal_idx)
        rows.append((tab_prob[i], img_y_prob[j][list(img_classes).index("malignant")], 1))
    for _ in range(n_pairs // 2):
        i = rng.choice(tab_ben_idx)
        j = rng.choice(ben_idx)
        rows.append((tab_prob[i], img_y_prob[j][list(img_classes).index("malignant")], 0))

    demo = pd.DataFrame(rows, columns=["p_tab", "p_img", "y_true"]).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)

    demo["p_fused_weighted"] = w_tab * demo["p_tab"] + w_img * demo["p_img"]
    demo["pred_weighted"] = (demo["p_fused_weighted"] >= 0.5).astype(int)

    # Stacked meta-classifier: logistic regression on [p_tab, p_img] -> label.
    # Split the DEMO pairs themselves into train/test so the meta-classifier
    # is also evaluated out-of-sample (not fit and scored on the same rows).
    split = int(len(demo) * 0.7)
    train, test = demo.iloc[:split], demo.iloc[split:]
    meta = LogisticRegression()
    meta.fit(train[["p_tab", "p_img"]], train["y_true"])
    test = test.copy()
    test["pred_meta"] = meta.predict(test[["p_tab", "p_img"]])
    test["p_meta"] = meta.predict_proba(test[["p_tab", "p_img"]])[:, 1]

    def score(y_true, y_pred, y_prob):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        }

    results = {
        "tabular_only (on demo pairs)": score(demo["y_true"], (demo["p_tab"] >= 0.5).astype(int), demo["p_tab"]),
        "imaging_only (on demo pairs)": score(demo["y_true"], (demo["p_img"] >= 0.5).astype(int), demo["p_img"]),
        "weighted_fusion (on demo pairs)": score(demo["y_true"], demo["pred_weighted"], demo["p_fused_weighted"]),
        "stacked_meta (held-out demo split)": score(test["y_true"], test["pred_meta"], test["p_meta"]),
    }
    return demo, results
