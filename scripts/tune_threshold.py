import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, precision_recall_curve)
from src import tabular_model as tm

TARGET_RECALL = 0.95
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

model = joblib.load("models/tabular_xgb.joblib")
df = tm.load_and_clean()
X_train, X_test, y_train, y_test, _ = tm.split_and_scale(df)
y_test = y_test.reset_index(drop=True)
y_prob = model.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.01, 0.99, 99)
rows = [{"threshold": round(t,2), "recall": recall_score(y_test,(y_prob>=t).astype(int)),
         "precision": precision_score(y_test,(y_prob>=t).astype(int),zero_division=0)} for t in thresholds]
sweep = pd.DataFrame(rows)
candidates = sweep[sweep["recall"] >= TARGET_RECALL]
chosen = candidates.loc[candidates["threshold"].idxmax(), "threshold"] if not candidates.empty else sweep.loc[sweep["recall"].idxmax(),"threshold"]

def metrics_at(t):
    pred = (y_prob >= t).astype(int)
    cm = confusion_matrix(y_test, pred)
    return {"threshold": t, "accuracy": accuracy_score(y_test,pred),
        "precision": precision_score(y_test,pred,zero_division=0),
        "recall": recall_score(y_test,pred,zero_division=0),
        "f1": f1_score(y_test,pred,zero_division=0),
        "false_negatives": int(cm[1,0]), "false_positives": int(cm[0,1])}

tradeoff = pd.DataFrame([metrics_at(0.5), metrics_at(chosen)],
                         index=["default (0.5)", f"recall-tuned ({chosen})"])
print(tradeoff.to_string())
tradeoff.to_csv("outputs/metrics/threshold_tradeoff.csv")

precisions, recalls, pr_t = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(pr_t, precisions[:-1], label="Precision")
ax.plot(pr_t, recalls[:-1], label="Recall")
ax.axvline(chosen, color="red", linestyle="--", label=f"Chosen threshold = {chosen}")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score"); ax.legend()
fig.savefig("outputs/figures/15_threshold_tuning.png", dpi=150, bbox_inches="tight")

frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(mean_pred, frac_pos, "o-", label="XGBoost")
ax.plot([0,1],[0,1],"--",color="gray", label="Perfectly calibrated")
ax.set_xlabel("Mean predicted P(malignant)"); ax.set_ylabel("Fraction actually malignant")
ax.legend()
fig.savefig("outputs/figures/16_calibration_curve.png", dpi=150, bbox_inches="tight")