import argparse, json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surrogate_preds", required=True, help="CSV with y_pred_surrogate and id")
    ap.add_argument("--fused_preds", required=True, help="CSV with id and y_pred_fused (and maybe y_prob_fused)")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--is_classification", action="store_true", help="Treat target as classification (use probs if present)")
    args = ap.parse_args()

    s = pd.read_csv(args.surrogate_preds)           # columns: id, y_pred_surrogate
    f = pd.read_csv(args.fused_preds)               # id, y_pred_fused[, y_prob_fused]

    if "id" not in s.columns or "id" not in f.columns:
        raise ValueError("Both CSVs must include 'id' column")

    m = s.merge(f, on="id", how="inner")

    out = {}
    if args.is_classification:
        # Prob-based fidelity if available; else fall back to labels (0/1)
        if "y_prob_fused" in m.columns:
            y_true_prob = m["y_prob_fused"].astype(float).to_numpy()
        else:
            y_true_prob = m["y_pred_fused"].astype(float).to_numpy()

        y_pred_prob = m["y_pred_surrogate"].astype(float).to_numpy()

        out["fidelity_prob_R2"] = float(r2_score(y_true_prob, y_pred_prob))
        out["fidelity_prob_MSE"] = float(mean_squared_error(y_true_prob, y_pred_prob))

        # Class agreement (threshold @0.5)
        y_true_cls = (y_true_prob >= 0.5).astype(int)
        y_pred_cls = (y_pred_prob >= 0.5).astype(int)
        out["fidelity_cls_Accuracy"] = float(accuracy_score(y_true_cls, y_pred_cls))
        out["fidelity_cls_F1"] = float(f1_score(y_true_cls, y_pred_cls, average="binary"))
    else:
        # Regression
        y_true = m["y_pred_fused"].astype(float).to_numpy()
        y_pred = m["y_pred_surrogate"].astype(float).to_numpy()
        out["fidelity_R2"] = float(r2_s]()
