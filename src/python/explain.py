import argparse
import os
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Back-compat alias so old pickles load
from src.python.ensemble import MultimodalEnsemble as _MME
MultimodalEnsemble = _MME

from src.python.generate_surrogates import generate_surrogates


def load_x_tab_x_img(idx: int):
    features = [
        'x_age', 'x_case', 'x_type', 'x_lymphnode_met', 'rad_timing', 'rad_recall',
        'libra_breastarea', 'libra_densearea', 'libra_percentdensity',
        'rad_recall_type_right_1', 'rad_recall_type_right_2',
        'rad_recall_type_left_1', 'rad_recall_type_left_2'
    ]
    df = pd.read_csv("/workspace/gdrive/Data_tabular/dataset_tabular.csv")
    x_tab0 = df[features].iloc[[idx]]
    X_img = np.load("/workspace/gdrive/Data_normalized_array/clean_data.npy")
    x_img0 = X_img[idx]
    return x_tab0, x_img0


def compute_mm_on_surrogates(mm, df_sur):
    X = df_sur.drop(columns=["class"]).values.astype(np.float32)
    prob = mm.predict_proba(X)
    p1 = prob[:, 1] if (prob.ndim == 2 and prob.shape[1] >= 2) else prob.ravel()
    lab = (p1 > 0.5).astype(int)
    return p1, lab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Index to explain")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--tab_noise", type=float, default=0.05)
    parser.add_argument("--img_noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=0,
                        help="Use only top-K features (by MI) for fuzzy explainer; 0=all features")
    args = parser.parse_args()

    run_dir = Path(f"/workspace/xai_fuzzy_model/run_{args.id}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load point + MM model
    x_tab0, x_img0 = load_x_tab_x_img(args.id)
    mm_pkl = "/workspace/multimodal_results/multimodal_model.pkl"
    if not os.path.exists(mm_pkl):
        raise FileNotFoundError(f"Multimodal model not found at '{mm_pkl}'. Run fusion first.")
    multimodal_model = joblib.load(mm_pkl)

    # Generate surrogates and save
    df_sur = generate_surrogates(
        x_tab0=x_tab0,
        x_img0=x_img0,
        multimodal_model=multimodal_model,
        n=args.n, tab_noise=args.tab_noise, img_noise=args.img_noise, seed=args.seed
    )
    surrogate_csv = run_dir / "surrogate_dataset.csv"
    df_sur.to_csv(surrogate_csv, index=False)

    # Train FRBS (R)
    import subprocess, json
    subprocess.check_call([
        "python", "-m", "src.python.train_fuzzy_surrogate",
        "--csv", str(surrogate_csv),
        "--run_dir", str(run_dir),
        "--top_k", str(args.top_k)
    ])

    # Pick the dataset used by FRBS
    selected_csv = run_dir / "surrogate_dataset_selected.csv"
    df_used = pd.read_csv(selected_csv) if selected_csv.exists() else pd.read_csv(surrogate_csv)

    # MM on used surrogates
    mm_prob, mm_lab = compute_mm_on_surrogates(multimodal_model, df_used)
    true_y = df_used["class"].astype(int).values

    # Load FRBS predictions (guaranteed by R now). Fallback to final if needed.
    frbs_cv_path = run_dir / "frbs_cv_predictions_aligned.csv"
    if frbs_cv_path.exists():
        frbs_df = pd.read_csv(frbs_cv_path)
        frbs_pred = frbs_df["FRBS_Pred"].values[:len(true_y)].astype(int)
    else:
        final_path = run_dir / "frbs_final_predictions.csv"
        if not final_path.exists():
            raise FileNotFoundError(
                f"Missing both CV and final FRBS prediction files in {run_dir}. "
                f"Expected {frbs_cv_path} or {final_path}."
            )
        frbs_df = pd.read_csv(final_path)
        frbs_pred = frbs_df["FRBS_Final_Pred"].values[:len(true_y)].astype(int)

    # Save joined predictions
    comp = pd.DataFrame({
        "Row": np.arange(1, len(true_y) + 1, dtype=int),
        "True_Class": true_y,
        "FRBS_Pred": frbs_pred,
        "MM_Prob": mm_prob[:len(true_y)],
        "MM_Label": mm_lab[:len(true_y)]
    })
    comp.to_csv(run_dir / "predictions_comparison.csv", index=False)

    # Metrics + fidelity
    frbs_acc = accuracy_score(comp["True_Class"], comp["FRBS_Pred"])
    frbs_prec = precision_score(comp["True_Class"], comp["FRBS_Pred"], zero_division=0)
    frbs_rec = recall_score(comp["True_Class"], comp["FRBS_Pred"], zero_division=0)
    frbs_f1 = f1_score(comp["True_Class"], comp["FRBS_Pred"], zero_division=0)

    mm_acc = accuracy_score(comp["True_Class"], comp["MM_Label"])
    mm_prec = precision_score(comp["True_Class"], comp["MM_Label"], zero_division=0)
    mm_rec = recall_score(comp["True_Class"], comp["MM_Label"], zero_division=0)
    mm_f1 = f1_score(comp["True_Class"], comp["MM_Label"], zero_division=0)

    fidelity = float((comp["FRBS_Pred"].values == comp["MM_Label"].values).mean())

    metrics_df = pd.DataFrame([
        {"Model": "FRBS", "Accuracy": frbs_acc, "Precision": frbs_prec, "Recall": frbs_rec, "F1_Score": frbs_f1},
        {"Model": "Multimodal", "Accuracy": mm_acc, "Precision": mm_prec, "Recall": mm_rec, "F1_Score": mm_f1},
    ])
    metrics_df.to_csv(run_dir / "comparison_metrics.csv", index=False)

    with open(run_dir / "fidelity.txt", "w") as f:
        f.write(str(fidelity))

    print(f"✅ Explained ID {args.id}. All artifacts in: {run_dir} (elapsed {round(time.time()-run_dir.stat().st_mtime,2)}s)")


if __name__ == "__main__":
    main()
