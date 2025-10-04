import argparse
import os
import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
from pathlib import Path
import time
import pickle

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    args = parser.parse_args()
    idx = args.id

    base_dir = f"/workspace/comparison_xai/run_{idx}"
    ensure_dir(base_dir)

    features = [
        'x_age', 'x_case', 'x_type', 'x_lymphnode_met', 'rad_timing', 'rad_recall',
        'libra_breastarea', 'libra_densearea', 'libra_percentdensity',
        'rad_recall_type_right_1', 'rad_recall_type_right_2',
        'rad_recall_type_left_1', 'rad_recall_type_left_2'
    ]
    tabular_csv = "/workspace/gdrive/Data_tabular/dataset_tabular.csv"
    train_df = pd.read_csv(tabular_csv)
    X_tab = train_df[features].reset_index(drop=True)
    x0 = X_tab.iloc[idx].values
    x0_2d = x0.reshape(1, -1)
    y_true = train_df["rad_decision"].values[idx]

    # Black-box: multimodal
    mm_preds_path = "/workspace/multimodal_results/fused_predictions.csv"
    mm_preds = pd.read_csv(mm_preds_path)
    bb_label = int(mm_preds["y_pred_fused"].iloc[idx])
    bb_prob = float(mm_preds["y_prob_fused"].iloc[idx])

    # -------- LIME --------
    lime_dir = os.path.join(base_dir, "lime"); ensure_dir(lime_dir)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_tab.values,
        mode='classification',
        feature_names=features,
        discretize_continuous=True
    )
    start = time.perf_counter()
    lime_exp = lime_explainer.explain_instance(x0, lambda x: np.array([[1-bb_prob, bb_prob]]))
    lime_time = time.perf_counter() - start
    # Save explainer object
    with open(os.path.join(lime_dir, "lime_exp.pkl"), "wb") as f:
        pickle.dump(lime_exp, f)
    lime_pred_prob = lime_exp.predict_proba[1]
    lime_pred = int(lime_pred_prob > 0.5)
    lime_feat_count = len([v for v in lime_exp.as_map()[1] if abs(v[1]) > 1e-6])
    # Save fig
    fig = lime_exp.as_pyplot_figure()
    fig.savefig(os.path.join(lime_dir, "lime_fig.png")); plt.close(fig)
    pd.DataFrame([{
        "id": idx,
        "LIME_Pred": lime_pred,
        "LIME_Prob": lime_pred_prob,
        "LIME_NumFeatures": lime_feat_count,
        "LIME_RuntimeSec": lime_time,
        "LIME_Fidelity": int(lime_pred == bb_label),
        "LIME_Prob_MSE": mean_squared_error([bb_prob], [lime_pred_prob])
    }]).to_csv(os.path.join(lime_dir, "lime_metrics.csv"), index=False)

    # -------- SHAP --------
    shap_dir = os.path.join(base_dir, "shap"); ensure_dir(shap_dir)
    shap_explainer = shap.KernelExplainer(lambda x: np.array([[1-bb_prob, bb_prob]]), X_tab.values[:100])
    start = time.perf_counter()
    shap_vals = shap_explainer.shap_values(x0_2d)
    shap_time = time.perf_counter() - start
    # Save explainer object
    with open(os.path.join(shap_dir, "shap_exp.pkl"), "wb") as f:
        pickle.dump(shap_explainer, f)
    shap_pred_prob = bb_prob
    shap_pred = int(shap_pred_prob > 0.5)
    shap_feat_count = int(np.sum(np.abs(shap_vals[1][0]) > 1e-3))
    # SHAP fig
    shap.summary_plot([shap_vals[1]], [x0_2d], feature_names=features, show=False)
    plt.savefig(os.path.join(shap_dir, "shap_fig.png")); plt.close()
    pd.DataFrame([{
        "id": idx,
        "SHAP_Pred": shap_pred,
        "SHAP_Prob": shap_pred_prob,
        "SHAP_NumFeatures": shap_feat_count,
        "SHAP_RuntimeSec": shap_time,
        "SHAP_Fidelity": int(shap_pred == bb_label),
        "SHAP_Prob_MSE": mean_squared_error([bb_prob], [shap_pred_prob])
    }]).to_csv(os.path.join(shap_dir, "shap_metrics.csv"), index=False)

    # -------- Decision Tree (DT) --------
    dt_dir = os.path.join(base_dir, "dt"); ensure_dir(dt_dir)
    dt_surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_surrogate.fit(X_tab, mm_preds["y_pred_fused"].values)
    joblib.dump(dt_surrogate, os.path.join(dt_dir, "dt_model.pkl"))
    start = time.perf_counter()
    dt_pred_prob = dt_surrogate.predict_proba(x0_2d)[0, 1]
    dt_pred = int(dt_pred_prob > 0.5)
    dt_time = time.perf_counter() - start
    tree = dt_surrogate.tree_
    rule_depths = []
    for i in range(tree.node_count):
        if tree.children_left[i] == tree.children_right[i]:  # leaf
            depth = 0
            j = i
            while j != 0:
                parent = np.where((tree.children_left == j) | (tree.children_right == j))[0][0]
                depth += 1
                j = parent
            rule_depths.append(depth)
    dt_rule_len = np.mean(rule_depths) if rule_depths else np.nan
    pd.DataFrame([{
        "id": idx,
        "DT_Pred": dt_pred,
        "DT_Prob": dt_pred_prob,
        "DT_MeanRuleLen": dt_rule_len,
        "DT_RuntimeSec": dt_time,
        "DT_Fidelity": int(dt_pred == bb_label),
        "DT_Prob_MSE": mean_squared_error([bb_prob], [dt_pred_prob])
    }]).to_csv(os.path.join(dt_dir, "dt_metrics.csv"), index=False)

    # Optional: DT rule visualization
    try:
        import graphviz
        from sklearn.tree import export_graphviz
        export_graphviz(dt_surrogate, out_file=os.path.join(dt_dir, "dt.dot"),
                        feature_names=features, class_names=['0','1'], filled=True)
        # To create a PNG, you need Graphviz installed in the container
        # os.system(f"dot -Tpng {os.path.join(dt_dir, 'dt.dot')} -o {os.path.join(dt_dir, 'dt_rule_visual.png')}")
    except Exception as e:
        print("DT rule visualization skipped:", e)

    # -------- Fuzzy surrogate --------
    fuzzy_dir = os.path.join(base_dir, "fuzzy"); ensure_dir(fuzzy_dir)
    fuzzy_root = "/workspace/xai_fuzzy_model"
    run_dirs = sorted([d for d in os.listdir(fuzzy_root) if d.startswith(f"run_{idx}_")], reverse=True)
    if not run_dirs:
        raise RuntimeError(f"No fuzzy surrogate run found for id {idx}!")
    run_dir = os.path.join(fuzzy_root, run_dirs[0])
    # Copy all outputs to fuzzy_dir
    for fn in ["frbs_model.RData", "frbs_predictions.csv", "frbs_metrics.csv", "frbs_rules.csv", "frbs_interpret.csv"]:
        src = os.path.join(run_dir, fn)
        if os.path.exists(src):
            os.system(f"cp {src} {fuzzy_dir}/")
    frbs_preds = pd.read_csv(os.path.join(fuzzy_dir, "frbs_predictions.csv"))["Pred"].values
    frbs_pred = int(frbs_preds[0])
    frbs_interpret = pd.read_csv(os.path.join(fuzzy_dir, "frbs_interpret.csv"))
    frbs_numrules = int(frbs_interpret["NumRules"][0])
    frbs_meanrulelen = float(frbs_interpret["MeanRuleLength"][0])
    frbs_metrics = pd.read_csv(os.path.join(fuzzy_dir, "frbs_metrics.csv"))
    frbs_acc = float(frbs_metrics["Accuracy"].mean())
    frbs_f1 = float(frbs_metrics["F1"].mean())
    # Fuzzy runtime (if you save it as a txt file)
    frbs_time_path = os.path.join(run_dir, "frbs_time.txt")
    frbs_time = float(open(frbs_time_path).read()) if os.path.exists(frbs_time_path) else np.nan
    pd.DataFrame([{
        "id": idx,
        "FRBS_Pred": frbs_pred,
        "FRBS_NumRules": frbs_numrules,
        "FRBS_MeanRuleLength": frbs_meanrulelen,
        "FRBS_Accuracy": frbs_acc,
        "FRBS_F1": frbs_f1,
        "FRBS_RuntimeSec": frbs_time,
        "FRBS_Fidelity": int(frbs_pred == bb_label)
    }]).to_csv(os.path.join(fuzzy_dir, "frbs_summary.csv"), index=False)

    # -------- Comparative Table --------
    comparison = pd.DataFrame([{
        "Index": idx,
        "True_Label": y_true,
        "BlackBox_Label": bb_label,
        "BlackBox_Prob": bb_prob,
        "LIME_Pred": lime_pred,
        "LIME_Prob": lime_pred_prob,
        "LIME_NumFeatures": lime_feat_count,
        "LIME_RuntimeSec": lime_time,
        "LIME_Fidelity": int(lime_pred == bb_label),
        "LIME_Prob_MSE": mean_squared_error([bb_prob], [lime_pred_prob]),
        "SHAP_Pred": shap_pred,
        "SHAP_Prob": shap_pred_prob,
        "SHAP_NumFeatures": shap_feat_count,
        "SHAP_RuntimeSec": shap_time,
        "SHAP_Fidelity": int(shap_pred == bb_label),
        "SHAP_Prob_MSE": mean_squared_error([bb_prob], [shap_pred_prob]),
        "DT_Pred": dt_pred,
        "DT_Prob": dt_pred_prob,
        "DT_MeanRuleLen": dt_rule_len,
        "DT_RuntimeSec": dt_time,
        "DT_Fidelity": int(dt_pred == bb_label),
        "DT_Prob_MSE": mean_squared_error([bb_prob], [dt_pred_prob]),
        "FRBS_Pred": frbs_pred,
        "FRBS_NumRules": frbs_numrules,
        "FRBS_MeanRuleLength": frbs_meanrulelen,
        "FRBS_Accuracy": frbs_acc,
        "FRBS_F1": frbs_f1,
        "FRBS_RuntimeSec": frbs_time,
        "FRBS_Fidelity": int(frbs_pred == bb_label)
    }])
    out_csv = os.path.join(base_dir, "pointwise_comparison.csv")
    comparison.to_csv(out_csv, index=False)
    print("✅ XAI single-point comparison complete. Results in:", base_dir)
    print(comparison.T)

if __name__ == "__main__":
    main()
