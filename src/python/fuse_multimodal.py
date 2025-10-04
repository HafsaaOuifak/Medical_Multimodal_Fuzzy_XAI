import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.python.ensemble import MultimodalEnsemble

TAB_ROOT = Path("/workspace/tabular_models")
IMG_ROOT = Path("/workspace/image_models")
FUSE_ROOT = Path("/workspace/multimodal_results")
FUSE_ROOT.mkdir(parents=True, exist_ok=True)

# -- Helper to select best model by mean accuracy --
def read_best_model(root, model_names):
    best_score = -1
    best_preds = None
    best_folder = None
    for m in model_names:
        metrics_path = root / m / "metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        mean_acc = metrics["Accuracy"].mean()
        if mean_acc > best_score:
            best_score = mean_acc
            best_preds = pd.read_csv(root / m / "predictions.csv")
            best_folder = m
    return best_preds, best_folder

tabular_models = ["DecisionTree", "RandomForest", "SVM", "MLP"]
image_models = ["ResNet50", "MobileNetV2", "EfficientNetB0"]

tab_preds, tab_name = read_best_model(TAB_ROOT, tabular_models)
img_preds, img_name = read_best_model(IMG_ROOT, image_models)

if tab_preds is None or img_preds is None:
    raise RuntimeError("Best tabular or image model could not be determined.")

print("Best tabular model:", tab_name)
print("Best image model:", img_name)

# -- Merge predictions on Sample_Index (must be aligned!) --
merged = tab_preds.merge(img_preds, on="Sample_Index", suffixes=("_tab", "_img"))
# Use predicted probability if available, else predicted class as float
p_tab = merged["Pred_Proba_Pos_tab"] if "Pred_Proba_Pos_tab" in merged else merged["Predicted_tab"].astype(float)
p_img = merged["Pred_Proba_Pos_img"] if "Pred_Proba_Pos_img" in merged else merged["Predicted_img"].astype(float)
# Fuse by equal average (update to weighted if desired)
prob_fused = 0.5 * p_tab + 0.5 * p_img
fused_class = (prob_fused >= 0.5).astype(int)

# -- Only save fused predictions in the CSV (required columns only) --
fused_df = pd.DataFrame({
    "Sample_Index": merged["Sample_Index"],
    "True_Label": merged["True_Label_tab"],
    "y_prob_fused": prob_fused,
    "y_pred_fused": fused_class
})
fused_df.to_csv(FUSE_ROOT / "fused_predictions.csv", index=False)

# -- Compute all metrics of the multimodal model only --
y_true = merged["True_Label_tab"]
y_pred = fused_class
y_prob = prob_fused
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "F1": f1_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "ROC_AUC": roc_auc_score(y_true, y_prob)
}
pd.DataFrame([metrics]).to_csv(FUSE_ROOT / "fused_metrics.csv", index=False)

print("Fusion accuracy:", metrics["Accuracy"])
print("All multimodal metrics saved in fused_metrics.csv")

# -- Save the multimodal ensemble as a Python object --
class MultimodalEnsemble:
    def __init__(self, tabular_preds, image_preds, tab_name, img_name):
        self.tabular_preds = tabular_preds
        self.image_preds = image_preds
        self.tab_name = tab_name
        self.img_name = img_name

    def predict(self, sample_idx):
        row_tab = self.tabular_preds[self.tabular_preds["Sample_Index"] == sample_idx]
        row_img = self.image_preds[self.image_preds["Sample_Index"] == sample_idx]
        if row_tab.empty or row_img.empty:
            raise ValueError(f"Sample_Index {sample_idx} not found in predictions.")
        p_tab = row_tab["Pred_Proba_Pos"].values[0] if "Pred_Proba_Pos" in row_tab else row_tab["Predicted"].astype(float).values[0]
        p_img = row_img["Pred_Proba_Pos"].values[0] if "Pred_Proba_Pos" in row_img else row_img["Predicted"].astype(float).values[0]
        prob = 0.5 * p_tab + 0.5 * p_img
        label = int(prob >= 0.5)
        return {"prob": prob, "label": label}

# Save the callable ensemble object (for use in explanations)
mm_ensemble = MultimodalEnsemble(tab_preds, img_preds, tab_name, img_name)
joblib.dump(mm_ensemble, FUSE_ROOT / "multimodal_model.pkl")

print("✅ Saved multimodal ensemble object to 'multimodal_model.pkl'")
