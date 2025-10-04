import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, f1_score
)
from .config import PRED_DIR
from .data_utils import save_df

def is_classification_series(y: pd.Series) -> bool:
    u = pd.unique(y.dropna())
    try:
        return len(u) <= 20 and np.allclose(u, u.astype(int))
    except Exception:
        return False

def main():
    tab = pd.read_csv(PRED_DIR / "tabular_predictions.csv")
    img = pd.read_csv(PRED_DIR / "image_predictions.csv")

    merged = tab.merge(img[[ "id", "y_pred_img" ] + ([ "y_prob_img" ] if "y_prob_img" in img.columns else [])], on="id", how="inner")
    y_true = merged["y_true"]

    is_cls = is_classification_series(y_true)

    rows = []
    if is_cls:
        # Tabular
        rows.append({
            "Model": "Tabular",
            "Metric": "Accuracy",
            "Value": accuracy_score(y_true, merged["y_pred_tab"])
        })
        rows.append({
            "Model": "Tabular",
            "Metric": "F1",
            "Value": f1_score(y_true, merged["y_pred_tab"], average="binary" if len(pd.unique(y_true))==2 else "macro")
        })
        # Image
        rows.append({
            "Model": "Image",
            "Metric": "Accuracy",
            "Value": accuracy_score(y_true, merged["y_pred_img"])
        })
        rows.append({
            "Model": "Image",
            "Metric": "F1",
            "Value": f1_score(y_true, merged["y_pred_img"], average="binary" if len(pd.unique(y_true))==2 else "macro")
        })
    else:
        # Regression
        rows.append({
            "Model": "Tabular",
            "Metric": "R2",
            "Value": r2_score(y_true, merged["y_pred_tab"])
        })
        rows.append({
            "Model": "Tabular",
            "Metric": "MSE",
            "Value": mean_squared_error(y_true, merged["y_pred_tab"])
        })
        rows.append({
            "Model": "Image",
            "Metric": "R2",
            "Value": r2_score(y_true, merged["y_pred_img"])
        })
        rows.append({
            "Model": "Image",
            "Metric": "MSE",
            "Value": mean_squared_error(y_true, merged["y_pred_img"])
        })

    dfm = pd.DataFrame(rows)
    save_df(dfm, PRED_DIR / "metrics_base.csv")
    print(dfm)

if __name__ == "__main__":
    main()
