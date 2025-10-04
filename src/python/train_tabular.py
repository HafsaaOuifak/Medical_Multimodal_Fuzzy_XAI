import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier

TABULAR_PATH = os.getenv("TABULAR_DRIVE_PATH", "/workspace/gdrive/Data_tabular/dataset_tabular.csv")
NOISY_INDICES_PATH = os.getenv("NOISY_INDICES_PATH", "/workspace/gdrive/Data_normalized_array/Label_issues_index.npy")
tab_out_root = "/workspace/tabular_models"

os.makedirs(tab_out_root, exist_ok=True)

train_df = pd.read_csv(TABULAR_PATH)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df = train_df.drop(columns=[
    'Unnamed: 0', "anon_patientid", 'anon_filename',
    "rad_recall_type_right_0", "rad_recall_type_left_0", "rad_r1", "rad_r2"
], errors='ignore')
train_df['rad_decision'] = train_df['rad_decision'].astype(int)

cols = ['libra_breastarea', 'libra_densearea', 'libra_percentdensity']
mmscaler = MinMaxScaler()
mmscaler.fit(train_df[cols])
train_df[cols] = mmscaler.transform(train_df[cols])

features = [
    'x_age', 'x_case', 'x_type', 'x_lymphnode_met', 'rad_timing', 'rad_recall',
    'libra_breastarea', 'libra_densearea', 'libra_percentdensity',
    'rad_recall_type_right_1', 'rad_recall_type_right_2',
    'rad_recall_type_left_1', 'rad_recall_type_left_2'
]
X_train = train_df[features]
Y_train = train_df["rad_decision"]

noisy_rank_label = np.load(NOISY_INDICES_PATH)
X_list = np.array(X_train)
Y_list = np.array(Y_train)
X_clean = [x for i, x in enumerate(X_list) if i not in noisy_rank_label]
Y_clean = [x for i, x in enumerate(Y_list) if i not in noisy_rank_label]
X_tab = pd.DataFrame(X_clean, columns=features)
y_tab = np.array(Y_clean)

os.makedirs(os.path.join(tab_out_root, "Scaler"), exist_ok=True)
joblib.dump(mmscaler, os.path.join(tab_out_root, "Scaler", "minmax_scaler.pkl"))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_tab)
joblib.dump(scaler, os.path.join(tab_out_root, "Scaler", "scaler.pkl"))

def build_mlp():
    tab_model = Sequential([
        Input(shape=(X_tab.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    tab_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return tab_model

models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': KerasClassifier(model=build_mlp, epochs=10, batch_size=32, verbose=0)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, _ in models.items():
    print(f"🔁 Training {model_name} with 5-fold CV...")
    model_folder = os.path.join(tab_out_root, model_name)
    os.makedirs(model_folder, exist_ok=True)
    per_model_metrics = []
    per_model_predictions = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_tab, y_tab), start=1):
        if model_name == 'DecisionTree':
            model = DecisionTreeClassifier(random_state=42)
            X_train_fold, X_val_fold = X_tab.values[train_idx], X_tab.values[val_idx]
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(random_state=42)
            X_train_fold, X_val_fold = X_tab.values[train_idx], X_tab.values[val_idx]
        elif model_name == 'SVM':
            model = SVC(probability=True, random_state=42)
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
        elif model_name == 'MLP':
            model = KerasClassifier(model=build_mlp, epochs=10, batch_size=32, verbose=0)
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        y_train_fold, y_val_fold = y_tab[train_idx], y_tab[val_idx]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        if hasattr(y_pred, "flatten"):
            y_pred = y_pred.flatten()
        if model_name == 'MLP' and y_pred.dtype != int:
            y_pred = (y_pred > 0.5).astype(int)

        y_proba = None
        try:
            proba = model.predict_proba(X_val_fold)
            if proba.ndim == 2 and proba.shape[1] == 2:
                y_proba = proba[:, 1]
            elif proba.ndim == 2 and proba.shape[1] == 1:
                y_proba = proba[:, 0]
        except Exception:
            try:
                proba = model.predict_proba(X_val_fold)
                if proba.ndim == 2:
                    y_proba = proba[:, -1]
            except Exception:
                y_proba = None

        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, zero_division=0)
        rec = recall_score(y_val_fold, y_pred, zero_division=0)
        f1 = f1_score(y_val_fold, y_pred, zero_division=0)
        per_model_metrics.append({
            'Model': model_name, 'Fold': fold, 'Accuracy': acc,
            'Precision': prec, 'Recall': rec, 'F1_Score': f1
        })

        if y_proba is not None:
            for j, (idx_pos, pred, p) in enumerate(zip(val_idx, y_pred, y_proba)):
                per_model_predictions.append({
                    'Model': model_name, 'Fold': fold, 'Sample_Index': int(idx_pos),
                    'Predicted': int(pred), 'Pred_Proba_Pos': float(p), 'True_Label': int(y_tab[idx_pos])
                })
        else:
            for j, (idx_pos, pred) in enumerate(zip(val_idx, y_pred)):
                per_model_predictions.append({
                    'Model': model_name, 'Fold': fold, 'Sample_Index': int(idx_pos),
                    'Predicted': int(pred), 'True_Label': int(y_tab[idx_pos])
                })

        if model_name == "MLP":
            model.model_.save(os.path.join(model_folder, f"{model_name}_fold{fold}.h5"))
        else:
            joblib.dump(model, os.path.join(model_folder, f"{model_name}_fold{fold}.pkl"))

    metrics_df = pd.DataFrame(per_model_metrics)
    metrics_df.to_csv(os.path.join(model_folder, "metrics.csv"), index=False)
    preds_df = pd.DataFrame(per_model_predictions)
    preds_df.to_csv(os.path.join(model_folder, "predictions.csv"), index=False)

print("✅ Tabular training complete. Metrics and predictions saved in '/workspace/tabular_models/'.")
