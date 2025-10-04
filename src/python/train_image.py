# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf

# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
# from tensorflow.keras.models import Model

# IMG_PATH = os.getenv("IMAGE_NPY_DATA_PATH", "/workspace/gdrive/Data_normalized_array/clean_data.npy")
# LABEL_PATH = os.getenv("IMAGE_NPY_LABELS_PATH", "/workspace/gdrive/Data_normalized_array/clean_labels.npy")
# img_out_root = "/workspace/image_models"

# os.makedirs(img_out_root, exist_ok=True)

# # --- Load and preprocess images ---
# X_img = np.load(IMG_PATH)
# Y_ = np.load(LABEL_PATH)
# y_img = tf.keras.utils.to_categorical(Y_, num_classes=2)

# # Use only first 25 samples (as in your notebook; remove or change if desired)
# X_img = X_img[:25]
# y_img = y_img[:25]
# y_img = np.argmax(y_img, axis=1)

# # Faithfully repeat to RGB if not already RGB (as in your notebook)
# X_img_scaled = X_img / 255.0
# if X_img_scaled.ndim == 3 or X_img_scaled.shape[-1] != 3:
#     # (N, H, W) or (N, H, W, 1): add channel axis and repeat to 3
#     X_img_scaled = np.repeat(X_img_scaled[..., np.newaxis], 3, axis=-1)

# print("DEBUG: After repeat, shape is", X_img_scaled.shape)
# X_img_resized = tf.image.resize(X_img_scaled, (224, 224)).numpy()
# y_class = y_img if y_img.ndim == 1 else np.argmax(y_img, axis=1)

# model_dict = {
#     "ResNet50": ResNet50,
#     "MobileNetV2": MobileNetV2,
#     "EfficientNetB0": EfficientNetB0
# }

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# for model_name, model_fn in model_dict.items():
#     print(f"🔁 Training {model_name} with 5-fold CV...")
#     model_folder = os.path.join(img_out_root, model_name)
#     os.makedirs(model_folder, exist_ok=True)

#     per_model_metrics = []
#     per_model_predictions = []

#     fold = 1
#     for train_idx, test_idx in skf.split(X_img_resized, y_class):
#         X_train, X_test = X_img_resized[train_idx], X_img_resized[test_idx]
#         y_train, y_test = y_class[train_idx], y_class[test_idx]

#         tf.keras.backend.clear_session()
#         base_model = model_fn(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
#         base_model.trainable = False
#         x = GlobalAveragePooling2D()(base_model.output)
#         x = Dense(64, activation="relu")(x)
#         out = Dense(1, activation="sigmoid")(x)
#         img_model = Model(inputs=base_model.input, outputs=out)
#         img_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#         img_model.fit(X_train, y_train, epochs=5, batch_size=8, verbose=0)

#         preds_prob = img_model.predict(X_test, verbose=0).flatten()
#         preds_cls = (preds_prob > 0.5).astype("int32")

#         acc = accuracy_score(y_test, preds_cls)
#         prec = precision_score(y_test, preds_cls, zero_division=0)
#         rec = recall_score(y_test, preds_cls, zero_division=0)
#         f1 = f1_score(y_test, preds_cls, zero_division=0)

#         per_model_metrics.append({
#             "Model": model_name, "Fold": fold, "Accuracy": acc,
#             "Precision": prec, "Recall": rec, "F1_Score": f1
#         })

#         for idx_pos, y_true, y_hat, p in zip(test_idx, y_test, preds_cls, preds_prob):
#             per_model_predictions.append({
#                 "Model": model_name, "Fold": fold, "Sample_Index": int(idx_pos),
#                 "True_Label": int(y_true), "Predicted": int(y_hat), "Pred_Proba_Pos": float(p)
#             })

#         model_path = os.path.join(model_folder, f"{model_name}_fold{fold}.h5")
#         img_model.save(model_path)
#         fold += 1

#     metrics_df = pd.DataFrame(per_model_metrics)
#     metrics_df.to_csv(os.path.join(model_folder, "metrics.csv"), index=False)
#     preds_df = pd.DataFrame(per_model_predictions)
#     preds_df.to_csv(os.path.join(model_folder, "predictions.csv"), index=False)

# print("✅ Image training complete. Metrics and predictions saved in '/workspace/image_models/'.")

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

IMG_PATH = os.getenv("IMAGE_NPY_DATA_PATH", "/workspace/gdrive/Data_normalized_array/clean_data.npy")
LABEL_PATH = os.getenv("IMAGE_NPY_LABELS_PATH", "/workspace/gdrive/Data_normalized_array/clean_labels.npy")
img_out_root = "/workspace/image_models"

os.makedirs(img_out_root, exist_ok=True)

# --- Load and preprocess images ---
X_img = np.load(IMG_PATH)
Y_ = np.load(LABEL_PATH)
y_img = tf.keras.utils.to_categorical(Y_, num_classes=2)

X_img = X_img[:25]
y_img = y_img[:25]
y_img = np.argmax(y_img, axis=1)

# 1. Ensure shape (N, H, W, C)
if X_img.ndim == 3:
    X_img = X_img[..., np.newaxis]
if X_img.shape[-1] == 1:
    X_img = np.repeat(X_img, 3, axis=-1)
elif X_img.shape[-1] != 3:
    raise ValueError(f"Expected channel dim 1 or 3, got {X_img.shape}")

print("DEBUG: After channel repeat, X_img.shape =", X_img.shape)

X_img_scaled = X_img / 255.0
X_img_resized = tf.image.resize(X_img_scaled, (224, 224)).numpy()
y_class = y_img if y_img.ndim == 1 else np.argmax(y_img, axis=1)

model_dict = {
    "ResNet50": ResNet50,
    "MobileNetV2": MobileNetV2
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model_fn in model_dict.items():
    model_folder = os.path.join(img_out_root, model_name)
    os.makedirs(model_folder, exist_ok=True)

    per_model_metrics = []
    per_model_predictions = []

    fold = 1
    folds_to_train = []
    indices_to_train = []

    # Precompute fold indices
    splits = list(skf.split(X_img_resized, y_class))

    # For each fold, check if model file exists
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model_path = os.path.join(model_folder, f"{model_name}_fold{fold}.h5")
        if not os.path.exists(model_path):
            folds_to_train.append(fold)
            indices_to_train.append((train_idx, test_idx))
        else:
            print(f"✅ {model_name} fold {fold} already trained, skipping.")

    if not folds_to_train:
        print(f"✅ All folds already trained for {model_name}, skipping.")
        continue

    for fold, (train_idx, test_idx) in zip(folds_to_train, indices_to_train):
        print(f"🔁 Training {model_name} fold {fold}...")
        X_train, X_test = X_img_resized[train_idx], X_img_resized[test_idx]
        y_train, y_test = y_class[train_idx], y_class[test_idx]

        tf.keras.backend.clear_session()
        base_model = model_fn(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(64, activation="relu")(x)
        out = Dense(1, activation="sigmoid")(x)
        img_model = Model(inputs=base_model.input, outputs=out)
        img_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        img_model.fit(X_train, y_train, epochs=5, batch_size=8, verbose=0)

        preds_prob = img_model.predict(X_test, verbose=0).flatten()
        preds_cls = (preds_prob > 0.5).astype("int32")

        acc = accuracy_score(y_test, preds_cls)
        prec = precision_score(y_test, preds_cls, zero_division=0)
        rec = recall_score(y_test, preds_cls, zero_division=0)
        f1 = f1_score(y_test, preds_cls, zero_division=0)

        per_model_metrics.append({
            "Model": model_name, "Fold": fold, "Accuracy": acc,
            "Precision": prec, "Recall": rec, "F1_Score": f1
        })

        for idx_pos, y_true, y_hat, p in zip(test_idx, y_test, preds_cls, preds_prob):
            per_model_predictions.append({
                "Model": model_name, "Fold": fold, "Sample_Index": int(idx_pos),
                "True_Label": int(y_true), "Predicted": int(y_hat), "Pred_Proba_Pos": float(p)
            })

        model_path = os.path.join(model_folder, f"{model_name}_fold{fold}.h5")
        img_model.save(model_path)

    # After all (newly trained) folds, update metrics/predictions files
    # Read in existing metrics/predictions, if present, then append new
    metrics_path = os.path.join(model_folder, "metrics.csv")
    preds_path = os.path.join(model_folder, "predictions.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = pd.concat([metrics_df, pd.DataFrame(per_model_metrics)], ignore_index=True)
    else:
        metrics_df = pd.DataFrame(per_model_metrics)
    metrics_df.to_csv(metrics_path, index=False)

    if os.path.exists(preds_path):
        preds_df = pd.read_csv(preds_path)
        preds_df = pd.concat([preds_df, pd.DataFrame(per_model_predictions)], ignore_index=True)
    else:
        preds_df = pd.DataFrame(per_model_predictions)
    preds_df.to_csv(preds_path, index=False)

print("✅ Image training complete. Metrics and predictions saved in '/workspace/image_models/'.")
