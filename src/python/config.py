import os
from pathlib import Path

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/workspace/data/artifacts"))
RAW_DIR       = Path(os.getenv("RAW_DIR", "/workspace/data/raw"))
SURROGATE_DIR = Path(os.getenv("SURROGATE_DIR", "/workspace/data/artifacts/surrogates"))

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "target")
ID_COLUMN     = os.getenv("ID_COLUMN", "id")

# Standard artifact locations
TAB_MODEL_DIR = ARTIFACTS_DIR / "tabular_model"
IMG_MODEL_DIR = ARTIFACTS_DIR / "image_model"
FUSE_DIR      = ARTIFACTS_DIR / "fusion"
PRED_DIR      = ARTIFACTS_DIR / "predictions"
REPORT_DIR    = ARTIFACTS_DIR / "reports"

for d in [ARTIFACTS_DIR, SURROGATE_DIR, TAB_MODEL_DIR, IMG_MODEL_DIR, FUSE_DIR, PRED_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
