import subprocess
from pathlib import Path

def all_tabular_models_exist(tab_root: Path, model_names=("DecisionTree", "RandomForest", "SVM", "MLP")):
    for m in model_names:
        model_dir = tab_root / m
        if not model_dir.exists() or not any(f.suffix in (".pkl", ".h5") for f in model_dir.glob("*")):
            return False
    return True

def ensure_tabular():
    tab_root = Path("/workspace/trained_models")
    if all_tabular_models_exist(tab_root):
        print("[train_once] All tabular models exist — skipping.")
        return
    print("[train_once] Training all tabular models...")
    subprocess.check_call(["python", "-m", "src.python.train_tabular"])

def ensure_image():
    img_root = Path("/workspace/image_trained_models")
    for m in ("ResNet50", "MobileNetV2", "EfficientNetB0"):
        d = img_root / m
        if not d.exists() or not any(f.suffix == ".h5" for f in d.glob("*.h5")):
            print("[train_once] Training all image models...")
            subprocess.check_call(["python", "-m", "src.python.train_image"])
            return
    print("[train_once] All image models exist — skipping.")

def main():
    ensure_tabular()
    ensure_image()
    print("[train_once] All models trained.")

if __name__ == "__main__":
    main()
