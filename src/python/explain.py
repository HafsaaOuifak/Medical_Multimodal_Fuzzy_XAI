import argparse
import os
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# --- IMPORTANT: provide alias so old pickles that expect
# src.python.explain.MultimodalEnsemble can find the class here.
from src.python.ensemble import MultimodalEnsemble as _MME
MultimodalEnsemble = _MME  # alias for backward-compatible unpickling

from src.python.generate_surrogates import generate_surrogates


def load_x_tab_x_img(idx: int):
    """
    Load the tabular row and image for a single index. Adjust feature list if needed.
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Index to explain")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--tab_noise", type=float, default=0.05)
    parser.add_argument("--img_noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Use only top-K features (by MI) for fuzzy explainer; 0 = all features"
    )
    args = parser.parse_args()

    run_dir = f"/workspace/xai_fuzzy_model/run_{args.id}_{int(time.time())}"
    os.makedirs(run_dir, exist_ok=True)

    # Load data point
    x_tab0, x_img0 = load_x_tab_x_img(args.id)

    # Load multimodal ensemble (pickled). The alias above ensures old pickles load.
    mm_pkl = "/workspace/multimodal_results/multimodal_model.pkl"
    if not os.path.exists(mm_pkl):
        raise FileNotFoundError(
            f"Multimodal model not found at '{mm_pkl}'. Run the fusion step first."
        )
    multimodal_model = joblib.load(mm_pkl)

    # Generate & save surrogate dataset with preserved feature names
    df_sur = generate_surrogates(
        x_tab0=x_tab0,
        x_img0=x_img0,
        multimodal_model=multimodal_model,
        n=args.n,
        tab_noise=args.tab_noise,
        img_noise=args.img_noise,
        seed=args.seed
    )
    surrogate_csv = os.path.join(run_dir, "surrogate_dataset.csv")
    df_sur.to_csv(surrogate_csv, index=False)

    # Train FRBS surrogate (R), with optional top-K feature selection
    import subprocess
    start = time.perf_counter()
    subprocess.check_call([
        "python", "-m", "src.python.train_fuzzy_surrogate",
        "--csv", surrogate_csv,
        "--run_dir", run_dir,
        "--top_k", str(args.top_k)
    ])
    elapsed = time.perf_counter() - start

    print(f"✅ Explained ID {args.id}. All artifacts in: {run_dir} (elapsed {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
