import argparse
import os
import subprocess
import sys
import time
import pandas as pd
import numpy as np

def select_top_k_features(csv_path: str, out_dir: str, top_k: int) -> str:
    """
    Load surrogate CSV, select top_k features by mutual information with 'class',
    save a reduced CSV (keeping 'class'), and return its path.
    """
    from sklearn.feature_selection import mutual_info_classif

    df = pd.read_csv(csv_path)
    if "class" not in df.columns:
        raise ValueError("Surrogate CSV must contain a 'class' column.")
    if top_k <= 0 or top_k >= (df.shape[1] - 1):
        # no selection or trivial -> return original
        return csv_path

    X = df.drop(columns=["class"])
    y = df["class"].astype(int).values

    # compute MI (handle constant columns robustly)
    X_num = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    try:
        mi = mutual_info_classif(X_num, y, random_state=42)
    except Exception:
        # fallback: variance ranking if MI fails
        mi = X_num.var(axis=0)

    # top-k indices
    idx_sorted = np.argsort(mi)[::-1]
    keep_idx = idx_sorted[:top_k]
    keep_cols = list(X.columns[keep_idx])

    reduced = pd.concat([X[keep_cols], df["class"]], axis=1)
    out_csv = os.path.join(out_dir, "surrogate_dataset_selected.csv")
    reduced.to_csv(out_csv, index=False)

    # Save which columns were selected
    pd.DataFrame({"Selected_Feature": keep_cols}).to_csv(
        os.path.join(out_dir, "selected_features.csv"), index=False
    )

    return out_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to surrogate_dataset.csv")
    parser.add_argument("--run_dir", required=True, help="Output directory for FRBS results")
    parser.add_argument("--top_k", type=int, default=0, help="Use only top-K features by MI (0 = use all)")
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    # If requested, reduce features first
    csv_for_r = args.csv
    if args.top_k and args.top_k > 0:
        csv_for_r = select_top_k_features(args.csv, args.run_dir, args.top_k)

    start = time.perf_counter()
    # Call R script
    cmd = [
        "Rscript",
        "/workspace/src/r/train_fuzzy_surrogate.R",
        "--csv", csv_for_r,
        "--outdir", args.run_dir
    ]
    subprocess.check_call(cmd)
    runtime = time.perf_counter() - start

    with open(os.path.join(args.run_dir, "frbs_time.txt"), "w") as f:
        f.write(str(runtime))

    print(f"✅ FRBS surrogate completed in {runtime:.3f}s. Outputs in {args.run_dir}")

if __name__ == "__main__":
    main()
