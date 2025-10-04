import argparse, subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train-tab","train-img","eval","fuse","surrogates","explain"])
    ap.add_argument("--id", type=int, default=1)
    args = ap.parse_args()

    mapping = {
        "train-tab": "python -m src.python.train_tabular",
        "train-img": "python -m src.python.train_image",
        "eval": "python -m src.python.evaluate_models",
        "fuse": "python -m src.python.fuse_multimodal",
        "surrogates": "python -m src.python.generate_surrogates",
        "explain": f"python -m src.python.explain --id {args.id}",
    }
    subprocess.check_call(mapping[args.cmd], shell=True)

if __name__ == "__main__":
    main()
