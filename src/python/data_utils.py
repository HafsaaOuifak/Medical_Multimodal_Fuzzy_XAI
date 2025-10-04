from pathlib import Path
import pandas as pd
from .config import RAW_DIR, TARGET_COLUMN, ID_COLUMN

def load_tabular(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tabular file not found: {p}")
    df = pd.read_csv(p)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in {p}")
    if ID_COLUMN not in df.columns:
        df.insert(0, ID_COLUMN, range(1, len(df) + 1))
    return df

def save_df(df: pd.DataFrame, path: Path | str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def split_Xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y
