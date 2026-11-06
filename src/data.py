"""Data loading and chronological splitting."""
import pandas as pd

from .config import DATA_PATH, TRAIN_FRAC, VAL_FRAC

REQUIRED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load creditcard.csv (Kaggle: mlg-ulb/creditcardfraud) and validate it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it in data/."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Split by time so the model is always evaluated on transactions that happen AFTER training data.

    A random split would let the model see "future" fraud patterns, which overstates real-world performance.
    """
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return (
        df.iloc[:i_train].reset_index(drop=True),
        df.iloc[i_train:i_val].reset_index(drop=True),
        df.iloc[i_val:].reset_index(drop=True),
    )
