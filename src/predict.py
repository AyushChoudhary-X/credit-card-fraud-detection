"""Score new transactions with the trained model.

CLI:  python -m src.predict path/to/transactions.csv [--threshold cost_optimal|precision_90|max_f1] [--out scored.csv]

The input needs the same raw columns as creditcard.csv (Time, V1..V28, Amount); `Class` is optional.
Because velocity / recent-average features look at *previous* transactions, pass transactions in a
batch (ideally many consecutive ones) rather than a single row.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH
from .features import engineer_features
from .models import get_scores


def load_artifact(path=MODEL_PATH) -> dict:
    return joblib.load(path)


def score_transactions(raw: pd.DataFrame, artifact: dict, threshold: str = "cost_optimal") -> pd.DataFrame:
    """Return the engineered-and-sorted transactions plus `fraud_score` and `flagged` columns."""
    needed = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    feats = engineer_features(raw)
    X = artifact["scaler"].transform(feats[artifact["feature_columns"]])
    out = feats[["Time", "Amount"] + (["Class"] if "Class" in feats else [])].copy()
    out["fraud_score"] = get_scores(artifact["model"], X)
    thr = artifact["thresholds"][threshold]
    out["flagged"] = (out["fraud_score"] >= thr).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--threshold", default="cost_optimal", choices=["cost_optimal", "precision_90", "max_f1"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    art = load_artifact()
    scored = score_transactions(pd.read_csv(args.csv), art, args.threshold)
    print(f"Model: {art['model_name']} ({art['strategy']}) | threshold={art['thresholds'][args.threshold]:.4f}")
    print(f"Scored {len(scored):,} transactions, flagged {int(scored.flagged.sum())}")
    if "Class" in scored:
        print(f"Actual frauds: {int(scored.Class.sum())}, caught: {int(((scored.Class == 1) & (scored.flagged == 1)).sum())}")
    print(scored.sort_values("fraud_score", ascending=False).head(10).to_string(index=False))
    if args.out:
        scored.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
