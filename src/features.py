"""Feature engineering.

The ULB dataset has no customer ID (V1-V28 are anonymised PCA components), so per-customer features
are impossible. We build the closest honest equivalents, using ONLY past transactions so nothing leaks
from the future:

* amount vs. recent average  -> amount relative to the rolling mean/std of the previous N transactions
* time-of-day                -> hour derived from `Time` (seconds since the first transaction; this assumes
                                the capture started near midnight, so treat it as an approximation)
* transaction velocity       -> number of transactions in the previous 60 s / 600 s, and the gap since
                                the previous transaction (global, not per customer)
"""
import numpy as np
import pandas as pd

from .config import ROLLING_WINDOW, VELOCITY_WINDOWS

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
ENGINEERED_COLUMNS = [
    "log_amount",
    "log_amount_ratio",
    "amount_zscore",
    "hour_sin",
    "hour_cos",
    "log_time_since_prev",
    *[f"txn_count_{w}s" for w in VELOCITY_WINDOWS],
]
FEATURE_COLUMNS = V_COLUMNS + ENGINEERED_COLUMNS


def engineer_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Return a copy of df (sorted by Time) with engineered feature columns added."""
    out = df.sort_values("Time", kind="stable").reset_index(drop=True).copy()
    amount = out["Amount"].astype(float)
    time = out["Time"].astype(float).to_numpy()

    # --- Amount vs. recent average (shift(1) => current row never sees itself or the future) ---
    past = amount.shift(1)
    roll_mean = past.rolling(window, min_periods=1).mean().fillna(amount)
    roll_std = past.rolling(window, min_periods=2).std().fillna(0.0)
    out["log_amount"] = np.log1p(amount)
    out["log_amount_ratio"] = np.log1p(amount / (roll_mean + 1.0))
    out["amount_zscore"] = ((amount - roll_mean) / (roll_std + 1.0)).clip(-5, 5)

    # --- Time of day (cyclic encoding so 23:00 and 00:00 are neighbours) ---
    hour = (time // 3600) % 24
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # --- Velocity: how busy was the system just before this transaction? ---
    gap = pd.Series(time).diff().fillna(0.0).to_numpy()
    out["log_time_since_prev"] = np.log1p(gap)
    idx = np.arange(len(out))
    for w in VELOCITY_WINDOWS:
        left = np.searchsorted(time, time - w, side="left")
        out[f"txn_count_{w}s"] = idx - left  # earlier transactions inside the window

    return out
