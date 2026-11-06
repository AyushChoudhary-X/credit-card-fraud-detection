"""Unit tests. Run with:  python -m unittest discover -s tests -v   (or pytest)"""
import unittest

import numpy as np
import pandas as pd

from src.evaluate import cost_optimal_threshold, metrics_at_threshold, threshold_for_precision, total_cost
from src.features import FEATURE_COLUMNS, engineer_features
from src.resampling import random_undersample, smote


def make_df(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.sort(rng.integers(0, 100000, n)).astype(float)
    df["Amount"] = rng.gamma(2.0, 40.0, n)
    df["Class"] = (rng.random(n) < 0.02).astype(int)
    return df


class TestFeatures(unittest.TestCase):
    def test_columns_and_no_nans(self):
        out = engineer_features(make_df())
        self.assertTrue(set(FEATURE_COLUMNS).issubset(out.columns))
        self.assertFalse(out[FEATURE_COLUMNS].isna().any().any())

    def test_no_future_leakage(self):
        """Changing a LATER transaction's amount must not change earlier rows' features."""
        df = make_df()
        a = engineer_features(df)
        df2 = df.copy()
        df2.loc[df2.index[-1], "Amount"] = 1e6
        b = engineer_features(df2)
        pd.testing.assert_frame_equal(a[FEATURE_COLUMNS].iloc[:-1], b[FEATURE_COLUMNS].iloc[:-1])

    def test_velocity_counts_only_past(self):
        df = pd.DataFrame({"Time": [0.0, 10.0, 20.0, 100.0], "Amount": [1, 1, 1, 1], "Class": 0})
        for i in range(1, 29):
            df[f"V{i}"] = 0.0
        out = engineer_features(df)
        # t=100: window is [40, 100) -> no earlier transactions fall inside it
        self.assertEqual(out["txn_count_60s"].tolist(), [0, 1, 2, 0])

    def test_hour_is_cyclic(self):
        out = engineer_features(make_df())
        np.testing.assert_allclose(out.hour_sin ** 2 + out.hour_cos ** 2, 1.0)


class TestResampling(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.X = rng.normal(size=(1000, 5))
        self.y = np.zeros(1000, dtype=int)
        self.y[:20] = 1
        self.X[:20] += 3

    def test_smote_reaches_ratio_and_stays_in_fraud_region(self):
        Xr, yr = smote(self.X, self.y, ratio=0.5)
        self.assertEqual(int((yr == 0).sum()), 980)
        self.assertEqual(int(yr.sum()), 490)
        synthetic = Xr[len(self.X):]
        lo, hi = self.X[:20].min(0), self.X[:20].max(0)
        self.assertTrue(((synthetic >= lo - 1e-9) & (synthetic <= hi + 1e-9)).all())  # interpolation => inside the hull box

    def test_undersample_keeps_all_fraud(self):
        Xr, yr = random_undersample(self.X, self.y, ratio=0.25)
        self.assertEqual(int(yr.sum()), 20)
        self.assertEqual(int((yr == 0).sum()), 80)


class TestCost(unittest.TestCase):
    def test_total_cost(self):
        y = np.array([1, 1, 0, 0]); amt = np.array([100.0, 50.0, 10.0, 10.0]); pred = np.array([1, 0, 1, 0])
        c = total_cost(y, amt, pred, cost_fp=5)
        self.assertEqual(c["missed_fraud_cost"], 50.0)
        self.assertEqual(c["false_alert_cost"], 5.0)

    def test_cost_optimal_threshold_beats_extremes(self):
        rng = np.random.default_rng(3)
        y = (rng.random(5000) < 0.05).astype(int)
        s = np.clip(0.5 * y + rng.normal(0.25, 0.15, 5000), 0, 1)
        amt = rng.gamma(2, 50, 5000)
        thr = cost_optimal_threshold(y, amt, s, 5.0)
        best = total_cost(y, amt, (s >= thr).astype(int), 5.0)["total_cost"]
        for t in (0.0, 0.5, 1.1):
            self.assertLessEqual(best, total_cost(y, amt, (s >= t).astype(int), 5.0)["total_cost"] + 1e-9)

    def test_precision_threshold_meets_target(self):
        rng = np.random.default_rng(4)
        y = (rng.random(5000) < 0.05).astype(int)
        s = np.clip(0.6 * y + rng.normal(0.2, 0.15, 5000), 0, 1)
        thr = threshold_for_precision(y, s, 0.9)
        self.assertGreaterEqual(metrics_at_threshold(y, s, thr)["precision"], 0.9)


if __name__ == "__main__":
    unittest.main()
