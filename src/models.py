"""Model factories and a uniform scoring interface."""
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import SEED


def build_model(name: str, class_weight=None):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=SEED)
    if name == "random_forest":
        cw = "balanced_subsample" if class_weight == "balanced" else None
        return RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=3, max_features="sqrt",
            class_weight=cw, n_jobs=-1, random_state=SEED,
        )
    if name == "isolation_forest":
        # Unsupervised: never sees labels. `contamination` only affects its internal threshold, not the scores.
        return IsolationForest(n_estimators=200, contamination=0.002, random_state=SEED, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


def get_scores(model, X) -> np.ndarray:
    """Higher score = more likely fraud, for every model type."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return -model.score_samples(X)  # IsolationForest: lower score_samples = more anomalous
