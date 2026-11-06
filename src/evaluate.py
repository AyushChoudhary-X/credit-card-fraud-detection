"""Metrics, thresholds and the cost model.

Why not accuracy? With 0.17% fraud, a model that says "never fraud" is 99.83% accurate and catches nothing.
We use Precision, Recall, F1 and PR-AUC (average precision), which all focus on the rare fraud class.
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pr_auc(y, s) -> float:
    return float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def confusion(y, pred) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_at_threshold(y, s, thr) -> dict:
    pred = (np.asarray(s) >= thr).astype(int)
    c = confusion(y, pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "alerts": c["tp"] + c["fp"], **c}


def threshold_max_f1(y, s) -> float:
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def threshold_for_precision(y, s, target: float) -> float:
    """Lowest threshold whose precision >= target (i.e. the highest-recall point that still meets it).
    If the target is never reached, fall back to the most conservative threshold."""
    p, r, t = precision_recall_curve(y, s)
    ok = np.flatnonzero(p[:-1] >= target)
    if len(ok) == 0:
        return float(t[-1])
    best = ok[np.argmax(r[:-1][ok])]
    return float(t[best])


def best_recall_at_precision(y, s, target: float) -> float:
    """Best recall on the given data's own PR curve with precision >= target (optimistic if y is the test set)."""
    p, r, _ = precision_recall_curve(y, s)
    ok = p >= target
    return float(r[ok].max()) if ok.any() else 0.0


# ----------------------------- cost model -----------------------------
def total_cost(y, amount, pred, cost_fp: float) -> dict:
    """Missed fraud costs the transaction amount; every false alert costs `cost_fp`."""
    y, amount, pred = np.asarray(y), np.asarray(amount, float), np.asarray(pred)
    missed = float(amount[(y == 1) & (pred == 0)].sum())
    n_fp = int(((y == 0) & (pred == 1)).sum())
    return {"missed_fraud_cost": missed, "false_alert_cost": n_fp * cost_fp,
            "total_cost": missed + n_fp * cost_fp}


def cost_optimal_threshold(y, amount, s, cost_fp: float) -> float:
    """Threshold minimising total cost on the given data (flag the top-k scores, choose best k)."""
    y, amount, s = np.asarray(y), np.asarray(amount, float), np.asarray(s, float)
    order = np.argsort(-s, kind="stable")
    ys, am, ss = y[order], amount[order], s[order]
    fp_cum = np.cumsum(1 - ys)
    caught_cum = np.cumsum(am * ys)
    cost = np.empty(len(s) + 1)
    cost[0] = (am * ys).sum()                      # flag nothing
    cost[1:] = (am * ys).sum() - caught_cum + cost_fp * fp_cum
    k = int(np.argmin(cost))
    if k == 0:
        return float("inf")
    # flag everything with score >= the k-th highest score (ties are flagged together)
    return float(ss[k - 1])


def bootstrap_ci(y, s, thr, n_boot: int = 500, seed: int = 42) -> dict:
    """95% bootstrap confidence intervals - important because the test set only has ~75 frauds."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    n = len(y)
    out = {"pr_auc": [], "precision": [], "recall": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0:
            continue
        out["pr_auc"].append(average_precision_score(yb, sb))
        m = metrics_at_threshold(yb, sb, thr)
        out["precision"].append(m["precision"])
        out["recall"].append(m["recall"])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}
