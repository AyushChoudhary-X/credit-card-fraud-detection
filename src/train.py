"""End-to-end training + evaluation pipeline.

Run:  python -m src.train
Outputs: results/*.csv|json|md, results/figures/*.png, models/fraud_model.joblib
"""
import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import load_data, temporal_split
from .evaluate import (best_recall_at_precision, bootstrap_ci, cost_optimal_threshold, metrics_at_threshold,
                       pr_auc, roc_auc, threshold_for_precision, threshold_max_f1, total_cost)
from .features import FEATURE_COLUMNS, engineer_features
from .models import build_model, get_scores
from .resampling import random_undersample, smote

STRATEGIES = ["none", "class_weight", "undersample", "smote"]


def prepare_training_data(strategy, X, y):
    """Apply an imbalance strategy to the TRAINING set only. Returns (X, y, class_weight)."""
    if strategy == "none":
        return X, y, None
    if strategy == "class_weight":
        return X, y, "balanced"
    if strategy == "undersample":
        Xr, yr = random_undersample(X, y, C.UNDERSAMPLE_RATIO, C.SEED)
        return Xr, yr, None
    if strategy == "smote":
        Xr, yr = smote(X, y, C.SMOTE_RATIO, seed=C.SEED)
        Xr, yr = random_undersample(Xr, yr, C.SMOTE_THEN_UNDER_RATIO, C.SEED)
        return Xr, yr, None
    raise ValueError(strategy)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    for d in (C.RESULTS_DIR, C.FIG_DIR, C.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    raw = load_data()
    log(f"Loaded {len(raw):,} transactions, {int(raw.Class.sum())} frauds ({raw.Class.mean():.4%})")
    df = engineer_features(raw)
    train, val, test = temporal_split(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        log(f"{name:5s}: {len(part):>7,} rows, {int(part.Class.sum()):>3} frauds ({part.Class.mean():.4%})")

    scaler = StandardScaler().fit(train[FEATURE_COLUMNS])  # fit on train only
    X_tr, X_va, X_te = (scaler.transform(p[FEATURE_COLUMNS]) for p in (train, val, test))
    y_tr, y_va, y_te = (p["Class"].to_numpy() for p in (train, val, test))
    amt_va, amt_te = val["Amount"].to_numpy(), test["Amount"].to_numpy()

    # --------------------------------------------- the "accuracy is misleading" demo
    always_legit_acc = float((y_te == 0).mean())
    log(f"'Always legit' baseline on test: accuracy={always_legit_acc:.4%}, recall=0%")

    # ------------------------------------------------- train + evaluate everything
    rows, scores = [], {}
    runs = [(m, s) for m in ("logistic_regression", "random_forest") for s in STRATEGIES]
    runs.append(("isolation_forest", "unsupervised"))
    fitted = {}
    cache_dir = C.ROOT / ".cache"  # lets an interrupted run resume instead of refitting everything
    cache_dir.mkdir(exist_ok=True)
    for model_name, strategy in runs:
        cache_file = cache_dir / f"{model_name}__{strategy}.joblib"
        if cache_file.exists():
            model, y_fit, fit_seconds = joblib.load(cache_file)
            log(f"{model_name} / {strategy}: loaded from cache")
        else:
            t0 = time.time()
            if model_name == "isolation_forest":
                y_fit = y_tr  # labels are never used for fitting
                model = build_model(model_name).fit(X_tr)
            else:
                X_fit, y_fit, cw = prepare_training_data(strategy, X_tr, y_tr)
                model = build_model(model_name, cw).fit(X_fit, y_fit)
            fit_seconds = time.time() - t0
            joblib.dump((model, y_fit, fit_seconds), cache_file)
        t0 = time.time() - fit_seconds
        s_va, s_te = get_scores(model, X_va), get_scores(model, X_te)
        scores[(model_name, strategy)] = (s_va, s_te)
        fitted[(model_name, strategy)] = model

        thr_f1 = threshold_max_f1(y_va, s_va)               # thresholds chosen on VALIDATION only
        thr_p90 = threshold_for_precision(y_va, s_va, C.TARGET_PRECISION)
        m_f1 = metrics_at_threshold(y_te, s_te, thr_f1)
        m_p90 = metrics_at_threshold(y_te, s_te, thr_p90)
        rows.append({
            "model": model_name, "strategy": strategy, "train_rows": len(y_fit) if model_name != "isolation_forest" else len(y_tr),
            "train_frauds": int(y_fit.sum()) if model_name != "isolation_forest" else int(y_tr.sum()),
            "val_pr_auc": pr_auc(y_va, s_va), "test_pr_auc": pr_auc(y_te, s_te), "test_roc_auc": roc_auc(y_te, s_te),
            "f1opt_precision": m_f1["precision"], "f1opt_recall": m_f1["recall"], "f1opt_f1": m_f1["f1"],
            "p90_precision": m_p90["precision"], "p90_recall": m_p90["recall"], "p90_alerts": m_p90["alerts"],
            "best_case_recall_at_p90": best_recall_at_precision(y_te, s_te, C.TARGET_PRECISION),
            "fit_seconds": fit_seconds,
        })
        log(f"{model_name:20s} {strategy:12s} val PR-AUC={rows[-1]['val_pr_auc']:.3f} "
            f"test PR-AUC={rows[-1]['test_pr_auc']:.3f} F1={m_f1['f1']:.3f} ({rows[-1]['fit_seconds']:.0f}s)")

    comp = pd.DataFrame(rows)
    comp.to_csv(C.RESULTS_DIR / "model_comparison.csv", index=False)

    # ------------------------------------------- pick the final model on VALIDATION PR-AUC
    best_row = comp.sort_values("val_pr_auc", ascending=False).iloc[0]
    key = (best_row["model"], best_row["strategy"])
    final_model = fitted[key]
    s_va, s_te = scores[key]
    log(f"Final model (best validation PR-AUC): {key}")

    thr_f1 = threshold_max_f1(y_va, s_va)
    thr_p90 = threshold_for_precision(y_va, s_va, C.TARGET_PRECISION)
    thr_cost = cost_optimal_threshold(y_va, amt_va, s_va, C.COST_FALSE_ALERT)
    ci = bootstrap_ci(y_te, s_te, thr_p90)

    # ----------------------------------------------------------------- cost analysis
    n_legit_te = int((y_te == 0).sum())
    fraud_total_te = float(amt_te[y_te == 1].sum())
    cost_none = total_cost(y_te, amt_te, np.zeros_like(y_te), C.COST_FALSE_ALERT)
    cost_all = total_cost(y_te, amt_te, np.ones_like(y_te), C.COST_FALSE_ALERT)
    pred_cost = (s_te >= thr_cost).astype(int)
    cost_model = total_cost(y_te, amt_te, pred_cost, C.COST_FALSE_ALERT)
    cm_cost = metrics_at_threshold(y_te, s_te, thr_cost)

    sens = []
    for c_fp in (1, 2, 5, 10, 25, 50, 100, 250):
        t = cost_optimal_threshold(y_va, amt_va, s_va, c_fp)
        m = metrics_at_threshold(y_te, s_te, t)
        c = total_cost(y_te, amt_te, (s_te >= t).astype(int), c_fp)
        base = total_cost(y_te, amt_te, np.zeros_like(y_te), c_fp)["total_cost"]
        sens.append({"cost_per_false_alert": c_fp, "threshold": t, "alerts": m["alerts"], "recall": m["recall"],
                     "precision": m["precision"], "total_cost": c["total_cost"], "cost_without_model": base})
    sens = pd.DataFrame(sens)
    sens.to_csv(C.RESULTS_DIR / "cost_sensitivity.csv", index=False)

    # ----------------------------------------------------------- explainability
    rng = np.random.default_rng(C.SEED)
    fraud_idx = np.flatnonzero(y_va == 1)
    legit_idx = rng.choice(np.flatnonzero(y_va == 0), size=min(8000, int((y_va == 0).sum())), replace=False)
    sub = np.concatenate([fraud_idx, legit_idx])
    pi = permutation_importance(final_model, X_va[sub], y_va[sub], scoring="average_precision",
                                n_repeats=5, random_state=C.SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": pi.importances_mean, "std": pi.importances_std})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(C.RESULTS_DIR / "feature_importance.csv", index=False)

    # -------------------------------------------------------------------- figures
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

    # 1. accuracy is misleading
    fig, ax = plt.subplots(figsize=(6, 3.6))
    m_final = metrics_at_threshold(y_te, s_te, thr_p90)
    acc_final = (m_final["tp"] + m_final["tn"]) / len(y_te)
    labels = ["Always predict\n'legit'", f"{key[0].replace('_', ' ')}\n({key[1]})"]
    ax.bar(np.arange(2) - 0.18, [always_legit_acc, acc_final], 0.36, label="Accuracy", color="#9aa5b1")
    ax.bar(np.arange(2) + 0.18, [0.0, m_final["recall"]], 0.36, label="Fraud recall", color="#d1495b")
    ax.set_xticks(range(2)); ax.set_xticklabels(labels); ax.set_ylim(0, 1.08)
    ax.set_title("Accuracy hides the problem: a useless model scores 99.8%")
    ax.legend(loc="center right", frameon=False); fig.tight_layout()
    fig.savefig(C.FIG_DIR / "accuracy_is_misleading.png"); plt.close(fig)

    # 2. strategy comparison
    sup = comp[comp.model != "isolation_forest"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (mname, grp) in enumerate(sup.groupby("model")):
        ax.bar(np.arange(len(STRATEGIES)) + (i - 0.5) * 0.38, grp.set_index("strategy").loc[STRATEGIES, "test_pr_auc"],
               0.38, label=mname.replace("_", " "))
    iso = comp[comp.model == "isolation_forest"].iloc[0]
    ax.axhline(iso.test_pr_auc, color="k", ls="--", lw=1, label=f"isolation forest ({iso.test_pr_auc:.2f})")
    ax.axhline(y_te.mean(), color="r", ls=":", lw=1, label=f"random guess ({y_te.mean():.4f})")
    ax.set_xticks(range(len(STRATEGIES))); ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("Test PR-AUC"); ax.set_title("Imbalance strategy x model (test set)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(C.FIG_DIR / "strategy_comparison.png"); plt.close(fig)

    # 3. PR curves: best strategy per model family (by validation PR-AUC)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for mname in ("logistic_regression", "random_forest", "isolation_forest"):
        sub_rows = comp[comp.model == mname].sort_values("val_pr_auc", ascending=False).iloc[0]
        _, st = scores[(mname, sub_rows["strategy"])]
        p, r, _ = precision_recall_curve(y_te, st)
        ax.plot(r, p, label=f"{mname.replace('_', ' ')} / {sub_rows['strategy']} (AP={sub_rows['test_pr_auc']:.2f})")
    ax.axhline(y_te.mean(), color="r", ls=":", lw=1)
    ax.axhline(C.TARGET_PRECISION, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves (test set)")
    ax.legend(frameon=False, fontsize=8, loc="lower left"); fig.tight_layout()
    fig.savefig(C.FIG_DIR / "pr_curves.png"); plt.close(fig)

    # 4. cost vs threshold
    grid = np.linspace(float(min(s_va.min(), s_te.min())), float(max(s_va.max(), s_te.max())), 400)
    cost_curve = [total_cost(y_te, amt_te, (s_te >= t).astype(int), C.COST_FALSE_ALERT) for t in grid]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(grid, [c["missed_fraud_cost"] for c in cost_curve], label="Missed-fraud cost")
    ax.plot(grid, [c["false_alert_cost"] for c in cost_curve], label=f"False-alert cost (${C.COST_FALSE_ALERT:g} each)")
    ax.plot(grid, [c["total_cost"] for c in cost_curve], "k", lw=2, label="Total cost")
    if np.isfinite(thr_cost):
        ax.axvline(thr_cost, color="g", ls="--", label=f"Cost-optimal threshold ({thr_cost:.2f}, chosen on validation)")
    ax.set_ylim(0, 2.0 * cost_none["total_cost"])  # zoom: flagging everything at threshold ~0 costs far more
    ax.axhline(cost_none["total_cost"], color="gray", ls=":", lw=1, label="No model (all fraud missed)")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Cost on test set ($)")
    ax.set_title("The cost trade-off: missed fraud vs. false alerts"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "cost_tradeoff.png"); plt.close(fig)

    # 5. cost sensitivity
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(sens.cost_per_false_alert, sens.recall, "o-", label="Recall")
    ax.plot(sens.cost_per_false_alert, sens.precision, "s-", label="Precision")
    ax.set_xscale("log"); ax.set_xlabel("Cost of one false alert ($, log scale)")
    ax.set_title("Optimal operating point shifts as false alerts get pricier"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "cost_sensitivity.png"); plt.close(fig)

    # 6. feature importance
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(top.feature, top.importance, xerr=top["std"], color="#2e86ab")
    ax.set_xlabel("Drop in validation PR-AUC when feature is shuffled")
    ax.set_title("Permutation importance (top 15)"); fig.tight_layout()
    fig.savefig(C.FIG_DIR / "feature_importance.png"); plt.close(fig)

    # --------------------------------------------------------------- save artifact
    joblib.dump({
        "model": final_model, "scaler": scaler, "feature_columns": FEATURE_COLUMNS,
        "model_name": key[0], "strategy": key[1],
        "thresholds": {"max_f1": thr_f1, "precision_90": thr_p90, "cost_optimal": thr_cost},
        "cost_false_alert": C.COST_FALSE_ALERT,
    }, C.MODEL_PATH, compress=3)

    # sample for the demo app: contiguous window of the TEST period containing the most frauds
    win = 4000
    csum = np.concatenate([[0], np.cumsum(y_te)])
    start = int(np.argmax(csum[win:] - csum[:-win]))
    sample = test.iloc[start:start + win][["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]]
    (C.ROOT / "sample_data").mkdir(exist_ok=True)
    sample.to_csv(C.ROOT / "sample_data" / "sample_transactions.csv", index=False)

    summary = {
        "dataset": {"rows": len(raw), "frauds": int(raw.Class.sum()), "fraud_rate": float(raw.Class.mean()),
                    "split_rows": [len(train), len(val), len(test)],
                    "split_frauds": [int(y_tr.sum()), int(y_va.sum()), int(y_te.sum())]},
        "final_model": {"model": key[0], "strategy": key[1], "val_pr_auc": float(best_row.val_pr_auc),
                        "test_pr_auc": float(best_row.test_pr_auc), "test_roc_auc": float(best_row.test_roc_auc),
                        "test_pr_auc_ci95": ci["pr_auc"]},
        "always_legit_accuracy": always_legit_acc,
        "operating_points_test": {
            "max_f1": {"threshold": thr_f1, **metrics_at_threshold(y_te, s_te, thr_f1)},
            "precision_90": {"threshold": thr_p90, **m_final, "precision_ci95": ci["precision"], "recall_ci95": ci["recall"]},
            "cost_optimal": {"threshold": None if not np.isfinite(thr_cost) else thr_cost, **cm_cost},
        },
        "cost_analysis_test": {
            "cost_per_false_alert": C.COST_FALSE_ALERT, "total_fraud_amount_in_test": fraud_total_te,
            "no_model": cost_none, "flag_everything": cost_all, "cost_optimal_model": cost_model,
            "savings_vs_no_model": cost_none["total_cost"] - cost_model["total_cost"],
        },
        "top_features": imp.head(10)[["feature", "importance"]].to_dict("records"),
    }
    (C.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log("Done. See results/ and models/.")


if __name__ == "__main__":
    main()
