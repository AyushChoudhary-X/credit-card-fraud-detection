"""Streamlit demo: upload transactions, see which ones get flagged, and explore the cost trade-off.

Run:  streamlit run app.py
Requires models/fraud_model.joblib (create it with:  python -m src.train)
"""
import numpy as np
import pandas as pd
import streamlit as st

from src.config import COST_FALSE_ALERT, MODEL_PATH, ROOT
from src.evaluate import cost_optimal_threshold, total_cost
from src.predict import load_artifact, score_transactions

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("Credit Card Fraud Detection")


@st.cache_resource
def get_artifact():
    return load_artifact()


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

art = get_artifact()
st.caption(f"Model: {art['model_name'].replace('_', ' ')} trained with {art['strategy']} "
           "(the model never saw the sample file below: it comes from the held-out test period).")

source = st.radio("Transactions", ["Sample (held-out test period)", "Upload CSV"], horizontal=True)
if source.startswith("Sample"):
    raw = pd.read_csv(ROOT / "sample_data" / "sample_transactions.csv")
else:
    up = st.file_uploader("CSV with columns Time, V1..V28, Amount (Class optional)", type="csv")
    if up is None:
        st.stop()
    raw = pd.read_csv(up)

# ---- operating point -------------------------------------------------------------------------
st.sidebar.header("Operating point")
cost_fp = st.sidebar.number_input("Cost of one false alert ($)", 0.5, 1000.0, float(COST_FALSE_ALERT), 0.5)
scored = score_transactions(raw, art)
has_labels = "Class" in scored

mode = st.sidebar.selectbox("Threshold", ["Cost-optimal for the cost above", "Saved: max F1", "Saved: precision >= 90%", "Manual"])
if mode.startswith("Cost-optimal"):
    if has_labels:
        thr = cost_optimal_threshold(scored["Class"].to_numpy(), scored["Amount"].to_numpy(),
                                     scored["fraud_score"].to_numpy(), cost_fp)
        if not np.isfinite(thr):
            thr = 1.01
        st.sidebar.caption("Optimised on the uploaded labels, so this is an in-sample illustration only.")
    else:
        thr = art["thresholds"]["cost_optimal"]
        st.sidebar.caption("No labels in the upload, so the saved threshold (tuned for $5) is used.")
elif mode == "Saved: max F1":
    thr = art["thresholds"]["max_f1"]
elif mode == "Saved: precision >= 90%":
    thr = art["thresholds"]["precision_90"]
else:
    thr = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
scored["flagged"] = (scored["fraud_score"] >= thr).astype(int)

# ---- results ----------------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Transactions", f"{len(scored):,}")
c2.metric("Flagged", int(scored.flagged.sum()))
c3.metric("Threshold", f"{thr:.2f}")

if has_labels:
    y, pred, amt = scored["Class"].to_numpy(), scored["flagged"].to_numpy(), scored["Amount"].to_numpy()
    tp, fp = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    cost = total_cost(y, amt, pred, cost_fp)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Frauds caught", f"{tp} / {tp + fn}")
    d2.metric("False alerts", fp)
    d3.metric("Total cost", f"${cost['total_cost']:,.0f}")
    d4.metric("Cost with no model", f"${amt[y == 1].sum():,.0f}")

st.subheader("Most suspicious transactions")
st.dataframe(scored.sort_values("fraud_score", ascending=False).head(25), use_container_width=True)
st.subheader("Fraud score distribution")
st.bar_chart(np.histogram(scored["fraud_score"], bins=20, range=(0, 1))[0])
