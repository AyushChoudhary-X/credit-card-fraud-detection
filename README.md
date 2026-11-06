# Credit Card Fraud Detection

An end-to-end machine-learning pipeline that detects fraudulent credit card transactions without drowning
genuine customers in false alarms. Built on the [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, only 492 frauds = 0.17%).

## Headline results (held-out, chronologically *later* test period)

| | |
|---|---|
| Final model | Random Forest trained on SMOTE-resampled data |
| PR-AUC | **0.81** (95% bootstrap CI 0.71 - 0.88) |
| At the validation-selected threshold | **Recall 0.75** (56 of 75 frauds caught), **precision 0.89**, F1 0.81, 7 false alerts in 56,962 transactions |
| Cost-optimal threshold (assuming $5 per false alert) | Total cost **$2,151 vs $7,729** with no model (-72%), recall 0.83, precision 0.44 |
| "Always predict legit" accuracy | 99.87% while catching **0%** of fraud - why accuracy is the wrong metric |

> The test set contains only 75 frauds, so every number above is noisy. The confidence intervals are the honest
> way to read them. The cost result also depends on the assumed $5 false-alert cost (see sensitivity analysis).

## What this project does

1. **Chronological split** (60% train / 20% validation / 20% test by time) so the model is always judged on
   transactions that happen *after* its training data. Thresholds and the final model are chosen on validation;
   the test set is only touched for reporting.
2. **Imbalance handling**, all applied to the training set only: nothing, cost-sensitive class weights,
   random undersampling, and SMOTE (implemented from scratch in `src/resampling.py`, then majority undersampling).
3. **Feature engineering** using only *past* transactions (unit-tested for leakage): amount vs. a rolling average of the
   previous 100 transactions (ratio and z-score), time-of-day (cyclic), transaction velocity (counts in the previous
   60 s / 600 s) and time since the previous transaction.
4. **Model comparison**: Logistic Regression, Random Forest and Isolation Forest (unsupervised anomaly detection).
5. **Evaluation** with Precision, Recall, F1 and PR-AUC, plus bootstrap confidence intervals.
6. **Cost analysis**: a missed fraud costs the transaction amount, a false alert costs a fixed amount. The decision
   threshold is optimised for total cost, and a sensitivity analysis shows how the best operating point moves.
7. **Extras**: permutation-importance explainability, a batch scoring CLI (`src/predict.py`) and a Streamlit demo (`app.py`).

## Model comparison (test set, PR-AUC)

| Model | none | class weights | undersample | SMOTE |
|---|---|---|---|---|
| Logistic Regression | 0.695 | 0.785 | 0.748 | 0.789 |
| Random Forest | 0.793 | 0.803 | 0.789 | **0.805** |
| Isolation Forest (unsupervised) | 0.035 | | | |

Full table (incl. precision/recall at each model's validation-selected thresholds): `results/model_comparison.csv`.

**How to read this honestly**
- Handling imbalance clearly helps Logistic Regression (0.70 -> ~0.79). For Random Forest the four strategies are within
  about 0.02 PR-AUC of each other, far inside the +/-0.08 bootstrap interval, so SMOTE "winning" is not statistically meaningful.
  It was chosen because it had the highest *validation* PR-AUC, by a tiny margin.
- Isolation Forest performs poorly here (PR-AUC 0.035, barely above the 0.0013 random baseline). It is unsupervised, so
  it flags anything unusual, and many legitimate transactions are unusual. Labels are worth a lot on this problem.
- Fraud is concentrated in a few anonymised components: `V14`, `V4`, `V12` are the most important features
  (`results/feature_importance.csv`). Because V1-V28 are PCA outputs, they cannot be interpreted in business terms.

## The cost trade-off

Assumption (editable in `src/config.py`): a missed fraud costs the transaction amount; a false alert costs **$5**
(analyst review plus customer friction).

- With no model, the test period loses **$7,729** to fraud. Flagging every transaction would cost $284,435 in false alerts.
- The cost-optimal threshold (0.33, chosen on validation) catches 62 of 75 frauds with 79 false alerts, for a total of **$2,151**.
- Most frauds in this dataset are small (median about $9), so when a false alert is cheap the model should flag aggressively,
  and when it is expensive it should be conservative. `results/cost_sensitivity.csv` shows the operating point moving from
  recall 0.83 / precision 0.44 (false alert <= $10) to recall 0.65 / precision 0.94 (false alert = $250).

Figures are in `results/figures/`.

## Run it

```bash
pip install -r requirements.txt
# download creditcard.csv from Kaggle and put it in data/creditcard.csv
python -m src.train                                   # ~5 min on one CPU; writes results/ and models/
python -m unittest discover -s tests -v               # 9 unit tests
python -m src.predict sample_data/sample_transactions.csv --threshold cost_optimal
streamlit run app.py                                  # demo app
```

## Project structure

```
src/config.py       paths, split sizes, cost assumptions
src/data.py         loading + chronological split
src/features.py     leakage-free feature engineering
src/resampling.py   random undersampling + SMOTE from scratch
src/models.py       model factories
src/evaluate.py     metrics, thresholds, cost model, bootstrap CIs
src/train.py        full pipeline: train, compare, cost analysis, figures, save model
src/predict.py      batch scoring (CLI + function used by the app)
app.py              Streamlit demo
tests/              unit tests (leakage, SMOTE, cost logic)
results/            metrics tables, summary.json, figures
sample_data/        4,000 consecutive held-out test transactions (14 frauds) for the demo
```

## Limitations

- **No customer IDs.** The dataset is anonymised, so "amount vs. this customer's average" and per-customer velocity are
  approximated with global rolling statistics. Real systems would build these per card or per account.
- **Time-of-day is approximate.** `Time` is seconds since the first transaction; the hour is derived assuming the capture began near midnight.
- **Small test set (75 frauds).** Differences between strategies are within noise. A real evaluation would use more data or repeated time-based folds.
- **Two days of data from one set of European cardholders (September 2013).** Fraud patterns drift; this is a methodology demo, not a production system.
- **Cost figures depend on assumed costs**, not measured ones.
- The Streamlit app was written but not run in the sandbox where this was built, so test it locally.

## Data

Dal Pozzolo, A. et al. (2015), *Calibrating Probability with Undersampling for Unbalanced Classification*. Dataset
released by the Machine Learning Group, ULB, under the Open Database License. `creditcard.csv` is not included in
this repo; `sample_data/` holds a small slice for the demo only.
