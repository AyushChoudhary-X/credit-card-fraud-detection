"""Central configuration for the fraud detection pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"

SEED = 42

# Chronological split: first 60% of time = train, next 20% = validation, last 20% = test.
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

# Feature engineering
ROLLING_WINDOW = 100          # number of previous transactions used for the "recent average"
VELOCITY_WINDOWS = (60, 600)  # seconds, for transaction-velocity counts

# Business assumptions (change these to match your own scenario)
COST_FALSE_ALERT = 5.0        # cost of one false alert: analyst review + customer friction
TARGET_PRECISION = 0.90       # operating point for the "recall at 90% precision" metric

# Resampling
UNDERSAMPLE_RATIO = 0.10      # fraud : legit after undersampling
SMOTE_RATIO = 0.10            # fraud : legit after SMOTE oversampling
SMOTE_THEN_UNDER_RATIO = 0.50 # fraud : legit after SMOTE + undersampling the majority
