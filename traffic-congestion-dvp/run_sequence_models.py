"""
run_sequence_models.py — Train and evaluate LSTM + Transformer models.

Uses the already-processed train.csv from run_pipeline.py.
Adds results to the existing improvement_steps.csv.
"""

import time, os, json
import numpy as np
import pandas as pd

from src.preprocessing import create_sequences
from src.train import FEATURE_COLS, cross_validate_model
from src.models import SequenceModelRegressor

RESULTS_DIR = "results"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

SEQ_LEN = 8  # 8 × 15 min = 2 hours lookback
N_SPLITS = 5

# =====================================================================
# LOAD DATA & CREATE SEQUENCES
# =====================================================================
print("=" * 70)
print("Loading data and creating sequences ...")
print("=" * 70)

train_df = pd.read_csv("data/processed/train.csv", parse_dates=["time_start"])
X_seq, y_seq = create_sequences(train_df, seq_len=SEQ_LEN, feature_cols=FEATURE_COLS)

# =====================================================================
# LSTM
# =====================================================================
print("\n" + "=" * 70)
print("LSTM (2-layer, hidden=128, GPU)")
print("=" * 70)
t0 = time.time()

lstm_model = SequenceModelRegressor(
    model_type="lstm",
    hidden_dim=128,
    n_layers=2,
    dropout=0.2,
    lr=1e-3,
    batch_size=1024,
    epochs=100,
    patience=12,
    device="cuda",
)
cv_lstm = cross_validate_model(lstm_model, X_seq, y_seq, n_splits=N_SPLITS)
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# TRANSFORMER
# =====================================================================
print("\n" + "=" * 70)
print("Transformer (2-layer, d_model=64, 4 heads, GPU)")
print("=" * 70)
t0 = time.time()

tf_model = SequenceModelRegressor(
    model_type="transformer",
    hidden_dim=64,
    n_heads=4,
    n_layers=2,
    dropout=0.1,
    lr=1e-3,
    batch_size=1024,
    epochs=100,
    patience=12,
    device="cuda",
)
cv_tf = cross_validate_model(tf_model, X_seq, y_seq, n_splits=N_SPLITS)
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# RESULTS
# =====================================================================
print("\n" + "=" * 70)
print("SEQUENCE MODEL RESULTS")
print("=" * 70)

seq_results = pd.DataFrame([
    {"model": "LSTM (2-layer)",
     **{k: cv_lstm[k] for k in ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}},
    {"model": "Transformer (2-layer)",
     **{k: cv_tf[k] for k in ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}},
])
print(seq_results[["model","mae_mean","mae_std","rmse_mean","r2_mean"]].to_string(index=False))
seq_results.to_csv(os.path.join(TABLES_DIR, "sequence_model_results.csv"), index=False)

# Append to improvement_steps if it exists
steps_file = os.path.join(TABLES_DIR, "improvement_steps.csv")
if os.path.exists(steps_file):
    steps_df = pd.read_csv(steps_file)
    next_step = steps_df["step"].max() + 1
    for i, row in seq_results.iterrows():
        new_row = {"step": next_step + i, "description": row["model"],
                   "model": row["model"], **{k: row[k] for k in
                   ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}}
        steps_df = pd.concat([steps_df, pd.DataFrame([new_row])], ignore_index=True)
    steps_df.to_csv(steps_file, index=False)
    print(f"\nUpdated {steps_file}")
    print(steps_df[["step","model","mae_mean","rmse_mean","r2_mean"]].to_string(index=False))

print("\nDone!")
