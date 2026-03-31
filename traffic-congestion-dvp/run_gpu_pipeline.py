"""
run_gpu_pipeline.py — CUDA-only pipeline.

All models run on GPU:
  - XGBoost (device=cuda)
  - Embedding Neural Net (PyTorch CUDA)
  - LSTM (PyTorch CUDA)
  - Transformer (PyTorch CUDA)
"""

import sys
import time, os, json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from src.data_loader import (
    load_traffic_data, filter_uoft_area, compute_average_speed, load_weather_data,
)
from src.preprocessing import (
    clean_traffic_data, clean_weather_data, merge_traffic_weather,
    engineer_features, create_congestion_labels, temporal_train_test_split,
    create_sequences, add_lag_features,
)
from src.train import (
    FEATURE_COLS, FEATURE_GROUPS, TARGET_COL, BASELINE_GROUP_COLS,
    encode_location, encode_direction,
    cross_validate_model, tune_model, feature_ablation,
    XGB_PARAM_DISTRIBUTIONS, save_model,
)
from src.models import (
    get_baseline_model, get_xgboost,
    EmbeddingNeuralNet, SequenceModelRegressor,
)
from src.evaluate import (
    plot_cv_results, plot_ablation_results, plot_improvement_steps, save_figure,
)

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
PROCESSED_DIR = os.path.join("data", "processed")
for d in [FIGURES_DIR, TABLES_DIR, MODELS_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

N_SPLITS = 5
TUNE_N_ITER = 50
SEQ_LEN = 8

# =====================================================================
# DATA
# =====================================================================
print("=" * 70)
print("DATA LOADING & PREPROCESSING")
print("=" * 70)
t0 = time.time()

traffic_df = load_traffic_data([
    "data/raw/svc_raw_data_speed_2020_2024.csv",
    "data/raw/svc_raw_data_speed_2025_2029.csv",
])
traffic_df = filter_uoft_area(traffic_df)
traffic_df = compute_average_speed(traffic_df)
traffic_df = clean_traffic_data(traffic_df)

weather_df = load_weather_data("data/raw/weather")
weather_df = clean_weather_data(weather_df)

merged_df = merge_traffic_weather(traffic_df, weather_df)
featured_df = engineer_features(merged_df)
featured_df = create_congestion_labels(featured_df)

featured_df, loc_enc = encode_location(featured_df)
featured_df, dir_enc = encode_direction(featured_df)
n_locations = featured_df["location_encoded"].nunique()
n_directions = featured_df["direction_encoded"].nunique()

# Add lag features (speed_lag_1, speed_lag_4, speed_lag_8, speed_rolling_4)
featured_df = add_lag_features(featured_df)

# Drop rows where no lag is available (start of each location's series)
lag_cols = ["speed_lag_1", "speed_lag_4", "speed_lag_8", "speed_rolling_4"]
before = len(featured_df)
featured_df = featured_df.dropna(subset=["speed_lag_1"]).reset_index(drop=True)
print(f"  Dropped {before - len(featured_df):,} rows without lag features")

imputer = SimpleImputer(strategy="median")
num_cols = [c for c in FEATURE_COLS if c in featured_df.columns]
featured_df[num_cols] = imputer.fit_transform(featured_df[num_cols])

train_df, test_df = temporal_train_test_split(featured_df)
train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

y_train = train_df[TARGET_COL].values
X_train = train_df[FEATURE_COLS]

print(f"\nData ready: {len(train_df):,} train, {len(test_df):,} test ({time.time()-t0:.1f}s)")

# Sequences for LSTM/Transformer
print("\nCreating sequences ...")
X_seq, y_seq = create_sequences(train_df, seq_len=SEQ_LEN, feature_cols=FEATURE_COLS)

steps = []

# =====================================================================
# STEP 0: Baseline
# =====================================================================
print("\n" + "=" * 70)
print("STEP 0: Historical Average Baseline")
print("=" * 70)
t0 = time.time()
baseline_cols = BASELINE_GROUP_COLS + [c for c in FEATURE_COLS if c not in BASELINE_GROUP_COLS]
cv0 = cross_validate_model(get_baseline_model(), train_df[baseline_cols], y_train, N_SPLITS)
steps.append({"step": 0, "description": "Historical Avg", "model": "Historical Average",
              **{k: cv0[k] for k in ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 1: XGBoost default — CUDA
# =====================================================================
print("\n" + "=" * 70)
print("STEP 1: XGBoost default — CUDA")
print("=" * 70)
t0 = time.time()
cv1 = cross_validate_model(get_xgboost(device="cuda"), X_train, y_train, N_SPLITS)
steps.append({"step": 1, "description": "XGBoost\n(default)", "model": "XGBoost (default)",
              **{k: cv1[k] for k in ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 2: XGBoost tuned — CUDA
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: Tune XGBoost — CUDA (n_iter=50)")
print("=" * 70)
t0 = time.time()
xgb_tune = tune_model(
    get_xgboost(device="cuda"), XGB_PARAM_DISTRIBUTIONS,
    X_train, y_train, n_iter=TUNE_N_ITER, n_splits=N_SPLITS, n_jobs=1,
)
cv2 = cross_validate_model(xgb_tune["best_estimator"], X_train, y_train, N_SPLITS)
steps.append({"step": 2, "description": "XGBoost\n(tuned)", "model": "XGBoost (tuned)",
              **{k: cv2[k] for k in ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 3: Embedding Neural Net — PyTorch CUDA
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: Embedding Neural Net — PyTorch CUDA")
print("=" * 70)
t0 = time.time()
nn_model = EmbeddingNeuralNet(
    n_locations=n_locations + 50, n_directions=n_directions + 5,
    loc_embed_dim=32, dir_embed_dim=8,
    hidden_dims=(256, 128, 64), dropout=0.3,
    lr=1e-3, batch_size=1024, epochs=150, patience=15, device="cuda",
)
cv3 = cross_validate_model(nn_model, X_train, y_train, N_SPLITS)
steps.append({"step": 3, "description": "Embedding NN", "model": "Embedding NN",
              **{k: cv3[k] for k in ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 4: LSTM — PyTorch CUDA
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: LSTM — PyTorch CUDA")
print("=" * 70)
t0 = time.time()
lstm = SequenceModelRegressor(
    model_type="lstm", hidden_dim=128, n_layers=2, dropout=0.2,
    lr=1e-3, batch_size=1024, epochs=100, patience=12, device="cuda",
)
cv4 = cross_validate_model(lstm, X_seq, y_seq, N_SPLITS)
steps.append({"step": 4, "description": "LSTM", "model": "LSTM (2-layer)",
              **{k: cv4[k] for k in ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 5: Transformer — PyTorch CUDA
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: Transformer — PyTorch CUDA")
print("=" * 70)
t0 = time.time()
transformer = SequenceModelRegressor(
    model_type="transformer", hidden_dim=64, n_heads=4, n_layers=2,
    dropout=0.1, lr=1e-3, batch_size=1024, epochs=100, patience=12, device="cuda",
)
cv5 = cross_validate_model(transformer, X_seq, y_seq, N_SPLITS)
steps.append({"step": 5, "description": "Transformer", "model": "Transformer (2-layer)",
              **{k: cv5[k] for k in ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 6: Feature ablation on tuned XGBoost
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: Feature ablation (XGBoost tuned)")
print("=" * 70)
t0 = time.time()
ablation_df = feature_ablation(
    clone(xgb_tune["best_estimator"]),
    X_train, y_train, feature_groups=FEATURE_GROUPS, n_splits=N_SPLITS,
)
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# SAVE
# =====================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

steps_df = pd.DataFrame(steps)
steps_df.to_csv(os.path.join(TABLES_DIR, "improvement_steps.csv"), index=False)
ablation_df.to_csv(os.path.join(TABLES_DIR, "feature_ablation.csv"), index=False)
xgb_tune["cv_results"].to_csv(os.path.join(TABLES_DIR, "xgb_tuning_cv_results.csv"), index=False)

with open(os.path.join(TABLES_DIR, "best_params.json"), "w") as f:
    json.dump({"xgboost": {k: (int(v) if isinstance(v, np.integer) else v)
                            for k, v in xgb_tune["best_params"].items()}}, f, indent=2)

save_model(xgb_tune["best_estimator"], os.path.join(MODELS_DIR, "xgb_tuned.joblib"))
save_model(loc_enc, os.path.join(MODELS_DIR, "label_encoder.joblib"))
save_model(dir_enc, os.path.join(MODELS_DIR, "direction_encoder.joblib"))
save_model(imputer, os.path.join(MODELS_DIR, "imputer.joblib"))

print(f"\n  Results:")
print(steps_df[["step","model","mae_mean","mae_std","rmse_mean","r2_mean"]].to_string(index=False))

# =====================================================================
# PLOTS
# =====================================================================
print("\nGenerating plots ...")

all_cv = {
    "Hist. Avg": cv0, "XGB (default)": cv1, "XGB (tuned)": cv2,
    "Embed. NN": cv3, "LSTM": cv4, "Transformer": cv5,
}
fig = plot_cv_results(all_cv)
save_figure(fig, os.path.join(FIGURES_DIR, "model_comparison_cv.png"))

fig = plot_improvement_steps(steps_df[["step","description","mae_mean","mae_std"]])
save_figure(fig, os.path.join(FIGURES_DIR, "improvement_steps.png"))

fig = plot_ablation_results(ablation_df)
save_figure(fig, os.path.join(FIGURES_DIR, "feature_ablation.png"))

print("\n" + "=" * 70)
print("PIPELINE COMPLETE — ALL MODELS ON CUDA")
print("=" * 70)
