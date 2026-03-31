"""
run_pipeline.py — Full incremental improvement pipeline (v2).

Steps:
  0  Historical average baseline
  1  XGBoost default (basic features, no spatial/COVID/calendar)
  2  + COVID + U of T calendar features
  3  + Spatial features (lat/lon, direction)
  4  LightGBM DART (same features)
  5  Neural net with entity embeddings (location + direction)
  6  Hyperparameter tuning (best tree model)
  7  Feature ablation on best model
"""

import time, os, json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.data_loader import (
    load_traffic_data, filter_uoft_area, compute_average_speed, load_weather_data,
)
from src.preprocessing import (
    clean_traffic_data, clean_weather_data, merge_traffic_weather,
    engineer_features, create_congestion_labels, temporal_train_test_split,
)
from src.train import (
    FEATURE_COLS, FEATURE_GROUPS, TARGET_COL, BASELINE_GROUP_COLS,
    EMBEDDING_COLS, CONTINUOUS_COLS,
    encode_location, encode_direction,
    cross_validate_model, tune_model, feature_ablation,
    RF_PARAM_DISTRIBUTIONS, XGB_PARAM_DISTRIBUTIONS, save_model,
)
from src.models import (
    get_baseline_model, get_linear_regression, get_random_forest,
    get_xgboost, get_lightgbm_dart, EmbeddingNeuralNet,
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

# =====================================================================
# DATA LOADING & PREPROCESSING
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

# Approximate school_in_session (for step 1 comparison)
featured_df["school_in_session"] = featured_df["month"].apply(
    lambda m: int(m >= 9 or m <= 6)
).astype(int)

# Encode categoricals
featured_df, loc_enc = encode_location(featured_df)
featured_df, dir_enc = encode_direction(featured_df)

n_locations = featured_df["location_encoded"].nunique()
n_directions = featured_df["direction_encoded"].nunique()
print(f"  {n_locations} unique locations, {n_directions} unique directions")

# Impute NaN in numeric features
imputer = SimpleImputer(strategy="median")
all_numeric = [c for c in set(FEATURE_COLS + ["school_in_session"])
               if c in featured_df.columns]
featured_df[all_numeric] = imputer.fit_transform(featured_df[all_numeric])

# Train/test split
train_df, test_df = temporal_train_test_split(featured_df)
train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

y_train = train_df[TARGET_COL].values
print(f"\nData ready: {len(train_df):,} train, {len(test_df):,} test ({time.time()-t0:.1f}s)")

# =====================================================================
# FEATURE SETS FOR EACH STEP
# =====================================================================
# Step 1: basic — temporal + weather + holidays + school_in_session + location_encoded
BASIC_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "temp", "humidity", "visibility", "wind_speed", "is_raining", "is_snowing",
    "is_holiday", "is_long_weekend", "school_in_session", "location_encoded",
]
# Step 2: + COVID + U of T calendar (replace school_in_session)
STEP2_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "temp", "humidity", "visibility", "wind_speed", "is_raining", "is_snowing",
    "is_holiday", "is_long_weekend",
    "uoft_classes", "uoft_exams", "uoft_reading_week",
    "is_covid_lockdown", "is_covid_era",
    "location_encoded",
]
# Step 3+: full features (adds lat/lon/direction)
FULL_FEATURES = FEATURE_COLS

improvement_steps = []

# =====================================================================
# STEP 0: Historical Average Baseline
# =====================================================================
print("\n" + "=" * 70)
print("STEP 0: Historical Average Baseline")
print("=" * 70)
t0 = time.time()
baseline_cols = BASELINE_GROUP_COLS + [
    c for c in BASIC_FEATURES if c not in BASELINE_GROUP_COLS
]
cv0 = cross_validate_model(get_baseline_model(), train_df[baseline_cols], y_train, N_SPLITS)
improvement_steps.append({"step": 0, "description": "Historical Avg\n(baseline)",
                          "model": "Historical Average", **{k: cv0[k] for k in
                          ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 1: XGBoost default (basic features)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 1: XGBoost default (basic features)")
print("=" * 70)
t0 = time.time()
cv1 = cross_validate_model(get_xgboost(), train_df[BASIC_FEATURES], y_train, N_SPLITS)
improvement_steps.append({"step": 1, "description": "XGBoost\n(basic features)",
                          "model": "XGBoost (basic)", **{k: cv1[k] for k in
                          ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 2: + COVID + U of T calendar
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: + COVID + U of T calendar features")
print("=" * 70)
t0 = time.time()
cv2 = cross_validate_model(get_xgboost(), train_df[STEP2_FEATURES], y_train, N_SPLITS)
improvement_steps.append({"step": 2, "description": "XGBoost\n(+COVID +UofT)",
                          "model": "XGBoost (+COVID +UofT)", **{k: cv2[k] for k in
                          ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 3: + Spatial features (lat/lon, direction)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: + Spatial features (lat/lon, direction)")
print("=" * 70)
t0 = time.time()
cv3 = cross_validate_model(get_xgboost(), train_df[FULL_FEATURES], y_train, N_SPLITS)
improvement_steps.append({"step": 3, "description": "XGBoost\n(+spatial)",
                          "model": "XGBoost (+spatial)", **{k: cv3[k] for k in
                          ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 4: LightGBM DART
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: LightGBM DART")
print("=" * 70)
t0 = time.time()
cv4 = cross_validate_model(get_lightgbm_dart(), train_df[FULL_FEATURES], y_train, N_SPLITS)
improvement_steps.append({"step": 4, "description": "LightGBM DART",
                          "model": "LightGBM DART", **{k: cv4[k] for k in
                          ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 5: Neural Net with Entity Embeddings
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: Neural Net with Entity Embeddings (PyTorch, GPU)")
print("=" * 70)
t0 = time.time()
nn_model = EmbeddingNeuralNet(
    n_locations=n_locations + 50,   # headroom for unseen
    n_directions=n_directions + 5,
    loc_embed_dim=32, dir_embed_dim=8,
    hidden_dims=(256, 128, 64), dropout=0.3,
    lr=1e-3, batch_size=1024, epochs=150,
    patience=15, device="cuda",
)
cv5 = cross_validate_model(nn_model, train_df[FULL_FEATURES], y_train, N_SPLITS)
improvement_steps.append({"step": 5, "description": "Neural Net\n(embeddings)",
                          "model": "Embedding NN", **{k: cv5[k] for k in
                          ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# STEP 6: Hyperparameter tuning
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: Hyperparameter tuning")
print("=" * 70)

# Tune Random Forest
print("\n  Tuning Random Forest (n_iter=50) ...")
t0 = time.time()
rf_tune = tune_model(
    get_random_forest(), RF_PARAM_DISTRIBUTIONS,
    train_df[FULL_FEATURES], y_train, n_iter=TUNE_N_ITER, n_splits=N_SPLITS, n_jobs=-1,
)
print(f"  RF tuning done ({time.time()-t0:.1f}s)")

# Tune XGBoost on GPU
print("\n  Tuning XGBoost on GPU (n_iter=50) ...")
t0 = time.time()
xgb_tune = tune_model(
    get_xgboost(device="cuda"), XGB_PARAM_DISTRIBUTIONS,
    train_df[FULL_FEATURES], y_train, n_iter=TUNE_N_ITER, n_splits=N_SPLITS, n_jobs=1,
)
print(f"  XGB tuning done ({time.time()-t0:.1f}s)")

# LightGBM DART tuning
LGBM_PARAM_DISTRIBUTIONS = {
    "n_estimators": [300, 500, 700, 1000],
    "max_depth": [4, 6, 8, 10, -1],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [5, 10, 20, 50],
    "drop_rate": [0.05, 0.1, 0.15, 0.2],
}
print("\n  Tuning LightGBM DART (n_iter=50) ...")
t0 = time.time()
lgbm_tune = tune_model(
    get_lightgbm_dart(), LGBM_PARAM_DISTRIBUTIONS,
    train_df[FULL_FEATURES], y_train, n_iter=TUNE_N_ITER, n_splits=N_SPLITS, n_jobs=-1,
)
print(f"  LGBM tuning done ({time.time()-t0:.1f}s)")

# CV the tuned models
print("\n  Cross-validating tuned models ...")
cv_rf_tuned = cross_validate_model(rf_tune["best_estimator"], train_df[FULL_FEATURES], y_train, N_SPLITS)
cv_xgb_tuned = cross_validate_model(xgb_tune["best_estimator"], train_df[FULL_FEATURES], y_train, N_SPLITS)
cv_lgbm_tuned = cross_validate_model(lgbm_tune["best_estimator"], train_df[FULL_FEATURES], y_train, N_SPLITS)

# Pick best tuned tree model
tuned_candidates = [
    ("RF (tuned)", rf_tune, cv_rf_tuned),
    ("XGBoost (tuned)", xgb_tune, cv_xgb_tuned),
    ("LightGBM (tuned)", lgbm_tune, cv_lgbm_tuned),
]
best_name, best_tune, best_cv = min(tuned_candidates, key=lambda x: x[2]["mae_mean"])
improvement_steps.append({"step": 6, "description": f"{best_name}",
                          "model": best_name, **{k: best_cv[k] for k in
                          ["mae_mean","mae_std","rmse_mean","rmse_std","r2_mean","r2_std"]}})
print(f"\n  Best tuned model: {best_name}")
print(f"  Best params: {best_tune['best_params']}")

# =====================================================================
# STEP 7: Feature ablation
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7: Feature ablation study")
print("=" * 70)
t0 = time.time()
from sklearn.base import clone
ablation_df = feature_ablation(
    clone(best_tune["best_estimator"]),
    train_df[FULL_FEATURES], y_train,
    feature_groups=FEATURE_GROUPS, n_splits=N_SPLITS,
)
print(f"  ({time.time()-t0:.1f}s)")

# =====================================================================
# SAVE RESULTS
# =====================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

steps_df = pd.DataFrame(improvement_steps)
steps_df.to_csv(os.path.join(TABLES_DIR, "improvement_steps.csv"), index=False)
print(f"\n  Improvement steps:")
print(steps_df[["step","model","mae_mean","mae_std","rmse_mean","r2_mean"]].to_string(index=False))

ablation_df.to_csv(os.path.join(TABLES_DIR, "feature_ablation.csv"), index=False)

# Tuning results
rf_tune["cv_results"].to_csv(os.path.join(TABLES_DIR, "rf_tuning_cv_results.csv"), index=False)
xgb_tune["cv_results"].to_csv(os.path.join(TABLES_DIR, "xgb_tuning_cv_results.csv"), index=False)
lgbm_tune["cv_results"].to_csv(os.path.join(TABLES_DIR, "lgbm_tuning_cv_results.csv"), index=False)

with open(os.path.join(TABLES_DIR, "best_params.json"), "w") as f:
    json.dump({
        "random_forest": {k: (int(v) if isinstance(v, np.integer) else v)
                          for k, v in rf_tune["best_params"].items()},
        "xgboost": {k: (int(v) if isinstance(v, np.integer) else v)
                    for k, v in xgb_tune["best_params"].items()},
        "lightgbm": {k: (int(v) if isinstance(v, np.integer) else v)
                     for k, v in lgbm_tune["best_params"].items()},
    }, f, indent=2)

# Save models
save_model(best_tune["best_estimator"], os.path.join(MODELS_DIR, "best_tuned_model.joblib"))
save_model(rf_tune["best_estimator"], os.path.join(MODELS_DIR, "rf_tuned.joblib"))
save_model(xgb_tune["best_estimator"], os.path.join(MODELS_DIR, "xgb_tuned.joblib"))
save_model(lgbm_tune["best_estimator"], os.path.join(MODELS_DIR, "lgbm_tuned.joblib"))
save_model(loc_enc, os.path.join(MODELS_DIR, "label_encoder.joblib"))
save_model(dir_enc, os.path.join(MODELS_DIR, "direction_encoder.joblib"))
save_model(imputer, os.path.join(MODELS_DIR, "imputer.joblib"))

# =====================================================================
# PLOTS
# =====================================================================
print("\nGenerating plots ...")

# All models comparison
all_cv = {
    "Hist. Avg": cv0,
    "XGB (basic)": cv1,
    "XGB (+COVID/cal)": cv2,
    "XGB (+spatial)": cv3,
    "LGBM DART": cv4,
    "Embed. NN": cv5,
    "RF (tuned)": cv_rf_tuned,
    "XGB (tuned)": cv_xgb_tuned,
    "LGBM (tuned)": cv_lgbm_tuned,
}
fig = plot_cv_results(all_cv)
save_figure(fig, os.path.join(FIGURES_DIR, "model_comparison_cv.png"))

plot_steps_df = steps_df[["step", "description", "mae_mean", "mae_std"]].copy()
fig = plot_improvement_steps(plot_steps_df)
save_figure(fig, os.path.join(FIGURES_DIR, "improvement_steps.png"))

fig = plot_ablation_results(ablation_df)
save_figure(fig, os.path.join(FIGURES_DIR, "feature_ablation.png"))

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"Results: {RESULTS_DIR}/")
print(f"Data:    {PROCESSED_DIR}/")
