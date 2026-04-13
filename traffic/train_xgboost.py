"""
ECE1513 Project — XGBoost Regression Model
Target: avg_speed_kph (raw speed)
Split: 80/10/10 chronological
Historical averages computed from training data only (no leakage).
"""
import pandas as pd
import numpy as np
import joblib
import time
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import MERGED_DATA, MODELS_DIR, RESULTS_DIR

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(MERGED_DATA)
df["time_start"] = pd.to_datetime(df["time_start"])
df["time_end"] = pd.to_datetime(df["time_end"])
df = df.sort_values(["centreline_id", "direction", "time_start"]).reset_index(drop=True)

print(f"Total records: {len(df)}")
print(f"Date range: {df['time_start'].min()} to {df['time_start'].max()}")

# ── 2. Time Features ─────────────────────────────────────────
df["hour"] = df["time_start"].dt.hour
df["minute"] = df["time_start"].dt.minute
df["dayofweek"] = df["time_start"].dt.dayofweek
df["month"] = df["time_start"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

# ── 3. Lag & Rolling Features (shift-based, safe) ────────────
group_keys = ["centreline_id", "direction"]

for lag in [1, 2, 4, 8]:
    df[f"speed_lag_{lag}"] = df.groupby(group_keys)["avg_speed_kph"].shift(lag)
    df[f"volume_lag_{lag}"] = df.groupby(group_keys)["volume_15min"].shift(lag)

for window in [4, 8]:
    grp = df.groupby(group_keys)
    df[f"speed_roll_mean_{window}"] = grp["avg_speed_kph"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df[f"speed_roll_std_{window}"] = grp["avg_speed_kph"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )
    df[f"volume_roll_mean_{window}"] = grp["volume_15min"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# ── 4. Time-Aware Split (80/10/10) ───────────────────────────
sorted_df = df.sort_values("time_start").reset_index(drop=True)
n = len(sorted_df)

train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = sorted_df.iloc[:train_end].copy()
val_df = sorted_df.iloc[train_end:val_end].copy()
test_df = sorted_df.iloc[val_end:].copy()

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train period: {train_df['time_start'].min()} to {train_df['time_start'].max()}")
print(f"Val period:   {val_df['time_start'].min()} to {val_df['time_start'].max()}")
print(f"Test period:  {test_df['time_start'].min()} to {test_df['time_start'].max()}")

# ── 5. Historical Averages (train-only, no leakage) ──────────
hist_keys = ["centreline_id", "direction", "hour", "dayofweek"]

train_hist = (
    train_df.groupby(hist_keys)
    .agg(hist_avg_speed=("avg_speed_kph", "mean"),
         hist_avg_volume=("volume_15min", "mean"))
    .reset_index()
)

# Segment-level fallback for unseen combos
train_seg_avg = (
    train_df.groupby(["centreline_id", "direction"])
    .agg(fallback_speed=("avg_speed_kph", "mean"),
         fallback_volume=("volume_15min", "mean"))
    .reset_index()
)
fallback_speed_map = train_seg_avg.set_index(["centreline_id", "direction"])["fallback_speed"]
fallback_volume_map = train_seg_avg.set_index(["centreline_id", "direction"])["fallback_volume"]

for split_df in [train_df, val_df, test_df]:
    split_df.drop(columns=["hist_avg_speed", "hist_avg_volume"], errors="ignore", inplace=True)
    merged = split_df.merge(train_hist, on=hist_keys, how="left")
    split_df["hist_avg_speed"] = merged["hist_avg_speed"].values
    split_df["hist_avg_volume"] = merged["hist_avg_volume"].values

# Fill missing with segment-level fallback
for split_df in [val_df, test_df]:
    for col, fmap in [("hist_avg_speed", fallback_speed_map), ("hist_avg_volume", fallback_volume_map)]:
        mask = split_df[col].isna()
        if mask.any():
            keys = list(zip(split_df.loc[mask, "centreline_id"], split_df.loc[mask, "direction"]))
            split_df.loc[mask, col] = [fmap.get(k, np.nan) for k in keys]

# Final fallback: global train mean
global_speed = train_df["avg_speed_kph"].mean()
global_volume = train_df["volume_15min"].mean()
for split_df in [train_df, val_df, test_df]:
    split_df["hist_avg_speed"].fillna(global_speed, inplace=True)
    split_df["hist_avg_volume"].fillna(global_volume, inplace=True)

print(f"Hist avg NaNs in test: speed={test_df['hist_avg_speed'].isna().sum()}, volume={test_df['hist_avg_volume'].isna().sum()}")

# ── 6. Deviation Features (using train-only avg) ────────────
for split_df in [train_df, val_df, test_df]:
    split_df["speed_deviation"] = split_df["avg_speed_kph"] - split_df["hist_avg_speed"]
    split_df["volume_deviation"] = split_df["volume_15min"] - split_df["hist_avg_volume"]

# Deviation lags: concat, compute, re-split (for correct boundary lags)
combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
combined = combined.sort_values(["centreline_id", "direction", "time_start"]).reset_index(drop=True)

for lag in [1, 2, 4, 8]:
    combined[f"speed_dev_lag_{lag}"] = combined.groupby(group_keys)["speed_deviation"].shift(lag)
    combined[f"volume_dev_lag_{lag}"] = combined.groupby(group_keys)["volume_deviation"].shift(lag)

for window in [4, 8]:
    grp = combined.groupby(group_keys)
    combined[f"speed_dev_roll_mean_{window}"] = grp["speed_deviation"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    combined[f"speed_dev_roll_std_{window}"] = grp["speed_deviation"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )
    combined[f"volume_dev_roll_mean_{window}"] = grp["volume_deviation"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# Re-split using original time boundaries
train_max_t = sorted_df.iloc[train_end - 1]["time_start"]
val_max_t = sorted_df.iloc[val_end - 1]["time_start"]

combined = combined.sort_values("time_start").reset_index(drop=True)
train_df = combined[combined["time_start"] <= train_max_t].copy()
val_df = combined[(combined["time_start"] > train_max_t) & (combined["time_start"] <= val_max_t)].copy()
test_df = combined[combined["time_start"] > val_max_t].copy()

print(f"\nAfter deviation features: Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")

# ── 7. Define Features & Target ──────────────────────────────
TARGET = "avg_speed_kph"

FEATURES = [
    # Location
    "longitude", "latitude", "centreline_id",
    # Time
    "hour", "dayofweek", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Weather
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)",
    "Wind_Sin", "Wind_Cos",
    "is_raining", "is_snowing", "is_foggy",
    # Historical context (train-only)
    "hist_avg_speed", "hist_avg_volume",
    # Raw speed lags
    "speed_lag_1", "speed_lag_2", "speed_lag_4", "speed_lag_8",
    # Raw volume lags
    "volume_lag_1", "volume_lag_2", "volume_lag_4", "volume_lag_8",
    # Raw rolling stats
    "speed_roll_mean_4", "speed_roll_mean_8",
    "speed_roll_std_4", "speed_roll_std_8",
    "volume_roll_mean_4", "volume_roll_mean_8",
    # Deviation lags (train-only avg)
    "speed_dev_lag_1", "speed_dev_lag_2", "speed_dev_lag_4", "speed_dev_lag_8",
    "volume_dev_lag_1", "volume_dev_lag_2", "volume_dev_lag_4", "volume_dev_lag_8",
    # Deviation rolling stats
    "speed_dev_roll_mean_4", "speed_dev_roll_mean_8",
    "speed_dev_roll_std_4", "speed_dev_roll_std_8",
    "volume_dev_roll_mean_4", "volume_dev_roll_mean_8",
    # Current volume
    "volume_15min",
]

# Drop NaN rows
for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    before = len(split_df)
    mask = split_df[FEATURES + [TARGET]].notna().all(axis=1)
    if name == "train":
        train_df = split_df[mask].copy()
    elif name == "val":
        val_df = split_df[mask].copy()
    else:
        test_df = split_df[mask].copy()
    print(f"  {name}: {before} -> {mask.sum()} (dropped {before - mask.sum()} NaN rows)")

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_val = val_df[FEATURES]
y_val = val_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# ── 8. Train XGBoost ─────────────────────────────────────────
print("\nTraining XGBoost...")
t0 = time.time()

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    device="cuda",
    early_stopping_rounds=30,
    eval_metric="rmse",
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

train_time = time.time() - t0
print(f"\nBest iteration: {xgb_model.best_iteration}")
print(f"Training time: {train_time:.1f}s")

# ── 9. Evaluate ──────────────────────────────────────────────
y_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== XGBoost Model Performance =====")
print(f"MAE  : {mae:.4f} km/h")
print(f"RMSE : {rmse:.4f} km/h")
print(f"R²   : {r2:.4f}")
print(f"Time : {train_time:.1f}s")

# ── 10. Save Model & Predictions ─────────────────────────────
model_path = os.path.join(MODELS_DIR, "xgb_speed_model.pkl")
joblib.dump(xgb_model, model_path)
print(f"\nSaved model to {model_path}")

preds_df = test_df[["centreline_id", "direction", "time_start", TARGET]].copy()
preds_df["predicted_speed"] = y_pred
preds_path = os.path.join(RESULTS_DIR, "xgb_speed_predictions.csv")
preds_df.to_csv(preds_path, index=False)
print(f"Saved predictions to {preds_path}")

# ── 11. Feature Importance ───────────────────────────────────
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop 20 important features:")
print(importance_df.head(20).to_string())

imp_path = os.path.join(RESULTS_DIR, "xgb_speed_feature_importance.csv")
importance_df.to_csv(imp_path, index=False)
print(f"\nSaved feature importance to {imp_path}")
