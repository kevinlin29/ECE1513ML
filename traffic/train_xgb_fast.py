"""
ECE1513 — Fast XGBoost iterations to push MAE toward 2.x
Target: next 15-min avg_speed_kph
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from time import perf_counter

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import MERGED_DATA, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
GROUP_COLS = ["centreline_id", "direction"]

# ── 1. Load & Prepare ────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(MERGED_DATA)
df["time_start"] = pd.to_datetime(df["time_start"])

for col in ["is_raining", "is_snowing", "is_foggy"]:
    if col in df.columns:
        df[col] = df[col].replace({True: 1, False: 0}).fillna(0).astype(int)

for col in ["longitude", "latitude", "volume_15min", "avg_speed_kph",
            "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
            "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(GROUP_COLS + ["time_start"]).reset_index(drop=True)

# Time features
df["hour"] = df["time_start"].dt.hour
df["minute"] = df["time_start"].dt.minute
df["day_of_week"] = df["time_start"].dt.dayofweek
df["month"] = df["time_start"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 9) & (df["is_weekend"] == 0)).astype(int)
df["is_evening_rush"] = ((df["hour"] >= 16) & (df["hour"] <= 18) & (df["is_weekend"] == 0)).astype(int)

# Lags
for lag in [1, 2, 4, 8, 12, 16]:
    df[f"speed_lag_{lag}"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(lag)
    df[f"volume_lag_{lag}"] = df.groupby(GROUP_COLS)["volume_15min"].shift(lag)

for window in [4, 8]:
    grp = df.groupby(GROUP_COLS)
    df[f"speed_roll_mean_{window}"] = grp["avg_speed_kph"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    df[f"speed_roll_std_{window}"] = grp["avg_speed_kph"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std())
    df[f"volume_roll_mean_{window}"] = grp["volume_15min"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())

# Target
df["target_speed"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(-1)
df = df.dropna(subset=["target_speed"]).copy()

# ── 2. Split ────────────────────────────────────────────────
df_sorted = df.sort_values("time_start").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = df_sorted.iloc[:train_end].copy()
val_df = df_sorted.iloc[train_end:val_end].copy()
test_df = df_sorted.iloc[val_end:].copy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

TARGET = "target_speed"

# ── 3. Historical averages (train only) ─────────────────────
print("Computing historical averages...")
hist_key = ["centreline_id", "direction", "hour", "day_of_week"]
hist_speed = train_df.groupby(hist_key)["avg_speed_kph"].mean()
hist_volume = train_df.groupby(hist_key)["volume_15min"].mean()
g_spd = train_df["avg_speed_kph"].mean()
g_vol = train_df["volume_15min"].mean()

for sdf in [train_df, val_df, test_df]:
    keys = list(zip(sdf["centreline_id"], sdf["direction"], sdf["hour"], sdf["day_of_week"]))
    sdf["hist_avg_speed"] = [hist_speed.get(k, g_spd) for k in keys]
    sdf["hist_avg_volume"] = [hist_volume.get(k, g_vol) for k in keys]
    sdf["speed_deviation"] = sdf["avg_speed_kph"] - sdf["hist_avg_speed"]
    sdf["volume_deviation"] = sdf["volume_15min"] - sdf["hist_avg_volume"]

for sdf in [train_df, val_df, test_df]:
    s = sdf.sort_values(GROUP_COLS + ["time_start"])
    for lag in [1, 2, 4]:
        sdf[f"speed_dev_lag_{lag}"] = s.groupby(GROUP_COLS)["speed_deviation"].shift(lag)
        sdf[f"volume_dev_lag_{lag}"] = s.groupby(GROUP_COLS)["volume_deviation"].shift(lag)

# Naive baseline
naive_mae = mean_absolute_error(test_df["target_speed"], test_df["avg_speed_kph"])
print(f"Naive baseline (next = current): MAE = {naive_mae:.4f}")

# ── 4. Feature sets ─────────────────────────────────────────
FEAT_BASE = [
    "centreline_id", "longitude", "latitude",
    "volume_15min", "avg_speed_kph",
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos",
    "is_raining", "is_snowing", "is_foggy",
    "hour", "minute", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_morning_rush", "is_evening_rush",
    "speed_lag_1", "speed_lag_2", "speed_lag_4", "speed_lag_8",
    "speed_lag_12", "speed_lag_16",
    "volume_lag_1", "volume_lag_2", "volume_lag_4", "volume_lag_8",
    "volume_lag_12", "volume_lag_16",
    "speed_roll_mean_4", "speed_roll_mean_8",
    "speed_roll_std_4", "speed_roll_std_8",
    "volume_roll_mean_4", "volume_roll_mean_8",
]

FEAT_HIST = FEAT_BASE + [
    "hist_avg_speed", "hist_avg_volume",
    "speed_deviation", "volume_deviation",
    "speed_dev_lag_1", "speed_dev_lag_2", "speed_dev_lag_4",
    "volume_dev_lag_1", "volume_dev_lag_2", "volume_dev_lag_4",
]

# Acceleration features (speed change between lags)
for sdf in [train_df, val_df, test_df]:
    sdf["speed_accel_1"] = sdf["avg_speed_kph"] - sdf["speed_lag_1"]   # current vs 15min ago
    sdf["speed_accel_2"] = sdf["speed_lag_1"] - sdf["speed_lag_2"]     # 15min ago vs 30min ago
    sdf["speed_accel_4"] = sdf["speed_lag_2"] - sdf["speed_lag_4"]
    sdf["volume_accel_1"] = sdf["volume_15min"] - sdf["volume_lag_1"]
    sdf["speed_range_4"] = sdf[["avg_speed_kph", "speed_lag_1", "speed_lag_2", "speed_lag_4"]].max(axis=1) - \
                           sdf[["avg_speed_kph", "speed_lag_1", "speed_lag_2", "speed_lag_4"]].min(axis=1)

FEAT_ACCEL = FEAT_BASE + [
    "speed_accel_1", "speed_accel_2", "speed_accel_4",
    "volume_accel_1", "speed_range_4",
]

FEAT_ALL = FEAT_ACCEL + [
    "hist_avg_speed", "hist_avg_volume",
    "speed_deviation",
]

# ── 5. Runner ───────────────────────────────────────────────
results = []

def run(name, feats, params):
    mask_tr = train_df[feats + [TARGET]].notna().all(axis=1)
    mask_val = val_df[feats + [TARGET]].notna().all(axis=1)
    mask_te = test_df[feats + [TARGET]].notna().all(axis=1)

    X_tr = train_df.loc[mask_tr, feats]
    X_v = val_df.loc[mask_val, feats]
    X_te = test_df.loc[mask_te, feats]
    y_tr = train_df.loc[mask_tr, TARGET]
    y_v = val_df.loc[mask_val, TARGET]
    y_te = test_df.loc[mask_te, TARGET]

    m = XGBRegressor(**params)
    t0 = perf_counter()
    m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=0)
    t = perf_counter() - t0

    yp = m.predict(X_te)
    mae = mean_absolute_error(y_te, yp)
    rmse = np.sqrt(mean_squared_error(y_te, yp))
    r2 = r2_score(y_te, yp)
    results.append({"step": name, "MAE": mae, "RMSE": rmse, "R2": r2, "time": t,
                     "best_iter": m.best_iteration, "n_feat": len(feats)})
    print(f"  {name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  "
          f"iter={m.best_iteration}  time={t:.1f}s")
    return m

XGB_BASE = dict(
    n_estimators=3000, learning_rate=0.05, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", device="cuda",
    eval_metric="mae", early_stopping_rounds=50,
    random_state=42,
)

# ── A. Baseline (no hist features) ─────────────────────────
print("\n--- A. Baseline (no hist) ---")
run("A. Baseline", FEAT_BASE, XGB_BASE)

# ── B. + hist_avg_speed features ────────────────────────────
print("\n--- B. + hist avg + deviation ---")
run("B. + Hist", FEAT_HIST, XGB_BASE)

# ── C. Base + acceleration features ─────────────────────────
print("\n--- C. + Acceleration ---")
run("C. + Accel", FEAT_ACCEL, XGB_BASE)

# ── D. Accel + deeper + lower LR ───────────────────────────
print("\n--- D. Accel+Deep ---")
run("D. Accel+Deep", FEAT_ACCEL, {**XGB_BASE,
    "n_estimators": 5000, "learning_rate": 0.02, "max_depth": 8,
    "early_stopping_rounds": 80})

# ── E. All features (accel + hist, less deviation noise) ───
print("\n--- E. Accel + hist (selective) ---")
run("E. Accel+Hist", FEAT_ALL, XGB_BASE)

# ── F. Best feats + tuned ──────────────────────────────────
print("\n--- F. Best feats tuned ---")
run("F. Tuned", FEAT_ACCEL, {**XGB_BASE,
    "n_estimators": 5000, "learning_rate": 0.02, "max_depth": 7,
    "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85,
    "early_stopping_rounds": 80})

# ── Summary ─────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"{'Step':<20} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Iter':>6} {'Feat':>5} {'Time':>6}")
print(f"{'='*75}")
for r in results:
    print(f"{r['step']:<20} {r['MAE']:>7.4f} {r['RMSE']:>7.4f} {r['R2']:>7.4f} "
          f"{r['best_iter']:>6} {r['n_feat']:>5} {r['time']:>5.1f}s")
print(f"{'='*75}")
print(f"Naive baseline MAE: {naive_mae:.4f}")
