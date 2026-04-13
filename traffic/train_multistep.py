"""
ECE1513 — Multi-step approach for traffic speed prediction
Step 1: XGBoost predicts CURRENT speed from lags (demonstrates ~2.x MAE)
Step 2: Use Step 1's prediction + error as features to predict NEXT speed
         (model learns: "is current situation unusual? how will it evolve?")
Step 3: ResNet on same enriched features for next-step prediction
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from time import perf_counter

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import torch
import torch.nn as nn

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

# Lag features
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

# Acceleration
df["speed_accel_1"] = df["avg_speed_kph"] - df["speed_lag_1"]
df["speed_accel_2"] = df["speed_lag_1"] - df["speed_lag_2"]

# Targets
df["target_speed"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(-1)
df = df.dropna(subset=["target_speed"]).copy()

# ── 2. Split (80/10/10) ─────────────────────────────────────
df_sorted = df.sort_values("time_start").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = df_sorted.iloc[:train_end].copy()
val_df = df_sorted.iloc[train_end:val_end].copy()
test_df = df_sorted.iloc[val_end:].copy()
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

results = []

# ── Features shared across steps ────────────────────────────
# Step 1 features: predict CURRENT speed from lags only (no avg_speed_kph!)
FEAT_STEP1 = [
    "centreline_id", "longitude", "latitude",
    "volume_15min",
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
    "direction",
]

# Step 2 features: predict NEXT speed — includes current speed + Step 1 outputs
FEAT_STEP2_BASE = FEAT_STEP1 + [
    "avg_speed_kph",
    "speed_accel_1", "speed_accel_2",
]

XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.05, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", device="cuda",
    eval_metric="mae", early_stopping_rounds=50,
    random_state=42, enable_categorical=True,
)

# ══════════════════════════════════════════════════════════════
# STEP 1: Predict CURRENT speed from lags
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Predict CURRENT speed (avg_speed_kph) from lags")
print("=" * 60)

TARGET_S1 = "avg_speed_kph"

# Convert direction to categorical for XGBoost
for sdf in [train_df, val_df, test_df]:
    sdf["direction"] = sdf["direction"].astype("category")

mask_tr = train_df[FEAT_STEP1 + [TARGET_S1]].notna().all(axis=1)
mask_val = val_df[FEAT_STEP1 + [TARGET_S1]].notna().all(axis=1)
mask_te = test_df[FEAT_STEP1 + [TARGET_S1]].notna().all(axis=1)

X_tr1 = train_df.loc[mask_tr, FEAT_STEP1]
X_v1 = val_df.loc[mask_val, FEAT_STEP1]
X_te1 = test_df.loc[mask_te, FEAT_STEP1]
y_tr1 = train_df.loc[mask_tr, TARGET_S1]
y_v1 = val_df.loc[mask_val, TARGET_S1]
y_te1 = test_df.loc[mask_te, TARGET_S1]

t0 = perf_counter()
model_s1 = XGBRegressor(**XGB_PARAMS)
model_s1.fit(X_tr1, y_tr1, eval_set=[(X_v1, y_v1)], verbose=0)
t1 = perf_counter() - t0

yp1 = model_s1.predict(X_te1)
mae1 = mean_absolute_error(y_te1, yp1)
rmse1 = np.sqrt(mean_squared_error(y_te1, yp1))
r2_1 = r2_score(y_te1, yp1)
results.append({"step": "1. Current speed", "MAE": mae1, "RMSE": rmse1, "R2": r2_1, "time": t1})
print(f"  MAE={mae1:.4f}  RMSE={rmse1:.4f}  R²={r2_1:.4f}  iter={model_s1.best_iteration}  time={t1:.1f}s")

# Generate Step 1 predictions for all splits (to use as features in Step 2)
for sdf in [train_df, val_df, test_df]:
    mask = sdf[FEAT_STEP1].notna().all(axis=1)
    sdf["s1_predicted_speed"] = np.nan
    sdf.loc[mask, "s1_predicted_speed"] = model_s1.predict(sdf.loc[mask, FEAT_STEP1])
    sdf["s1_error"] = sdf["avg_speed_kph"] - sdf["s1_predicted_speed"]  # positive = faster than expected

# ══════════════════════════════════════════════════════════════
# STEP 2: Predict NEXT speed — XGBoost baseline (no step1 info)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2a: Predict NEXT speed — XGBoost baseline")
print("=" * 60)

TARGET_S2 = "target_speed"

mask_tr = train_df[FEAT_STEP2_BASE + [TARGET_S2]].notna().all(axis=1)
mask_val = val_df[FEAT_STEP2_BASE + [TARGET_S2]].notna().all(axis=1)
mask_te = test_df[FEAT_STEP2_BASE + [TARGET_S2]].notna().all(axis=1)

X_tr2 = train_df.loc[mask_tr, FEAT_STEP2_BASE]
X_v2 = val_df.loc[mask_val, FEAT_STEP2_BASE]
X_te2 = test_df.loc[mask_te, FEAT_STEP2_BASE]
y_tr2 = train_df.loc[mask_tr, TARGET_S2]
y_v2 = val_df.loc[mask_val, TARGET_S2]
y_te2 = test_df.loc[mask_te, TARGET_S2]

t0 = perf_counter()
model_s2a = XGBRegressor(**XGB_PARAMS)
model_s2a.fit(X_tr2, y_tr2, eval_set=[(X_v2, y_v2)], verbose=0)
t2a = perf_counter() - t0

yp2a = model_s2a.predict(X_te2)
mae2a = mean_absolute_error(y_te2, yp2a)
rmse2a = np.sqrt(mean_squared_error(y_te2, yp2a))
r2_2a = r2_score(y_te2, yp2a)
results.append({"step": "2a. Next (base)", "MAE": mae2a, "RMSE": rmse2a, "R2": r2_2a, "time": t2a})
print(f"  MAE={mae2a:.4f}  RMSE={rmse2a:.4f}  R²={r2_2a:.4f}  iter={model_s2a.best_iteration}  time={t2a:.1f}s")

# ══════════════════════════════════════════════════════════════
# STEP 2b: Predict NEXT speed — with Step 1 stacking features
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2b: Predict NEXT speed — + Step 1 stacked features")
print("=" * 60)

FEAT_STEP2_STACKED = FEAT_STEP2_BASE + ["s1_predicted_speed", "s1_error"]

mask_tr = train_df[FEAT_STEP2_STACKED + [TARGET_S2]].notna().all(axis=1)
mask_val = val_df[FEAT_STEP2_STACKED + [TARGET_S2]].notna().all(axis=1)
mask_te = test_df[FEAT_STEP2_STACKED + [TARGET_S2]].notna().all(axis=1)

X_tr2b = train_df.loc[mask_tr, FEAT_STEP2_STACKED]
X_v2b = val_df.loc[mask_val, FEAT_STEP2_STACKED]
X_te2b = test_df.loc[mask_te, FEAT_STEP2_STACKED]
y_tr2b = train_df.loc[mask_tr, TARGET_S2]
y_v2b = val_df.loc[mask_val, TARGET_S2]
y_te2b = test_df.loc[mask_te, TARGET_S2]

t0 = perf_counter()
model_s2b = XGBRegressor(**XGB_PARAMS)
model_s2b.fit(X_tr2b, y_tr2b, eval_set=[(X_v2b, y_v2b)], verbose=0)
t2b = perf_counter() - t0

yp2b = model_s2b.predict(X_te2b)
mae2b = mean_absolute_error(y_te2b, yp2b)
rmse2b = np.sqrt(mean_squared_error(y_te2b, yp2b))
r2_2b = r2_score(y_te2b, yp2b)
results.append({"step": "2b. Next (stacked)", "MAE": mae2b, "RMSE": rmse2b, "R2": r2_2b, "time": t2b})
print(f"  MAE={mae2b:.4f}  RMSE={rmse2b:.4f}  R²={r2_2b:.4f}  iter={model_s2b.best_iteration}  time={t2b:.1f}s")

# ══════════════════════════════════════════════════════════════
# STEP 3: Ensemble — average Step 2a and 2b
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Ensemble (average of 2a + 2b)")
print("=" * 60)

# Need common test indices
mask_common = test_df[FEAT_STEP2_STACKED + [TARGET_S2]].notna().all(axis=1)
X_te_c_base = test_df.loc[mask_common, FEAT_STEP2_BASE]
X_te_c_stack = test_df.loc[mask_common, FEAT_STEP2_STACKED]
y_te_c = test_df.loc[mask_common, TARGET_S2]

yp_a = model_s2a.predict(X_te_c_base)
yp_b = model_s2b.predict(X_te_c_stack)

for w in [0.3, 0.5, 0.7]:
    yp_ens = w * yp_a + (1 - w) * yp_b
    mae_ens = mean_absolute_error(y_te_c, yp_ens)
    print(f"  w={w:.1f} base + {1-w:.1f} stacked: MAE={mae_ens:.4f}")

yp_avg = 0.5 * yp_a + 0.5 * yp_b
mae_avg = mean_absolute_error(y_te_c, yp_avg)
rmse_avg = np.sqrt(mean_squared_error(y_te_c, yp_avg))
r2_avg = r2_score(y_te_c, yp_avg)
results.append({"step": "3. Ensemble", "MAE": mae_avg, "RMSE": rmse_avg, "R2": r2_avg, "time": 0})

# ── Summary ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"{'Step':<25} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Time':>6}")
print(f"{'='*65}")
for r in results:
    print(f"{r['step']:<25} {r['MAE']:>7.4f} {r['RMSE']:>7.4f} {r['R2']:>7.4f} {r['time']:>5.1f}s")
print(f"{'='*65}")
