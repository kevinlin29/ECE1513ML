"""
ECE1513 — XGBoost (Tuned) Learning Curve
Matches the final tuned config: MAE objective, lr=0.01, depth=8, 5000 rounds
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from time import perf_counter

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from config import MERGED_DATA, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
GROUP_COLS = ["centreline_id", "direction"]

# ── 1. Load & Prepare (same as train_progression.py) ───────
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

# Target
df["target_speed"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(-1)
df = df.dropna(subset=["target_speed"]).copy()

# Feature set (FEAT_ENHANCED from train_progression.py)
FEAT_BASE = [
    "centreline_id", "longitude", "latitude",
    "volume_15min", "avg_speed_kph",
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos",
    "is_raining", "is_snowing", "is_foggy",
    "hour", "minute", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "speed_lag_1", "speed_lag_2", "speed_lag_4", "speed_lag_8",
    "volume_lag_1", "volume_lag_2", "volume_lag_4", "volume_lag_8",
    "speed_roll_mean_4", "speed_roll_mean_8",
    "speed_roll_std_4", "speed_roll_std_8",
    "volume_roll_mean_4", "volume_roll_mean_8",
    "direction",
]
FEATURES = FEAT_BASE + [
    "is_morning_rush", "is_evening_rush",
    "speed_lag_12", "speed_lag_16",
    "volume_lag_12", "volume_lag_16",
]
TARGET = "target_speed"

# ── 2. Split (80/10/10) ────────────────────────────────────
df_sorted = df.sort_values("time_start").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = df_sorted.iloc[:train_end].copy()
val_df = df_sorted.iloc[train_end:val_end].copy()
test_df = df_sorted.iloc[val_end:].copy()
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── 3. Preprocess ──────────────────────────────────────────
num_feats = [f for f in FEATURES if f != "direction"]
cat_feats = ["direction"]

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_feats),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])

mask_tr = train_df[FEATURES + [TARGET]].notna().all(axis=1)
mask_val = val_df[FEATURES + [TARGET]].notna().all(axis=1)
mask_te = test_df[FEATURES + [TARGET]].notna().all(axis=1)

X_tr = preprocessor.fit_transform(train_df[mask_tr][FEATURES])
X_va = preprocessor.transform(val_df[mask_val][FEATURES])
X_te = preprocessor.transform(test_df[mask_te][FEATURES])
y_tr = train_df[mask_tr][TARGET]
y_va = val_df[mask_val][TARGET]
y_te = test_df[mask_te][TARGET]

# ── 4. Train tuned XGBoost with eval on train + val ────────
print("Training tuned XGBoost (capturing learning curve)...")

xgb = XGBRegressor(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:absoluteerror",
    tree_method="hist",
    device="cuda",
    eval_metric="mae",
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

t0 = perf_counter()
xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_va, y_va)],
    verbose=200,
)
elapsed = perf_counter() - t0

y_pred = xgb.predict(X_te)
test_mae = mean_absolute_error(y_te, y_pred)

print(f"Best iteration: {xgb.best_iteration}")
print(f"Training time: {elapsed:.1f}s")
print(f"Test MAE: {test_mae:.4f}")

# ── 5. Extract learning curve data ─────────────────────────
evals = xgb.evals_result()
train_mae = evals["validation_0"]["mae"]
val_mae = evals["validation_1"]["mae"]
rounds = list(range(len(train_mae)))

# ── 6. Plot ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(rounds, train_mae, label="Train MAE", color="#2196F3", linewidth=1.5)
ax.plot(rounds, val_mae, label="Validation MAE", color="#F44336", linewidth=1.5)
ax.axvline(x=xgb.best_iteration, color="gray", linestyle="--", alpha=0.7,
           label=f"Best iteration ({xgb.best_iteration})")

ax.set_xlabel("Boosting Round", fontsize=12)
ax.set_ylabel("MAE (km/h)", fontsize=12)
ax.set_title("XGBoost (Tuned) Learning Curve — Train vs Validation MAE", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate final values
best_iter = xgb.best_iteration
ax.annotate(f"Train: {train_mae[best_iter]:.3f}",
            xy=(best_iter, train_mae[best_iter]),
            xytext=(best_iter + 80, train_mae[best_iter] - 0.2),
            fontsize=10, color="#2196F3",
            arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.2))
ax.annotate(f"Val: {val_mae[best_iter]:.3f}",
            xy=(best_iter, val_mae[best_iter]),
            xytext=(best_iter + 80, val_mae[best_iter] + 0.2),
            fontsize=10, color="#F44336",
            arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.2))

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "xgb_learning_curve.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved learning curve to {out_path}")
plt.close()
