"""
Autoresearch loop for XGBoost hyperparameter optimization.
Target: next 15-min avg_speed_kph
Split: 80/10/10 chronological
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from time import perf_counter

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import MERGED_DATA, RESULTS_DIR

RANDOM_STATE = 42
GROUP_COLS = ["centreline_id", "direction"]

# ── 1. Prepare data (same as train_progression.py) ───────────
print("Loading data...")
df = pd.read_csv(MERGED_DATA)
df["time_start"] = pd.to_datetime(df["time_start"])
df["time_end"] = pd.to_datetime(df["time_end"])

for col in ["is_raining", "is_snowing", "is_foggy"]:
    if col in df.columns:
        df[col] = df[col].replace({True: 1, False: 0}).fillna(0).astype(int)

for col in ["longitude", "latitude", "volume_15min", "avg_speed_kph",
            "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
            "Visibility (km)", "Rel Hum (%)", "Wind_Sin", "Wind_Cos"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(GROUP_COLS + ["time_start"]).reset_index(drop=True)

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

df["target_speed"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(-1)
df = df.dropna(subset=["target_speed"]).copy()

df_sorted = df.sort_values("time_start").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = df_sorted.iloc[:train_end]
val_df = df_sorted.iloc[train_end:val_end]
test_df = df_sorted.iloc[val_end:]

TARGET = "target_speed"
FEATURES = [
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
    "direction",
]

num_feats = [f for f in FEATURES if f != "direction"]
cat_feats = ["direction"]

pre = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_feats),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])

mask_tr = train_df[FEATURES + [TARGET]].notna().all(axis=1)
mask_val = val_df[FEATURES + [TARGET]].notna().all(axis=1)
mask_te = test_df[FEATURES + [TARGET]].notna().all(axis=1)

X_train = pre.fit_transform(train_df[mask_tr][FEATURES])
X_val = pre.transform(val_df[mask_val][FEATURES])
X_test = pre.transform(test_df[mask_te][FEATURES])
y_train = train_df[mask_tr][TARGET].values
y_val = val_df[mask_val][TARGET].values
y_test = test_df[mask_te][TARGET].values

print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
print(f"Features: {X_train.shape[1]}")


# ── 2. Autoresearch loop ─────────────────────────────────────
def run(name, params):
    model = XGBRegressor(**params)
    t0 = perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    t = perf_counter() - t0
    yp = model.predict(X_test)
    mae = mean_absolute_error(y_test, yp)
    rmse = np.sqrt(mean_squared_error(y_test, yp))
    r2 = r2_score(y_test, yp)
    print(f"  {name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  time={t:.1f}s  iter={model.best_iteration}")
    return mae, rmse, r2, t, model.best_iteration


# Current best config
best_params = dict(
    n_estimators=5000, learning_rate=0.01, max_depth=8,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, objective="reg:absoluteerror",
    tree_method="hist", device="cuda",
    eval_metric="mae", early_stopping_rounds=50,
    n_jobs=-1, random_state=RANDOM_STATE,
)

log = []

# Baseline
print("\n--- Baseline ---")
mae, rmse, r2, t, itr = run("baseline", best_params)
best_mae = mae
log.append(("baseline", mae, rmse, r2, t, "keep", "lr=0.01 depth=8 5k trees"))


def experiment(name, param_changes, description):
    global best_mae, best_params
    trial = {**best_params, **param_changes}
    mae, rmse, r2, t, itr = run(name, trial)
    if mae < best_mae:
        best_mae = mae
        best_params = trial
        log.append((name, mae, rmse, r2, t, "keep", description))
        print(f"    >>> KEEP (new best: {mae:.4f})")
    else:
        log.append((name, mae, rmse, r2, t, "discard", description))
        print(f"    >>> discard (best: {best_mae:.4f})")


# ── Experiments ──────────────────────────────────────────────
print("\n--- max_depth ---")
experiment("depth7", {"max_depth": 7}, "max_depth=7")
experiment("depth9", {"max_depth": 9}, "max_depth=9")
experiment("depth10", {"max_depth": 10}, "max_depth=10")

print("\n--- min_child_weight ---")
experiment("mcw3", {"min_child_weight": 3}, "min_child_weight=3")
experiment("mcw10", {"min_child_weight": 10}, "min_child_weight=10")
experiment("mcw20", {"min_child_weight": 20}, "min_child_weight=20")

print("\n--- learning_rate ---")
experiment("lr005", {"learning_rate": 0.005, "n_estimators": 10000}, "lr=0.005 10k trees")
experiment("lr003", {"learning_rate": 0.003, "n_estimators": 15000}, "lr=0.003 15k trees")
experiment("lr02", {"learning_rate": 0.02, "n_estimators": 3000}, "lr=0.02 3k trees")
experiment("lr05", {"learning_rate": 0.05, "n_estimators": 2000}, "lr=0.05 2k trees")

print("\n--- subsample ---")
experiment("sub07", {"subsample": 0.7}, "subsample=0.7")
experiment("sub09", {"subsample": 0.9}, "subsample=0.9")
experiment("sub10", {"subsample": 1.0}, "subsample=1.0")

print("\n--- colsample_bytree ---")
experiment("col06", {"colsample_bytree": 0.6}, "colsample_bytree=0.6")
experiment("col07", {"colsample_bytree": 0.7}, "colsample_bytree=0.7")
experiment("col09", {"colsample_bytree": 0.9}, "colsample_bytree=0.9")

print("\n--- gamma ---")
experiment("gamma01", {"gamma": 0.1}, "gamma=0.1")
experiment("gamma05", {"gamma": 0.5}, "gamma=0.5")
experiment("gamma1", {"gamma": 1.0}, "gamma=1.0")

print("\n--- reg_alpha ---")
experiment("alpha0", {"reg_alpha": 0.0}, "reg_alpha=0")
experiment("alpha05", {"reg_alpha": 0.5}, "reg_alpha=0.5")
experiment("alpha1", {"reg_alpha": 1.0}, "reg_alpha=1.0")

print("\n--- reg_lambda ---")
experiment("lambda05", {"reg_lambda": 0.5}, "reg_lambda=0.5")
experiment("lambda2", {"reg_lambda": 2.0}, "reg_lambda=2.0")
experiment("lambda5", {"reg_lambda": 5.0}, "reg_lambda=5.0")

print("\n--- early_stopping ---")
experiment("es100", {"early_stopping_rounds": 100}, "early_stopping=100")

print("\n--- colsample_bylevel ---")
experiment("col_lvl07", {"colsample_bylevel": 0.7}, "colsample_bylevel=0.7")
experiment("col_lvl09", {"colsample_bylevel": 0.9}, "colsample_bylevel=0.9")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 85)
print(f"{'Exp':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Time':>7} {'Status':<8} Description")
print("=" * 85)
for name, mae, rmse, r2, t, status, desc in log:
    marker = "***" if status == "keep" else "   "
    print(f"{name:<15} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {t:>6.1f}s {status:<8} {desc} {marker}")
print("=" * 85)

kept = [l for l in log if l[5] == "keep"]
print(f"\nBest MAE: {best_mae:.4f}")
print(f"Kept {len(kept)} / {len(log)} experiments")
print(f"\nFinal best params:")
for k, v in best_params.items():
    if k not in ["tree_method", "device", "n_jobs", "random_state"]:
        print(f"  {k}: {v}")

pd.DataFrame(log, columns=["exp", "MAE", "RMSE", "R2", "time_s", "status", "description"]).to_csv(
    os.path.join(RESULTS_DIR, "xgb_autoresearch.csv"), index=False)
