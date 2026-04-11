"""
Step 5: Train and evaluate KNN, LightGBM, and XGBoost models.
Notebook cells 7+8 (expanded to three models).
"""
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from config import MERGED_DATA, MODELS_DIR, RESULTS_DIR, XGB_MODEL_PATH

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────
print("Loading merged dataset...")
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

# ── 3. Speed Deviation ───────────────────────────────────────
historical_avg = (
    df.groupby(["centreline_id", "direction", "hour", "dayofweek"])["avg_speed_kph"]
    .transform("mean")
)
df["speed_deviation"] = df["avg_speed_kph"] - historical_avg

historical_vol_avg = (
    df.groupby(["centreline_id", "direction", "hour", "dayofweek"])["volume_15min"]
    .transform("mean")
)
df["volume_deviation"] = df["volume_15min"] - historical_vol_avg

# ── 4. Lag Features ──────────────────────────────────────────
group_keys = ["centreline_id", "direction"]

for lag in [1, 2, 4, 8]:
    df[f"speed_dev_lag_{lag}"] = df.groupby(group_keys)["speed_deviation"].shift(lag)
    df[f"volume_dev_lag_{lag}"] = df.groupby(group_keys)["volume_deviation"].shift(lag)

for lag in [2, 4, 8]:
    df[f"speed_lag_{lag}"] = df.groupby(group_keys)["avg_speed_kph"].shift(lag)
    df[f"volume_lag_{lag}"] = df.groupby(group_keys)["volume_15min"].shift(lag)

# ── 5. Rolling Window Features ────────────────────────────────
for window in [4, 8]:
    grp = df.groupby(group_keys)

    df[f"speed_dev_roll_mean_{window}"] = grp["speed_deviation"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df[f"speed_dev_roll_std_{window}"] = grp["speed_deviation"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )
    df[f"volume_dev_roll_mean_{window}"] = grp["volume_deviation"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df[f"speed_roll_mean_{window}"] = grp["avg_speed_kph"].transform(
        lambda x: x.shift(2).rolling(window, min_periods=1).mean()
    )

# ── 6. Historical context ────────────────────────────────────
df["hist_avg_speed"] = historical_avg
df["hist_avg_volume"] = historical_vol_avg

# ── 7. Feature / Target Definition ───────────────────────────
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
    # Historical context
    "hist_avg_speed", "hist_avg_volume",
    # Deviation lags
    "speed_dev_lag_1", "speed_dev_lag_2", "speed_dev_lag_4", "speed_dev_lag_8",
    "volume_dev_lag_1", "volume_dev_lag_2", "volume_dev_lag_4", "volume_dev_lag_8",
    # Deviation rolling stats
    "speed_dev_roll_mean_4", "speed_dev_roll_mean_8",
    "speed_dev_roll_std_4", "speed_dev_roll_std_8",
    "volume_dev_roll_mean_4", "volume_dev_roll_mean_8",
    # Raw lags (>= 2)
    "speed_lag_2", "speed_lag_4", "speed_lag_8",
    "volume_lag_2", "volume_lag_4", "volume_lag_8",
    # Raw rolling (shifted by 2)
    "speed_roll_mean_4", "speed_roll_mean_8",
    # Current volume
    "volume_15min",
]

df_model = df.dropna(subset=FEATURES + [TARGET]).copy()
print(f"\nRecords after dropping NaN: {len(df_model)}")

# ── 8. Time-Aware Train / Val / Test Split ───────────────────
n = len(df_model)
sorted_df = df_model.sort_values("time_start").reset_index(drop=True)

train_end = int(n * 0.70)
val_end = int(n * 0.80)

train_df = sorted_df.iloc[:train_end]
val_df = sorted_df.iloc[train_end:val_end]
test_df = sorted_df.iloc[val_end:]

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train range: {train_df['time_start'].min()} to {train_df['time_start'].max()}")
print(f"Test range:  {test_df['time_start'].min()} to {test_df['time_start'].max()}")

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_val = val_df[FEATURES]
y_val = val_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# ── 9. Evaluate helper ───────────────────────────────────────
results = []


def evaluate(name, y_true, y_pred, elapsed):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "train_time_s": elapsed})
    print(f"\n{'=' * 40}")
    print(f" {name}")
    print(f"{'=' * 40}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R^2  : {r2:.4f}")
    print(f"  Time : {elapsed:.1f}s")


# ── 10. KNN ──────────────────────────────────────────────────
print("\n--- Training KNN ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

t0 = time.time()
knn = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
knn.fit(X_train_scaled, y_train)
elapsed_knn = time.time() - t0

y_pred_knn = knn.predict(X_test_scaled)
evaluate("KNN (k=10)", y_test, y_pred_knn, elapsed_knn)

joblib.dump(knn, os.path.join(MODELS_DIR, "knn_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "knn_scaler.pkl"))

# ── 11. LightGBM ─────────────────────────────────────────────
print("\n--- Training LightGBM ---")
t0 = time.time()
lgbm = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        __import__("lightgbm").early_stopping(30, verbose=False),
        __import__("lightgbm").log_evaluation(50),
    ],
)
elapsed_lgbm = time.time() - t0

y_pred_lgbm = lgbm.predict(X_test)
evaluate("LightGBM", y_test, y_pred_lgbm, elapsed_lgbm)

joblib.dump(lgbm, os.path.join(MODELS_DIR, "lgbm_model.pkl"))

# ── 12. XGBoost ──────────────────────────────────────────────
print("\n--- Training XGBoost ---")
t0 = time.time()
xgb = XGBRegressor(
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
    early_stopping_rounds=30,
    eval_metric="rmse",
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
elapsed_xgb = time.time() - t0

y_pred_xgb = xgb.predict(X_test)
evaluate("XGBoost", y_test, y_pred_xgb, elapsed_xgb)

joblib.dump(xgb, XGB_MODEL_PATH)

# ── 13. Summary Table ────────────────────────────────────────
print("\n" + "=" * 60)
print(" MODEL COMPARISON")
print("=" * 60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

# ── 14. XGBoost Feature Importance ───────────────────────────
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": xgb.feature_importances_,
}).sort_values("importance", ascending=False)

print(f"\nTop 20 XGBoost features:")
print(importance_df.head(20).to_string(index=False))
importance_df.to_csv(os.path.join(RESULTS_DIR, "xgb_feature_importance.csv"), index=False)

# ── 15. Save XGBoost predictions ─────────────────────────────
preds_df = test_df[["centreline_id", "direction", "time_start", TARGET]].copy()
preds_df["predicted_speed"] = y_pred_xgb
preds_df.to_csv(os.path.join(RESULTS_DIR, "xgb_predictions.csv"), index=False)

print(f"\nAll models saved to: {MODELS_DIR}")
print(f"Results saved to: {RESULTS_DIR}")
