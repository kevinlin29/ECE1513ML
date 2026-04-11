"""
ECE1513 — Naive baselines (no ML)
Shows how much ML adds over simple heuristics.
Target: next 15-min avg_speed_kph
Split: 80/10/10 chronological
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import MERGED_DATA, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
GROUP_COLS = ["centreline_id", "direction"]

# ── 1. Load & prepare ────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(MERGED_DATA)
df["time_start"] = pd.to_datetime(df["time_start"])
df = df.sort_values(GROUP_COLS + ["time_start"]).reset_index(drop=True)

df["hour"] = df["time_start"].dt.hour
df["day_of_week"] = df["time_start"].dt.dayofweek

# Target: next 15-min speed
df["target_speed"] = df.groupby(GROUP_COLS)["avg_speed_kph"].shift(-1)
df = df.dropna(subset=["target_speed"]).copy()

# Split 80/10/10
df_sorted = df.sort_values("time_start").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_df = df_sorted.iloc[:train_end].copy()
test_df = df_sorted.iloc[val_end:].copy()

y_test = test_df["target_speed"].values

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

results = []


def evaluate(name, y_pred, description):
    mask = ~np.isnan(y_pred)
    mae = mean_absolute_error(y_test[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
    r2 = r2_score(y_test[mask], y_pred[mask])
    coverage = mask.sum() / len(y_test) * 100
    results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "coverage": coverage, "desc": description})
    print(f"  {name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  coverage={coverage:.1f}%")


# ── 2. Naive Baseline 1: Global Mean ─────────────────────────
print("\n--- Baseline 1: Global Mean ---")
global_mean = train_df["avg_speed_kph"].mean()
y_pred_global = np.full(len(y_test), global_mean)
evaluate("Global Mean", y_pred_global, f"Predict {global_mean:.1f} km/h for everything")

# ── 3. Naive Baseline 2: Current Speed = Next Speed ──────────
print("\n--- Baseline 2: Persistence (current speed = next speed) ---")
y_pred_persist = test_df["avg_speed_kph"].values
evaluate("Persistence", y_pred_persist, "Predict next speed = current speed")

# ── 4. Naive Baseline 3: Historical Average ──────────────────
print("\n--- Baseline 3: Historical Average (per segment/hour/DOW) ---")
hist_keys = ["centreline_id", "direction", "hour", "day_of_week"]
hist_avg = train_df.groupby(hist_keys)["avg_speed_kph"].mean()

# Map to test set
idx = pd.MultiIndex.from_arrays([test_df[k] for k in hist_keys])
y_pred_hist = idx.map(hist_avg).values.astype(float)

# Fallback for missing combos: segment-level avg
seg_avg = train_df.groupby(["centreline_id", "direction"])["avg_speed_kph"].mean()
missing = np.isnan(y_pred_hist)
if missing.any():
    seg_idx = pd.MultiIndex.from_arrays([test_df.loc[test_df.index[missing], "centreline_id"],
                                          test_df.loc[test_df.index[missing], "direction"]])
    y_pred_hist[missing] = seg_idx.map(seg_avg).values.astype(float)

# Final fallback: global mean
still_missing = np.isnan(y_pred_hist)
y_pred_hist[still_missing] = global_mean

evaluate("Historical Avg", y_pred_hist, "Avg speed for this segment/hour/DOW from training data")

# ── 5. Summary ───────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"{'Model':<20} {'MAE':>7} {'RMSE':>7} {'R²':>7}  Description")
print("=" * 75)
for r in results:
    print(f"{r['model']:<20} {r['MAE']:>7.4f} {r['RMSE']:>7.4f} {r['R2']:>7.4f}  {r['desc']}")

# Add ML results for comparison
print("-" * 75)
ml_results = [
    ("XGBoost (raw)", 3.840, 5.566, 0.705, "19 raw features, no engineering"),
    ("XGBoost (tuned)", 3.180, 4.775, 0.781, "45 features, MAE loss, depth=8"),
    ("ResNet MLP", 3.172, 4.771, 0.781, "45 features, skip connections, SiLU"),
]
for name, mae, rmse, r2, desc in ml_results:
    print(f"{name:<20} {mae:>7.4f} {rmse:>7.4f} {r2:>7.4f}  {desc}")
print("=" * 75)

best_naive = min(r["MAE"] for r in results)
best_ml = 3.172
print(f"\nBest naive: MAE {best_naive:.4f}")
print(f"Best ML:    MAE {best_ml:.4f}")
print(f"ML improvement over best naive: {(best_naive - best_ml) / best_naive * 100:.1f}%")

pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "naive_baselines.csv"), index=False)
