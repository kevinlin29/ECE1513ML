"""Full model comparison — text table + saved CSV."""
import pandas as pd
import os
from config import RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

data = [
    ("Global Mean",        9.596, 11.865, -0.341,   0,   "Naive"),
    ("Historical Avg",     8.403, 10.668, -0.084,   0,   "Naive"),
    ("Persistence",        4.193,  6.384,  0.612,   0,   "Naive (best)"),
    ("KNN (raw)",          5.042,  6.808,  0.559,   5.1, "ML raw"),
    ("LightGBM (raw)",     3.954,  5.732,  0.687, 119.2, "ML raw"),
    ("XGBoost (raw)",      3.840,  5.566,  0.705,   2.4, "ML raw"),
    ("ResNet MLP (raw)",   3.858,  5.647,  0.696,  47.9, "ML raw"),
    ("KNN (eng)",          3.889,  5.526,  0.709,   0.0, "ML engineered"),
    ("LightGBM (eng)",     3.308,  4.931,  0.769,  30.8, "ML engineered"),
    ("XGBoost (eng)",      3.194,  4.752,  0.783,   4.2, "ML engineered"),
    ("ResNet MLP (eng)",   3.280,  4.954,  0.766, 105.7, "ML engineered"),
    ("XGBoost (tuned)",    3.180,  4.775,  0.781,  49.0, "ML tuned"),
    ("ResNet MLP (tuned)", 3.172,  4.771,  0.781, 139.1, "ML tuned"),
]

df = pd.DataFrame(data, columns=["Model", "MAE", "RMSE", "R2", "Time_s", "Type"])

# Print formatted table
print()
print("=" * 85)
print(f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Time':>8} {'Type':<16}")
print("=" * 85)

prev_type = None
for _, row in df.iterrows():
    if row["Type"] != prev_type:
        if prev_type is not None:
            print("-" * 85)
        prev_type = row["Type"]

    time_str = "—" if row["Time_s"] == 0 else f"{row['Time_s']:.1f}s"
    print(f"{row['Model']:<22} {row['MAE']:>7.3f} {row['RMSE']:>7.3f} {row['R2']:>7.3f} {time_str:>8} {row['Type']:<16}")

print("=" * 85)

# Summary stats
best_naive = df[df["Type"].str.contains("Naive")]["MAE"].min()
best_raw = df[df["Type"] == "ML raw"]["MAE"].min()
best_eng = df[df["Type"] == "ML engineered"]["MAE"].min()
best_tuned = df[df["Type"] == "ML tuned"]["MAE"].min()

print(f"\nBest naive (Persistence):   MAE {best_naive:.3f}")
print(f"Best ML raw (XGBoost):      MAE {best_raw:.3f}  ({(best_naive - best_raw)/best_naive*100:.1f}% over naive)")
print(f"Best ML engineered (XGB):   MAE {best_eng:.3f}  ({(best_naive - best_eng)/best_naive*100:.1f}% over naive)")
print(f"Best ML tuned (ResNet MLP): MAE {best_tuned:.3f}  ({(best_naive - best_tuned)/best_naive*100:.1f}% over naive)")

df.to_csv(os.path.join(RESULTS_DIR, "full_comparison.csv"), index=False)
print(f"\nSaved to {RESULTS_DIR}/full_comparison.csv")
