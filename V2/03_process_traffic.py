"""
Step 3: Process raw traffic speed data — compute avg speed from speed bins.
Notebook cell 3.
"""
import pandas as pd
import numpy as np
import os
from config import SPEED_DATA_PATHS, PROCESSED_DIR, TRAFFIC_PROCESSED

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── 1. Load speed data ───────────────────────────────────────
speed_cols = [
    "centreline_id", "location_name", "longitude", "latitude",
    "time_start", "time_end", "direction",
    "vol_1_19kph", "vol_20_25kph", "vol_26_30kph", "vol_31_35kph",
    "vol_36_40kph", "vol_41_45kph", "vol_46_50kph", "vol_51_55kph",
    "vol_56_60kph", "vol_61_65kph", "vol_66_70kph", "vol_71_75kph",
    "vol_76_80kph", "vol_81_160kph",
]

dfs = []
for path in SPEED_DATA_PATHS:
    if os.path.exists(path):
        print(f"Reading: {path}")
        dfs.append(pd.read_csv(path, usecols=speed_cols, low_memory=False))
        print(f"  Loaded {len(dfs[-1])} rows")
df = pd.concat(dfs, ignore_index=True)
print(f"Total: {len(df)} rows")

# ── 2. Parse timestamps ──────────────────────────────────────
df["time_start"] = pd.to_datetime(df["time_start"])
df["time_end"] = pd.to_datetime(df["time_end"])

# ── 3. Compute volume and weighted average speed ─────────────
speed_bins = [
    "vol_1_19kph", "vol_20_25kph", "vol_26_30kph", "vol_31_35kph",
    "vol_36_40kph", "vol_41_45kph", "vol_46_50kph", "vol_51_55kph",
    "vol_56_60kph", "vol_61_65kph", "vol_66_70kph", "vol_71_75kph",
    "vol_76_80kph", "vol_81_160kph",
]
midpoints = np.array([10, 22.5, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 100])

df["volume_15min"] = df[speed_bins].sum(axis=1)

weighted_sum = (df[speed_bins] * midpoints).sum(axis=1)
df["avg_speed_kph"] = np.where(
    df["volume_15min"] > 0, weighted_sum / df["volume_15min"], np.nan
)

# ── 4. Keep only useful columns, filter zero-volume rows ─────
final_cols = [
    "centreline_id", "location_name", "longitude", "latitude", "direction",
    "time_start", "time_end", "volume_15min", "avg_speed_kph",
]
clean = df[final_cols].sort_values(["centreline_id", "time_start"]).reset_index(drop=True)
clean = clean[clean["volume_15min"] > 0].reset_index(drop=True)

# ── 5. Save ──────────────────────────────────────────────────
clean.to_csv(TRAFFIC_PROCESSED, index=False)
print(f"\nProcessed traffic saved ({len(clean)} rows): {TRAFFIC_PROCESSED}")
print(clean.head())
