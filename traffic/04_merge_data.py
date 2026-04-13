"""
Step 4: Merge traffic + weather data, then run data quality checks.
Notebook cells 4 + 5 + 6.
"""
import pandas as pd
import os
from config import (
    PROCESSED_DIR, WEATHER_PROCESSED, TRAFFIC_PROCESSED,
    MERGED_DATA, LOCATION_SUMMARY,
)

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── 1. Load processed datasets ───────────────────────────────
print("Reading processed weather and traffic data...")
df_weather = pd.read_csv(WEATHER_PROCESSED, low_memory=False)
df_traffic = pd.read_csv(TRAFFIC_PROCESSED, low_memory=False)
print(f"  Traffic: {len(df_traffic)} rows | Weather: {len(df_weather)} rows")

# ── 2. Parse timestamps ──────────────────────────────────────
df_weather["Date/Time (LST)"] = pd.to_datetime(df_weather["Date/Time (LST)"])
df_traffic["time_start"] = pd.to_datetime(df_traffic["time_start"])
df_traffic["time_end"] = pd.to_datetime(df_traffic["time_end"])

# ── 3. Merge on floored hour ─────────────────────────────────
print("Merging on hourly alignment...")
df_traffic["join_time_hour"] = df_traffic["time_start"].dt.floor("h")

merged = pd.merge(
    df_traffic, df_weather,
    left_on="join_time_hour", right_on="Date/Time (LST)",
    how="left",
)

# Drop redundant columns
drop_cols = ["join_time_hour", "Date/Time (LST)", "Year", "Month", "Day", "Time (LST)"]
merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

# Forward-fill any missing weather features
weather_features = [
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)", "Visibility (km)",
    "Rel Hum (%)", "Wind_Sin", "Wind_Cos", "is_raining", "is_snowing", "is_foggy",
]
for col in weather_features:
    if col in merged.columns:
        merged[col] = merged[col].ffill()

merged.to_csv(MERGED_DATA, index=False)
print(f"\nMerged dataset saved ({len(merged)} rows): {MERGED_DATA}")

# ── 4. Data quality report ───────────────────────────────────
print("\n" + "=" * 50)
print(" DATA QUALITY REPORT")
print("=" * 50)

unique_locs = merged["centreline_id"].nunique()
print(f"\nUnique road segments: {unique_locs}")

merged["hour_trunc"] = merged["time_start"].dt.floor("h")
hours_per_loc = merged.groupby(["centreline_id", "location_name"])["hour_trunc"].nunique().reset_index()
hours_per_loc.rename(columns={"hour_trunc": "total_hours"}, inplace=True)
hours_per_loc["approx_days"] = (hours_per_loc["total_hours"] / 24).round(1)
hours_per_loc = hours_per_loc.sort_values("total_hours", ascending=False).reset_index(drop=True)

print(f"\nTop 10 segments by data duration:")
print(hours_per_loc.head(10).to_string())
print(f"\nBottom 10 segments:")
print(hours_per_loc.tail(10).to_string())

print(f"\nMax hours: {hours_per_loc['total_hours'].max()} ({hours_per_loc['approx_days'].max()} days)")
print(f"Min hours: {hours_per_loc['total_hours'].min()} ({hours_per_loc['approx_days'].min()} days)")
print(f"Median:    {hours_per_loc['total_hours'].median()} ({round(hours_per_loc['total_hours'].median()/24, 1)} days)")

hours_per_loc.to_csv(LOCATION_SUMMARY, index=False)
print(f"\nLocation summary saved: {LOCATION_SUMMARY}")

# ── 5. Filter impact analysis ────────────────────────────────
total_rows = len(merged)
total_locs = merged["centreline_id"].nunique()
hours_by_id = merged.groupby("centreline_id")["hour_trunc"].nunique()

for min_hours, label in [(72, "3 days"), (168, "7 days")]:
    valid = hours_by_id[hours_by_id >= min_hours].index
    kept_rows = merged[merged["centreline_id"].isin(valid)]
    lost_rows = total_rows - len(kept_rows)
    lost_locs = total_locs - len(valid)
    print(f"\nFilter < {label} ({min_hours}h):")
    print(f"  Lost segments: {lost_locs} ({lost_locs/total_locs*100:.1f}%) -> {len(valid)} remaining")
    print(f"  Lost rows:     {lost_rows:,} ({lost_rows/total_rows*100:.1f}%) -> {len(kept_rows):,} remaining")
