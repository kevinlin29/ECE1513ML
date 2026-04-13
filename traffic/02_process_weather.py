"""
Step 2: Merge raw weather CSVs and engineer features.
Notebook cells 1 + 2. Handles both GeoMet API and bulk-download column formats.
"""
import pandas as pd
import numpy as np
import glob
import os
from config import WEATHER_RAW_DIR, PROCESSED_DIR, WEATHER_MERGED_RAW, WEATHER_PROCESSED

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── 1. Merge all weather CSVs ────────────────────────────────
all_files = sorted(glob.glob(os.path.join(WEATHER_RAW_DIR, "*.csv")))
print(f"Found {len(all_files)} weather CSV files")

df_list = []
for f in all_files:
    temp = pd.read_csv(f, low_memory=False)
    df_list.append(temp)

raw = pd.concat(df_list, ignore_index=True)
raw.to_csv(WEATHER_MERGED_RAW, index=False)
print(f"Merged raw weather saved ({len(raw)} rows): {WEATHER_MERGED_RAW}")

# ── 2. Detect column format and normalize ────────────────────
# GeoMet API format uses: LOCAL_DATE, TEMP, PRECIP_AMOUNT, WIND_SPEED, etc.
# Bulk download format uses: Date/Time (LST), Temp (C), Precip. Amount (mm), etc.

if "LOCAL_DATE" in raw.columns:
    print("Detected GeoMet API format, renaming columns...")
    rename_map = {
        "LOCAL_DATE": "Date/Time (LST)",
        "LOCAL_YEAR": "Year",
        "LOCAL_MONTH": "Month",
        "LOCAL_DAY": "Day",
        "LOCAL_HOUR": "Time (LST)",
        "TEMP": "Temp (°C)",
        "PRECIP_AMOUNT": "Precip. Amount (mm)",
        "WIND_SPEED": "Wind Spd (km/h)",
        "WIND_DIRECTION": "Wind Dir (10s deg)",
        "VISIBILITY": "Visibility (km)",
        "RELATIVE_HUMIDITY": "Rel Hum (%)",
        "WEATHER_ENG_DESC": "Weather",
    }
    raw = raw.rename(columns=rename_map)

# ── 3. Filter core columns ───────────────────────────────────
cols_to_keep = [
    "Date/Time (LST)", "Year", "Month", "Day", "Time (LST)",
    "Temp (°C)", "Precip. Amount (mm)", "Wind Spd (km/h)",
    "Wind Dir (10s deg)", "Visibility (km)", "Rel Hum (%)", "Weather",
]
clean = raw[[c for c in cols_to_keep if c in raw.columns]].copy()

# ── 4. Standardize timeline ──────────────────────────────────
clean["Date/Time (LST)"] = pd.to_datetime(clean["Date/Time (LST)"])
clean = clean.sort_values("Date/Time (LST)").reset_index(drop=True)
clean = clean.drop_duplicates(subset=["Date/Time (LST)"], keep="first")

# ── 5. Fill missing values ───────────────────────────────────
continuous = ["Temp (°C)", "Wind Spd (km/h)", "Visibility (km)", "Rel Hum (%)", "Wind Dir (10s deg)"]
for col in continuous:
    if col in clean.columns:
        clean[col] = clean[col].ffill()

if "Precip. Amount (mm)" in clean.columns:
    clean["Precip. Amount (mm)"] = clean["Precip. Amount (mm)"].fillna(0)

# ── 6. Cyclical wind direction encoding ──────────────────────
if "Wind Dir (10s deg)" in clean.columns:
    wind_rad = clean["Wind Dir (10s deg)"] * 10 * (np.pi / 180)
    clean["Wind_Sin"] = np.sin(wind_rad)
    clean["Wind_Cos"] = np.cos(wind_rad)
    clean = clean.drop(columns=["Wind Dir (10s deg)"])

# ── 7. Weather state one-hot flags ───────────────────────────
if "Weather" in clean.columns:
    clean["is_raining"] = clean["Weather"].str.contains("Rain|Drizzle", case=False, na=False).astype(int)
    clean["is_snowing"] = clean["Weather"].str.contains("Snow|Ice", case=False, na=False).astype(int)
    clean["is_foggy"] = clean["Weather"].str.contains("Fog|Mist", case=False, na=False).astype(int)
    clean = clean.drop(columns=["Weather"])

# ── 8. Save ──────────────────────────────────────────────────
clean.to_csv(WEATHER_PROCESSED, index=False)
print(f"\nProcessed weather saved ({len(clean)} rows): {WEATHER_PROCESSED}")
print(clean.head())
print(f"\nColumns: {list(clean.columns)}")
