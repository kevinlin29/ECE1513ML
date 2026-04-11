"""
Step 1: Download hourly weather data from Environment Canada.
Station: Toronto City Centre (climate_id=6158359)
Years: 2020-2026

Notebook cell 0. Skip this if weather CSVs already exist in data/raw/weather/.
"""
import requests
import time
import random
import os
from config import WEATHER_RAW_DIR

os.makedirs(WEATHER_RAW_DIR, exist_ok=True)

base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

climate_id = "6158359"
years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

print(f"Saving weather data to: {WEATHER_RAW_DIR}")
print("Starting download...\n")

for year in years:
    for month in range(1, 13):
        filename = os.path.join(
            WEATHER_RAW_DIR, f"weather_hourly_{climate_id}_{year}_{month:02d}.csv"
        )
        if os.path.exists(filename):
            print(f"  {year}-{month:02d} already exists, skipping")
            continue

        print(f"  Requesting {year}-{month:02d}...")
        params = {
            "format": "csv",
            "climate_id": climate_id,
            "Year": year,
            "Month": month,
            "Day": 1,
            "time": "LST",
            "timeframe": 1,
            "submit": "Download Data",
        }

        try:
            response = requests.get(
                base_url, headers=headers, params=params, timeout=20
            )
            if response.status_code == 200:
                if response.text.strip().lower().startswith("<!doctype html>"):
                    print(f"    Rate limited, sleeping 10s...")
                    time.sleep(10)
                    continue
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"    OK")
            else:
                print(f"    Failed (HTTP {response.status_code})")
        except Exception as e:
            print(f"    Error: {e}")

        time.sleep(random.uniform(2.0, 4.0))

print("\nDownload complete.")
