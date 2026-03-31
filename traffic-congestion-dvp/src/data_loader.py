"""
data_loader.py — Load traffic, weather, and holiday data for U of T area
congestion prediction using City of Toronto midblock speed data.

Data sources:
  - City of Toronto Open Data: midblock vehicle speed/volume/classification CSV
    (2020-2024 and 2025-2029 files)
  - Environment Canada GeoMet API: hourly weather observations (CSV)
    Station 6158359 — Toronto City Centre (~3.6 km from U of T)
  - Ontario statutory holidays (hardcoded + computed)
"""

import os
import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Speed bin midpoints (km/h)
# ---------------------------------------------------------------------------

SPEED_BIN_COLUMNS = [
    "vol_1_19kph", "vol_20_25kph", "vol_26_30kph", "vol_31_35kph",
    "vol_36_40kph", "vol_41_45kph", "vol_46_50kph", "vol_51_55kph",
    "vol_56_60kph", "vol_61_65kph", "vol_66_70kph", "vol_71_75kph",
    "vol_76_80kph", "vol_81_160kph",
]

SPEED_BIN_MIDPOINTS = np.array([
    10, 22.5, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 100
])


# ---------------------------------------------------------------------------
# Traffic data
# ---------------------------------------------------------------------------

def load_traffic_data(filepaths) -> pd.DataFrame:
    """Load City of Toronto midblock vehicle speed data from one or more CSVs.

    Parses time_start and time_end as datetime columns. When multiple files
    are provided they are concatenated and deduplicated by ``id``.

    Parameters
    ----------
    filepaths : str, Path, or list of str/Path
        Path(s) to traffic CSV file(s), e.g.::

            "data/raw/svc_raw_data_speed_2020_2024.csv"
            ["data/raw/svc_raw_data_speed_2020_2024.csv",
             "data/raw/svc_raw_data_speed_2025_2029.csv"]

    Returns
    -------
    pd.DataFrame
        Raw traffic dataframe with parsed datetime columns.
    """
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]

    dfs = []
    for fp in filepaths:
        fp = Path(fp)
        chunk = pd.read_csv(fp, parse_dates=["time_start", "time_end"])
        print(f"  Loaded {chunk.shape[0]:,} rows from {fp.name}")
        dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)

    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
        dupes = before - len(df)
        if dupes:
            print(f"  Removed {dupes:,} duplicate rows by id")

    print(f"Loaded traffic data: {df.shape[0]:,} rows, {df.shape[1]} columns "
          f"from {len(filepaths)} file(s)")
    return df


def filter_uoft_area(df: pd.DataFrame, radius_km: float = 1.5) -> pd.DataFrame:
    """Filter traffic data to locations near U of T St George campus.

    Uses a simple lat/lon bounding box centred on (43.66, -79.40):
        lat: 43.645 to 43.675
        lon: -79.415 to -79.385

    Parameters
    ----------
    df : pd.DataFrame
        Traffic dataframe with ``latitude`` and ``longitude`` columns.
    radius_km : float
        Approximate radius in km (used for documentation; the box is hardcoded).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only rows within the bounding box.
    """
    lat_min, lat_max = 43.645, 43.675
    lon_min, lon_max = -79.415, -79.385

    mask = (
        (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max) &
        (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
    )
    filtered = df[mask].copy().reset_index(drop=True)

    n_locations = filtered["location_name"].nunique() if "location_name" in filtered.columns else "?"
    print(f"filter_uoft_area: {len(filtered):,}/{len(df):,} rows kept "
          f"({n_locations} unique locations)")
    return filtered


def compute_average_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted average speed from speed-bin volume columns.

    For each row, average speed = sum(midpoint_i * count_i) / sum(count_i).
    Rows where total count is 0 get NaN for avg_speed.

    Also adds a ``total_volume`` column with the sum of all bin counts.

    Parameters
    ----------
    df : pd.DataFrame
        Traffic dataframe containing the 14 speed-bin volume columns.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with ``avg_speed`` and ``total_volume`` columns added.
    """
    df = df.copy()

    bin_values = df[SPEED_BIN_COLUMNS].fillna(0).values  # (n_rows, 14)
    total_volume = bin_values.sum(axis=1)
    weighted_sum = (bin_values * SPEED_BIN_MIDPOINTS).sum(axis=1)

    df["total_volume"] = total_volume
    df["avg_speed"] = np.where(total_volume > 0, weighted_sum / total_volume, np.nan)

    print(f"compute_average_speed: mean={df['avg_speed'].mean():.1f} km/h, "
          f"median={df['avg_speed'].median():.1f} km/h")
    return df


# ---------------------------------------------------------------------------
# Weather data
# ---------------------------------------------------------------------------

def load_weather_data(weather_dir: str | Path) -> pd.DataFrame:
    """Load all monthly weather CSVs from a directory and concatenate.

    Expects files named like ``toronto_city_centre_YYYY_MM.csv`` from the
    Environment Canada GeoMet API (Climate ID 6158359, Toronto City Centre,
    ~3.6 km from U of T campus).

    Parameters
    ----------
    weather_dir : str or Path
        Directory containing monthly weather CSV files.

    Returns
    -------
    pd.DataFrame
        Concatenated weather dataframe with parsed LOCAL_DATE and selected
        columns: LOCAL_DATE, TEMP, PRECIP_AMOUNT, RELATIVE_HUMIDITY,
        VISIBILITY, WIND_SPEED, WEATHER_ENG_DESC.
    """
    weather_dir = Path(weather_dir)
    csv_files = sorted(weather_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {weather_dir}")

    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, ignore_index=True)

    # Parse datetime
    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"], errors="coerce")

    # Select relevant columns
    keep_cols = [
        "LOCAL_DATE", "TEMP", "PRECIP_AMOUNT", "RELATIVE_HUMIDITY",
        "VISIBILITY", "WIND_SPEED", "WEATHER_ENG_DESC",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    print(f"Loaded weather data: {len(df):,} rows from {len(csv_files)} files in {weather_dir.name}/")
    return df


# ---------------------------------------------------------------------------
# Holidays
# ---------------------------------------------------------------------------

def _easter_date(year: int) -> datetime.date:
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return datetime.date(year, month, day + 1)


def get_holidays(years) -> list[datetime.date]:
    """Return Ontario statutory/public holiday dates for the given year(s).

    Includes: New Year's Day, Family Day, Good Friday, Victoria Day,
    Canada Day, Civic Holiday, Labour Day, Thanksgiving, Christmas Day,
    Boxing Day.

    Parameters
    ----------
    years : int or iterable of int
        One or more calendar years.

    Returns
    -------
    list of datetime.date
        Sorted list of holiday dates.
    """
    if isinstance(years, int):
        years = [years]

    holidays = []
    for year in years:
        # New Year's Day
        holidays.append(datetime.date(year, 1, 1))

        # Family Day — 3rd Monday of February
        feb1 = datetime.date(year, 2, 1)
        first_monday = feb1 + datetime.timedelta(days=(7 - feb1.weekday()) % 7)
        holidays.append(first_monday + datetime.timedelta(weeks=2))

        # Good Friday — 2 days before Easter Sunday
        easter = _easter_date(year)
        holidays.append(easter - datetime.timedelta(days=2))

        # Victoria Day — Monday before May 25
        may25 = datetime.date(year, 5, 25)
        offset = (may25.weekday() - 0) % 7
        holidays.append(may25 - datetime.timedelta(days=offset if offset else 7))

        # Canada Day
        holidays.append(datetime.date(year, 7, 1))

        # Civic Holiday — 1st Monday of August
        aug1 = datetime.date(year, 8, 1)
        first_monday_aug = aug1 + datetime.timedelta(days=(7 - aug1.weekday()) % 7)
        if first_monday_aug.month != 8:
            first_monday_aug = aug1
        holidays.append(first_monday_aug)

        # Labour Day — 1st Monday of September
        sep1 = datetime.date(year, 9, 1)
        first_monday_sep = sep1 + datetime.timedelta(days=(7 - sep1.weekday()) % 7)
        if first_monday_sep.month != 9:
            first_monday_sep = sep1
        holidays.append(first_monday_sep)

        # Thanksgiving — 2nd Monday of October
        oct1 = datetime.date(year, 10, 1)
        first_monday_oct = oct1 + datetime.timedelta(days=(7 - oct1.weekday()) % 7)
        if first_monday_oct.month != 10:
            first_monday_oct = oct1
        holidays.append(first_monday_oct + datetime.timedelta(weeks=1))

        # Christmas Day
        holidays.append(datetime.date(year, 12, 25))

        # Boxing Day
        holidays.append(datetime.date(year, 12, 26))

    return sorted(holidays)


# ---------------------------------------------------------------------------
# Main — quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("U of T Area Congestion Prediction — Data Loader Demo")
    print("=" * 60)

    # Show holidays for the data range
    for yr in (2020, 2021, 2022, 2023, 2024, 2025, 2026):
        hols = get_holidays(yr)
        print(f"\nOntario holidays in {yr}:")
        for h in hols:
            print(f"  {h}  ({h.strftime('%A')})")

    # Attempt to load traffic data if it exists
    traffic_files = [
        RAW_DATA_DIR / "svc_raw_data_speed_2020_2024.csv",
        RAW_DATA_DIR / "svc_raw_data_speed_2025_2029.csv",
    ]
    traffic_files = [f for f in traffic_files if f.exists()]
    if traffic_files:
        traffic_df = load_traffic_data(traffic_files)
        traffic_df = filter_uoft_area(traffic_df)
        traffic_df = compute_average_speed(traffic_df)
        print(f"\nTraffic data preview:\n{traffic_df.head()}")
    else:
        print(f"\nNo traffic files found — skipping.")

    # Attempt to load weather data if directory exists
    weather_dir = RAW_DATA_DIR / "weather"
    if weather_dir.exists():
        weather_df = load_weather_data(weather_dir)
        print(f"\nWeather data preview:\n{weather_df.head()}")
    else:
        print(f"\nNo weather directory found at {weather_dir} — skipping.")
