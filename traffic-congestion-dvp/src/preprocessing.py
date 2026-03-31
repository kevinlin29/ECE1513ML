"""
preprocessing.py — Data cleaning, feature engineering, and dataset preparation
for U of T area congestion prediction using City of Toronto midblock speed data.

Typical pipeline:
    traffic_df = load_traffic_data(["data/raw/svc_raw_data_speed_2020_2024.csv",
                                     "data/raw/svc_raw_data_speed_2025_2029.csv"])
    traffic_df = filter_uoft_area(traffic_df)
    traffic_df = compute_average_speed(traffic_df)
    traffic_df = clean_traffic_data(traffic_df)
    weather_df = load_weather_data("data/raw/weather")
    weather_df = clean_weather_data(weather_df)
    merged     = merge_traffic_weather(traffic_df, weather_df)
    featured   = engineer_features(merged)
    featured   = create_congestion_labels(featured, speed_col="avg_speed")
    train, test = temporal_train_test_split(featured, date_col="time_start")
"""

import datetime

import numpy as np
import pandas as pd

try:
    from src.data_loader import get_holidays
except ImportError:
    from data_loader import get_holidays


# ---------------------------------------------------------------------------
# U of T St. George sessional dates (2020-2026)
# Source: U of T Faculty of Arts & Science sessional dates
# ---------------------------------------------------------------------------

UOFT_CALENDAR = {
    2020: {
        "fall_classes":  ("2020-09-09", "2020-12-08"),
        "fall_exams":    ("2020-12-10", "2020-12-22"),
        "fall_reading":  ("2020-11-09", "2020-11-13"),
        "winter_classes": ("2020-01-06", "2020-03-13"),
        "winter_exams":  ("2020-04-01", "2020-04-30"),
        "winter_reading": ("2020-02-17", "2020-02-21"),
    },
    2021: {
        "fall_classes":  ("2021-09-09", "2021-12-07"),
        "fall_exams":    ("2021-12-09", "2021-12-21"),
        "fall_reading":  ("2021-11-08", "2021-11-12"),
        "winter_classes": ("2021-01-11", "2021-04-09"),
        "winter_exams":  ("2021-04-12", "2021-04-30"),
        "winter_reading": ("2021-02-15", "2021-02-19"),
    },
    2022: {
        "fall_classes":  ("2022-09-08", "2022-12-06"),
        "fall_exams":    ("2022-12-08", "2022-12-20"),
        "fall_reading":  ("2022-11-07", "2022-11-11"),
        "winter_classes": ("2022-01-10", "2022-04-08"),
        "winter_exams":  ("2022-04-11", "2022-04-29"),
        "winter_reading": ("2022-02-21", "2022-02-25"),
    },
    2023: {
        "fall_classes":  ("2023-09-07", "2023-12-05"),
        "fall_exams":    ("2023-12-07", "2023-12-20"),
        "fall_reading":  ("2023-11-06", "2023-11-10"),
        "winter_classes": ("2023-01-09", "2023-04-05"),
        "winter_exams":  ("2023-04-10", "2023-04-28"),
        "winter_reading": ("2023-02-20", "2023-02-24"),
    },
    2024: {
        "fall_classes":  ("2024-09-03", "2024-12-03"),
        "fall_exams":    ("2024-12-05", "2024-12-19"),
        "fall_reading":  ("2024-11-04", "2024-11-08"),
        "winter_classes": ("2024-01-08", "2024-04-05"),
        "winter_exams":  ("2024-04-08", "2024-04-26"),
        "winter_reading": ("2024-02-19", "2024-02-23"),
    },
    2025: {
        "fall_classes":  ("2025-09-04", "2025-12-03"),
        "fall_exams":    ("2025-12-05", "2025-12-19"),
        "fall_reading":  ("2025-11-10", "2025-11-14"),
        "winter_classes": ("2025-01-06", "2025-04-04"),
        "winter_exams":  ("2025-04-07", "2025-04-25"),
        "winter_reading": ("2025-02-17", "2025-02-21"),
    },
    2026: {
        "winter_classes": ("2026-01-05", "2026-04-03"),
        "winter_exams":  ("2026-04-06", "2026-04-24"),
        "winter_reading": ("2026-02-16", "2026-02-20"),
    },
}


# ---------------------------------------------------------------------------
# Ontario COVID-19 major restriction periods affecting Toronto traffic
# ---------------------------------------------------------------------------

COVID_LOCKDOWNS = [
    ("2020-03-17", "2020-06-11"),  # First state of emergency + full lockdown
    ("2020-11-23", "2021-02-16"),  # Toronto second-wave lockdown (Grey zone)
    ("2021-04-03", "2021-06-11"),  # Third-wave province-wide stay-at-home
    ("2022-01-05", "2022-03-01"),  # Omicron capacity restrictions
]

COVID_ERA_START = "2020-03-17"
COVID_ERA_END = "2022-04-27"


# ---------------------------------------------------------------------------
# U of T calendar & COVID helpers
# ---------------------------------------------------------------------------

def _build_date_ranges(calendar_dict, key_prefix):
    """Build a set of datetime.date objects from UOFT_CALENDAR entries."""
    dates = set()
    for _year, periods in calendar_dict.items():
        for key, (start, end) in periods.items():
            if key.startswith(key_prefix):
                s = pd.Timestamp(start).date()
                e = pd.Timestamp(end).date()
                d = s
                while d <= e:
                    dates.add(d)
                    d += datetime.timedelta(days=1)
    return dates


def add_uoft_calendar_features(df, datetime_col="time_start"):
    """Add U of T academic calendar features using actual sessional dates.

    New columns:
        uoft_classes — 1 if fall or winter classes are in session
        uoft_exams   — 1 if fall or winter exam period
        uoft_reading_week — 1 if fall or winter reading week

    Parameters
    ----------
    df : pd.DataFrame
    datetime_col : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    dt_dates = df[datetime_col].dt.date

    classes_dates = (_build_date_ranges(UOFT_CALENDAR, "fall_classes") |
                     _build_date_ranges(UOFT_CALENDAR, "winter_classes"))
    exam_dates = (_build_date_ranges(UOFT_CALENDAR, "fall_exams") |
                  _build_date_ranges(UOFT_CALENDAR, "winter_exams"))
    reading_dates = (_build_date_ranges(UOFT_CALENDAR, "fall_reading") |
                     _build_date_ranges(UOFT_CALENDAR, "winter_reading"))

    df["uoft_classes"] = dt_dates.isin(classes_dates).astype(int)
    df["uoft_exams"] = dt_dates.isin(exam_dates).astype(int)
    df["uoft_reading_week"] = dt_dates.isin(reading_dates).astype(int)

    print(f"add_uoft_calendar_features: {df['uoft_classes'].sum():,} class rows, "
          f"{df['uoft_exams'].sum():,} exam rows, "
          f"{df['uoft_reading_week'].sum():,} reading-week rows")
    return df


def add_covid_features(df, datetime_col="time_start"):
    """Add COVID-19 restriction flags.

    New columns:
        is_covid_lockdown — 1 during major Ontario lockdown periods
        is_covid_era      — 1 for the broad 2020-03 to 2022-04 period

    Parameters
    ----------
    df : pd.DataFrame
    datetime_col : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    dt = df[datetime_col]

    lockdown_mask = pd.Series(False, index=df.index)
    for start, end in COVID_LOCKDOWNS:
        lockdown_mask |= (dt >= start) & (dt <= end)
    df["is_covid_lockdown"] = lockdown_mask.astype(int)

    df["is_covid_era"] = ((dt >= COVID_ERA_START) & (dt <= COVID_ERA_END)).astype(int)

    n_lockdown = df["is_covid_lockdown"].sum()
    n_era = df["is_covid_era"].sum()
    print(f"add_covid_features: {n_lockdown:,} lockdown rows, {n_era:,} COVID-era rows")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean traffic data after average speed has been computed.

    Steps:
        1. Drop fully duplicated rows.
        2. Ensure time_start is parsed as datetime; drop rows that failed.
        3. Remove rows where total_volume is 0 or NaN.
        4. Remove rows where avg_speed is NaN.
        5. Ensure correct dtypes for numeric columns.
        6. Sort by time_start.

    Parameters
    ----------
    df : pd.DataFrame
        Traffic dataframe with ``time_start``, ``avg_speed``, and
        ``total_volume`` columns (output of ``compute_average_speed``).

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the dataframe, sorted by time_start.
    """
    df = df.copy()
    initial_rows = len(df)

    # 1. Duplicates
    df = df.drop_duplicates()

    # 2. Datetime
    df["time_start"] = pd.to_datetime(df["time_start"], errors="coerce")
    df = df.dropna(subset=["time_start"])

    # 3. Remove rows with zero or missing total volume
    df["total_volume"] = pd.to_numeric(df["total_volume"], errors="coerce")
    df = df[df["total_volume"] > 0]

    # 4. Remove rows with missing avg_speed
    df["avg_speed"] = pd.to_numeric(df["avg_speed"], errors="coerce")
    df = df.dropna(subset=["avg_speed"])

    # 5. Ensure numeric dtypes for latitude/longitude
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Sort
    df = df.sort_values("time_start").reset_index(drop=True)

    removed = initial_rows - len(df)
    print(f"clean_traffic_data: kept {len(df):,}/{initial_rows:,} rows (removed {removed:,})")
    return df


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean GeoMet API weather data.

    Steps:
        1. Ensure LOCAL_DATE is parsed as datetime; drop rows that failed.
        2. Rename columns to short names.
        3. Coerce numeric columns and interpolate short gaps (up to 3 hours).
        4. Sort by datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Raw weather dataframe from ``load_weather_data`` with columns
        LOCAL_DATE, TEMP, PRECIP_AMOUNT, RELATIVE_HUMIDITY, VISIBILITY,
        WIND_SPEED, WEATHER_ENG_DESC.

    Returns
    -------
    pd.DataFrame
        Cleaned weather dataframe with short column names and a
        ``datetime`` column.
    """
    df = df.copy()

    # 1. Parse datetime
    df["LOCAL_DATE"] = pd.to_datetime(df["LOCAL_DATE"], errors="coerce")
    df = df.dropna(subset=["LOCAL_DATE"])

    # 2. Rename to short names
    rename_map = {
        "LOCAL_DATE": "datetime",
        "TEMP": "temp",
        "PRECIP_AMOUNT": "precip",
        "RELATIVE_HUMIDITY": "humidity",
        "VISIBILITY": "visibility",
        "WIND_SPEED": "wind_speed",
        "WEATHER_ENG_DESC": "weather_desc",
    }
    cols_present = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=cols_present)

    # Keep only renamed columns
    keep = [v for v in rename_map.values() if v in df.columns]
    df = df[keep]

    # 3. Derive precipitation indicators from weather_desc (PRECIP_AMOUNT is
    #    empty for hourly data at this station)
    if "weather_desc" in df.columns:
        desc_lower = df["weather_desc"].fillna("").str.lower()
        df["is_raining"] = desc_lower.str.contains("rain|drizzle|thunderstorm").astype(int)
        df["is_snowing"] = desc_lower.str.contains("snow|ice|freezing").astype(int)

    # Drop precip column (always NaN for this station)
    df = df.drop(columns=["precip"], errors="ignore")

    # 4. Coerce numeric and interpolate
    numeric_cols = ["temp", "humidity", "visibility", "wind_speed"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)

    numeric_present = [c for c in numeric_cols if c in df.columns]
    df[numeric_present] = df[numeric_present].interpolate(method="linear", limit=3)

    print(f"clean_weather_data: {len(df):,} rows, columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_traffic_weather(traffic_df: pd.DataFrame,
                          weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge traffic and weather dataframes on nearest hour.

    Uses ``pd.merge_asof`` to match each traffic observation to the
    closest weather observation within a 2-hour tolerance.

    Parameters
    ----------
    traffic_df : pd.DataFrame
        Cleaned traffic dataframe with ``time_start`` datetime column.
    weather_df : pd.DataFrame
        Cleaned weather dataframe with ``datetime`` column.

    Returns
    -------
    pd.DataFrame
        Merged dataframe sorted by time_start.
    """
    traffic_df = traffic_df.sort_values("time_start").copy()
    weather_df = weather_df.sort_values("datetime").copy()

    merged = pd.merge_asof(
        traffic_df,
        weather_df,
        left_on="time_start",
        right_on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("2h"),
    )

    # Drop the redundant weather datetime column
    if "datetime" in merged.columns:
        merged = merged.drop(columns=["datetime"])

    print(f"merge_traffic_weather: {len(merged):,} rows after merge")
    return merged


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame,
                      datetime_col: str = "time_start") -> pd.DataFrame:
    """Add temporal, calendar, and COVID features to the dataframe.

    New columns:
        hour_of_day, day_of_week (0=Mon), month, is_weekend,
        is_rush_hour (7-9 AM or 4-7 PM), is_holiday, is_long_weekend,
        uoft_classes, uoft_exams, uoft_reading_week,
        is_covid_lockdown, is_covid_era.

    Parameters
    ----------
    df : pd.DataFrame
    datetime_col : str
        Name of the datetime column (default ``"time_start"``).

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with new feature columns appended.
    """
    df = df.copy()
    dt = df[datetime_col]

    # --- Basic temporal features ------------------------------------------
    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek          # 0 = Monday
    df["month"] = dt.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Rush hour: 7-9 AM (hours 7, 8) or 4-7 PM (hours 16, 17, 18)
    df["is_rush_hour"] = df["hour_of_day"].apply(
        lambda h: int(h in (7, 8) or h in (16, 17, 18))
    )

    # --- Holidays ---------------------------------------------------------
    years = sorted(dt.dt.year.dropna().unique())
    all_holidays: set[datetime.date] = set(get_holidays([int(y) for y in years]))

    df["is_holiday"] = dt.dt.date.isin(all_holidays).astype(int)

    # Long weekend: the Friday before or Monday after a holiday that falls
    # on Mon/Fri, or any holiday-adjacent weekend day.
    long_weekend_dates: set[datetime.date] = set()
    for h in all_holidays:
        long_weekend_dates.add(h)
        wd = h.weekday()
        if wd == 0:  # Monday holiday -> Sat & Sun before
            long_weekend_dates.update([h - datetime.timedelta(days=1),
                                       h - datetime.timedelta(days=2)])
        elif wd == 4:  # Friday holiday -> Sat & Sun after
            long_weekend_dates.update([h + datetime.timedelta(days=1),
                                       h + datetime.timedelta(days=2)])

    df["is_long_weekend"] = dt.dt.date.isin(long_weekend_dates).astype(int)

    # --- U of T academic calendar (actual sessional dates) -----------------
    df = add_uoft_calendar_features(df, datetime_col=datetime_col)

    # --- COVID-19 restriction flags ----------------------------------------
    df = add_covid_features(df, datetime_col=datetime_col)

    print(f"engineer_features: added temporal/calendar/COVID features "
          f"({len(all_holidays)} holidays across {len(years)} year(s))")
    return df


# ---------------------------------------------------------------------------
# Labels and derived columns
# ---------------------------------------------------------------------------

def create_congestion_labels(df: pd.DataFrame,
                             speed_col: str = "avg_speed") -> pd.DataFrame:
    """Bin speeds into four congestion levels for city streets.

    Thresholds (city street speeds, not highway):
        - gridlock:   speed < 20 km/h
        - heavy:      20 <= speed < 35
        - moderate:   35 <= speed < 50
        - free_flow:  speed >= 50

    A new ``congestion`` column (categorical) is added.

    Parameters
    ----------
    df : pd.DataFrame
    speed_col : str
        Column containing average speed in km/h.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    bins = [-np.inf, 20, 35, 50, np.inf]
    labels = ["gridlock", "heavy", "moderate", "free_flow"]

    df["congestion"] = pd.cut(
        df[speed_col], bins=bins, labels=labels, right=False
    )

    # Print distribution for a quick sanity check
    dist = df["congestion"].value_counts()
    print("Congestion label distribution:")
    for label in labels:
        count = dist.get(label, 0)
        print(f"  {label:>10s}: {count}")

    return df


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def add_lag_features(df, target_col="avg_speed", location_col="location_encoded",
                     time_col="time_start", lags=(1, 4, 8),
                     rolling_windows=(4,)):
    """Add lagged speed and rolling average features per location.

    Creates columns like ``speed_lag_1`` (previous 15-min interval),
    ``speed_lag_4`` (1 hour ago), ``speed_rolling_4`` (1-hour rolling mean).

    Only creates lags within continuous segments (gap <= 30 min).
    Rows where lags are unavailable get NaN (should be imputed downstream).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    location_col : str
    time_col : str
    lags : tuple of int
        Lag offsets in number of observations (1 = 15 min, 4 = 1 hour).
    rolling_windows : tuple of int
        Rolling mean window sizes.

    Returns
    -------
    pd.DataFrame
        Copy with new lag/rolling columns appended.
    """
    df = df.sort_values([location_col, time_col]).copy()

    # Compute time diff within each location to detect gaps
    df["_tdiff"] = df.groupby(location_col)[time_col].diff().dt.total_seconds() / 60.0

    for lag in lags:
        col = f"speed_lag_{lag}"
        df[col] = df.groupby(location_col)[target_col].shift(lag)
        # Invalidate where there's a gap > 30 min in any of the shifted positions
        # Simple approach: invalidate if the current _tdiff is > 30 (break in continuity)
        # For larger lags, check cumulative continuity
        if lag == 1:
            df.loc[df["_tdiff"] > 30, col] = np.nan
        else:
            # Invalidate if any of the last `lag` intervals has a gap
            gap_mask = df.groupby(location_col)["_tdiff"].transform(
                lambda s: s.rolling(lag, min_periods=1).max()
            ) > 30
            df.loc[gap_mask, col] = np.nan

    for w in rolling_windows:
        col = f"speed_rolling_{w}"
        df[col] = df.groupby(location_col)[target_col].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        gap_mask = df.groupby(location_col)["_tdiff"].transform(
            lambda s: s.rolling(w, min_periods=1).max()
        ) > 30
        df.loc[gap_mask, col] = np.nan

    df = df.drop(columns=["_tdiff"])

    lag_cols = [f"speed_lag_{l}" for l in lags] + [f"speed_rolling_{w}" for w in rolling_windows]
    valid = df[lag_cols].notna().all(axis=1).sum()
    print(f"add_lag_features: {len(lag_cols)} lag columns, "
          f"{valid:,}/{len(df):,} rows with all lags valid ({100*valid/len(df):.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Sequence creation for LSTM / Transformer models
# ---------------------------------------------------------------------------

def create_sequences(df, seq_len=8, feature_cols=None, target_col="avg_speed",
                     location_col="location_encoded", time_col="time_start",
                     max_gap_minutes=30):
    """Create sliding-window sequences grouped by location for LSTM/Transformer.

    For each location, observations are sorted by time. Consecutive
    observations within ``max_gap_minutes`` form continuous segments.
    Sliding windows of length ``seq_len`` are extracted from each segment.
    The target is the ``target_col`` value at the *last* time step.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe (already feature-engineered).
    seq_len : int
        Number of time steps per sequence (default 8 = 2 hours at 15-min).
    feature_cols : list[str] or None
        Columns to use as features per time step. If None, uses a default set.
    target_col : str
        Column to predict (default ``"avg_speed"``).
    location_col : str
        Column identifying the location group.
    time_col : str
        Datetime column for ordering and gap detection.
    max_gap_minutes : int
        Maximum gap in minutes between consecutive observations within a
        valid segment (default 30).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - X_seq: shape ``(n_sequences, seq_len, n_features)``, float32
        - y_seq: shape ``(n_sequences,)``, float32
    """
    if feature_cols is None:
        feature_cols = [
            "hour_of_day", "day_of_week", "month", "is_weekend", "is_rush_hour",
            "temp", "humidity", "visibility", "wind_speed", "is_raining", "is_snowing",
            "is_holiday", "is_long_weekend",
            "uoft_classes", "uoft_exams", "uoft_reading_week",
            "is_covid_lockdown", "is_covid_era",
            "latitude", "longitude",
            "location_encoded", "direction_encoded",
        ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    all_X, all_y = [], []

    for _loc, group in df.groupby(location_col):
        group = group.sort_values(time_col).reset_index(drop=True)
        times = group[time_col].values
        feats = group[feature_cols].values.astype(np.float32)
        targets = group[target_col].values.astype(np.float32)

        # Identify break points (gaps > max_gap)
        diffs = np.diff(times).astype("timedelta64[m]").astype(float)
        breaks = np.where(diffs > max_gap_minutes)[0] + 1
        segments = np.split(np.arange(len(group)), breaks)

        for seg_idx in segments:
            if len(seg_idx) < seq_len + 1:
                continue
            seg_feats = feats[seg_idx]
            seg_targets = targets[seg_idx]
            for i in range(seq_len, len(seg_idx)):
                all_X.append(seg_feats[i - seq_len:i])
                all_y.append(seg_targets[i])

    X_seq = np.array(all_X, dtype=np.float32)
    y_seq = np.array(all_y, dtype=np.float32)

    print(f"create_sequences: {len(y_seq):,} sequences of length {seq_len} "
          f"from {df[location_col].nunique()} locations, "
          f"{len(feature_cols)} features per step")
    return X_seq, y_seq


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def temporal_train_test_split(df: pd.DataFrame,
                              date_col: str = "time_start",
                              split_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe chronologically (no shuffling).

    The first *split_ratio* fraction of rows (sorted by *date_col*) becomes
    the training set; the remainder becomes the test set.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
        Datetime column to sort by (default ``"time_start"``).
    split_ratio : float
        Proportion of data to use for training (default 0.8).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (train_df, test_df)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * split_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    split_date = df[date_col].iloc[split_idx]
    print(f"temporal_train_test_split: train={len(train_df):,}, test={len(test_df):,} "
          f"(split at {split_date})")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Main — quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("U of T Area Congestion Prediction — Preprocessing Demo")
    print("=" * 60)

    # Create a small synthetic dataset to demonstrate the pipeline
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-06-01", periods=n, freq="h")

    traffic_df = pd.DataFrame({
        "time_start": dates,
        "time_end": dates + pd.Timedelta("15min"),
        "avg_speed": np.random.normal(loc=35, scale=15, size=n).clip(5, 80),
        "total_volume": np.random.poisson(lam=50, size=n),
        "latitude": 43.66,
        "longitude": -79.40,
        "location_name": "Demo Street",
    })

    weather_df = pd.DataFrame({
        "datetime": dates + pd.Timedelta("15min"),
        "temp": np.random.normal(loc=18, scale=5, size=n),
        "visibility": np.random.uniform(2, 25, size=n),
        "precip": np.random.exponential(scale=0.5, size=n),
        "humidity": np.random.uniform(30, 95, size=n),
        "wind_speed": np.random.uniform(0, 40, size=n),
    })

    print("\n--- Cleaning traffic data ---")
    traffic_clean = clean_traffic_data(traffic_df)

    print("\n--- Merging ---")
    merged = merge_traffic_weather(traffic_clean, weather_df)

    print("\n--- Feature engineering ---")
    featured = engineer_features(merged)

    print("\n--- Congestion labels ---")
    featured = create_congestion_labels(featured, speed_col="avg_speed")

    print("\n--- Train/test split ---")
    train, test = temporal_train_test_split(featured)

    print(f"\nFinal feature columns:\n  {list(featured.columns)}")
    print(f"\nSample rows:\n{featured.head()}")
