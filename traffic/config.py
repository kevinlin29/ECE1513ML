"""
Shared path configuration for the ECE1513 traffic congestion pipeline.
All paths are relative to this file's directory.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # project/

# ── Input (raw) data ──────────────────────────────────────────
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "traffic-congestion-dvp", "data", "raw")
WEATHER_RAW_DIR = os.path.join(RAW_DATA_DIR, "weather")
SPEED_DATA_PATHS = [
    os.path.join(RAW_DATA_DIR, "svc_raw_data_speed_2020_2024.csv"),
    os.path.join(RAW_DATA_DIR, "svc_raw_data_speed_2025_2029.csv"),
]

# ── Processed (intermediate) data ─────────────────────────────
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
WEATHER_MERGED_RAW = os.path.join(PROCESSED_DIR, "merged_weather_raw.csv")
WEATHER_PROCESSED = os.path.join(PROCESSED_DIR, "merged_weather_processed.csv")
TRAFFIC_PROCESSED = os.path.join(PROCESSED_DIR, "clean_traffic_2020_2024.csv")
MERGED_DATA = os.path.join(PROCESSED_DIR, "final_traffic_weather_merged.csv")
LOCATION_SUMMARY = os.path.join(PROCESSED_DIR, "location_data_summary.csv")

# ── Model outputs ─────────────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")

# ── Results ───────────────────────────────────────────────────
RESULTS_DIR = os.path.join(BASE_DIR, "results")
XGB_PREDICTIONS = os.path.join(RESULTS_DIR, "xgb_predictions.csv")
XGB_FEATURE_IMPORTANCE = os.path.join(RESULTS_DIR, "xgb_feature_importance.csv")
