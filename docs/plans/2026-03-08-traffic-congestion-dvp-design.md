# Traffic Congestion Prediction Near University of Toronto — Design Document

## Problem Statement

Predict traffic speed, congestion level, and travel time on streets within ~1.5km of the University of Toronto St. George campus using machine learning.

Traffic congestion depends on complex, nonlinear interactions between time of day, day of week, weather, holidays, and segment-specific patterns. Simple historical averages cannot capture these interactions — ML models can learn them from data.

The project follows an improvement narrative: naive baseline (historical averages) → Linear Regression → Random Forest → XGBoost, showing measurable improvement at each step.

## Data Sources & Features

### Primary Data
- City of Toronto Open Data — "Traffic Volumes - Midblock Vehicle Speed, Volume and Classification Counts" dataset
  - 104 street segments within ~1.5km of U of T St. George campus
  - ~105K rows of speed data (2020-2024), 15-minute intervals
  - Speed stored as vehicle counts per speed bin (1-19, 20-25, ..., 81-160 km/h)
  - Covers: Bloor, College, Spadina, Avenue Rd, Yonge, Dundas, Harbord, Beverley, Huron, Kensington area
  - Average speed per interval computed from bin midpoints

### Supplementary Data
- Environment Canada historical hourly weather data (2020-2024) from Toronto Pearson (Station ID 51459, Climate ID 6158731)
  - Downloaded via GeoMet API: temperature, precipitation, humidity, visibility, wind speed/direction, weather description
- Canadian public holiday calendar (Ontario statutory holidays)

### Features

| Category | Features |
|---|---|
| Temporal | Hour of day, day of week, month, is_weekend, is_rush_hour |
| Weather | Temperature, precipitation, snow, visibility |
| Calendar | Is_holiday, is_long_weekend, school_in_session |
| Segment | Segment ID (categorical) |

### Target Variables
- **Primary:** Average speed per segment per time interval
- **Derived:** Congestion level (binned from speed: free flow / moderate / heavy / gridlock)
- **Derived:** Travel time per segment (distance / predicted speed)

## Model Pipeline

### Preprocessing
1. Load and merge traffic data with weather and calendar features
2. Handle missing values (interpolation for gaps in traffic data)
3. Normalize numerical features
4. Encode categorical variables (segment ID, day of week)
5. Train/test split — temporal split (e.g., train on Jan–Oct, test on Nov–Dec) to avoid data leakage

### Models (Progressive Improvement)

| Step | Model | Purpose |
|---|---|---|
| Baseline | Historical averages (mean speed by segment + hour + day of week) | Naive reference |
| Step 1 | Linear Regression | Simple ML baseline |
| Step 2 | Random Forest | Capture nonlinear patterns |
| Step 3 | XGBoost | Best tabular performance |

### Evaluation Metrics
- MAE (Mean Absolute Error) — most interpretable ("off by X km/h")
- RMSE (Root Mean Squared Error) — penalizes large errors
- R² — overall fit
- Per-congestion-level accuracy (for classification task)

### Key Analyses
- Feature importance (which factors drive congestion most?)
- Error analysis by time of day (does the model struggle during transitions?)
- Segment-level performance comparison
- Predicted vs actual speed plots
- Congestion heatmap by hour/day of week

## Repository Structure

```
traffic-congestion-dvp/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── figures/
│   └── tables/
│
└── report/
    └── final_report.pdf
```

## Deliverables
- Git repo with documented code
- README with setup/usage instructions
- requirements.txt
- Demo notebook (notebook 05)
- Final report (5 pages, mandatory template, includes Teamwork/Consent/AI attestations)
- 5-minute presentation (April 6–10)
