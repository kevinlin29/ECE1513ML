# Traffic Congestion Prediction Near University of Toronto

ECE1513 Course Project -- University of Toronto

## Description

This project develops machine learning models to predict traffic speed and congestion levels on streets within approximately 1.5 km of the University of Toronto St. George campus. Using midblock vehicle speed, volume, and classification count data (2020--2024) from the City of Toronto Open Data Portal and hourly weather observations from Environment Canada, we train and evaluate progressively more powerful models -- from historical averages to linear regression, random forests, and XGBoost -- to forecast traffic conditions across 104 street segments (~105K data points). The goal is to identify key factors contributing to congestion and provide actionable predictions for commuters and city planners in the campus neighbourhood.

## Setup

1. Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebooks in order:

```
01 - Data Collection
02 - Data Preprocessing & Feature Engineering
03 - Exploratory Data Analysis
04 - Model Training & Evaluation
05 - Results & Visualization
```

To launch the notebook server:

```bash
jupyter notebook notebooks/
```

## Repository Structure

```
traffic-congestion-dvp/
├── data/
│   ├── raw/                # Original datasets (not committed to version control)
│   └── processed/          # Cleaned and feature-engineered datasets
├── notebooks/              # Jupyter notebooks (01 through 05)
├── src/                    # Reusable Python modules and utilities
├── results/
│   ├── figures/            # Plots and visualizations
│   └── tables/             # Model performance tables and summaries
├── report/                 # Final project report and presentation materials
├── requirements.txt        # Python dependencies
└── README.md
```

## Dataset Sources

- **City of Toronto Open Data Portal** -- Midblock vehicle speed, volume, and classification counts collected at intersections near the U of T St. George campus (2020--2024). Dataset: "Traffic Volumes - Midblock Vehicle Speed, Volume and Classification Counts." Available at [https://open.toronto.ca/](https://open.toronto.ca/).
- **Environment Canada** -- Historical hourly weather observations (temperature, precipitation, wind, visibility) from Toronto weather stations. Available at [https://climate.weather.gc.ca/](https://climate.weather.gc.ca/).

## Team Members

| Name | Student ID | Email |
|------|-----------|-------|
| TBD  | TBD       | TBD   |
| TBD  | TBD       | TBD   |
| TBD  | TBD       | TBD   |
