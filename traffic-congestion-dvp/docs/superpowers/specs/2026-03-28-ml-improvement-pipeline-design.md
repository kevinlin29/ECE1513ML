# ML Improvement Pipeline Design

**Date:** 2026-03-28
**Goal:** Demonstrate rigorous ML improvement process for ECE1513 project — incremental, measurable steps from baseline to tuned model.

## Context

Traffic speed prediction near U of T St. George campus. Data: City of Toronto midblock speed bins (2020-2025, ~144K rows in U of T area) + Toronto City Centre hourly weather (2020-2026).

Current pipeline: Historical Average → Linear Regression → Random Forest → XGBoost. Single 80/20 temporal split, no CV, no tuning, approximate school calendar, no COVID handling.

## Improvement Steps (measured incrementally)

Each step is evaluated with **5-fold TimeSeriesSplit CV** on MAE, RMSE, R².

| Step | Change | What it demonstrates |
|------|--------|---------------------|
| 0 | Historical average baseline | Floor performance |
| 1 | Existing models on updated data | New data baseline |
| 2 | Add COVID lockdown features | Data quality / domain knowledge |
| 3 | Replace school_in_session with U of T calendar | Feature engineering precision |
| 4 | Hyperparameter tuning (RandomizedSearchCV) | Systematic model selection |
| 5 | Feature ablation study | Feature importance / model understanding |

## New Features

### COVID (preprocessing.py)

Two binary features:

- `is_covid_lockdown` — 1 during major Ontario lockdowns:
  - 2020-03-17 to 2020-06-11 (first emergency)
  - 2020-11-23 to 2021-02-16 (Toronto second wave)
  - 2021-04-03 to 2021-06-11 (third wave stay-at-home)
  - 2022-01-05 to 2022-03-01 (Omicron restrictions)
- `is_covid_era` — 1 for 2020-03-17 to 2022-04-27

### U of T Academic Calendar (preprocessing.py)

Three binary features based on actual Faculty of Arts & Science sessional dates (2020-2026):

- `uoft_classes` — fall or winter classes in session
- `uoft_exams` — fall or winter exam period
- `uoft_reading_week` — fall or winter reading week

Replaces the approximate `school_in_session` (month >= 9 or month <= 6).

## Updated Feature List (train.py)

```python
FEATURE_COLS = [
    # Temporal
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_rush_hour",
    # Weather
    "temp", "humidity", "visibility", "wind_speed", "is_raining", "is_snowing",
    # Calendar
    "is_holiday", "is_long_weekend",
    "uoft_classes", "uoft_exams", "uoft_reading_week",
    # COVID
    "is_covid_lockdown", "is_covid_era",
    # Location
    "location_encoded",
]

FEATURE_GROUPS = {
    "temporal": ["hour_of_day", "day_of_week", "month", "is_weekend", "is_rush_hour"],
    "location": ["location_encoded"],
    "weather": ["temp", "humidity", "visibility", "wind_speed", "is_raining", "is_snowing"],
    "calendar": ["is_holiday", "is_long_weekend", "uoft_classes", "uoft_exams", "uoft_reading_week"],
    "covid": ["is_covid_lockdown", "is_covid_era"],
}
```

## Cross-Validation (train.py)

- `sklearn.model_selection.TimeSeriesSplit(n_splits=5)`
- Used for all evaluation comparisons and during hyperparameter tuning
- Final test set (last 20% chronologically) kept for reporting but CV is the primary metric

### Function: `cross_validate_model(model, X, y, n_splits=5)`

Returns dict with mean/std of MAE, RMSE, R² across folds.

## Hyperparameter Tuning (train.py)

`RandomizedSearchCV` with `n_iter=50`, `scoring="neg_mean_absolute_error"`, `cv=TimeSeriesSplit(n_splits=5)`.

### Random Forest search space

```python
{
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 15, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2", 0.5, 0.8],
}
```

### XGBoost search space

```python
{
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [4, 6, 8, 10, 12],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [0.5, 1.0, 2.0],
}
```

### Function: `tune_model(model, param_distributions, X, y, n_iter=50, n_splits=5)`

Returns best estimator, best params, and CV results DataFrame.

## Feature Ablation (train.py)

Drop-one-group method: train the best model (tuned XGBoost) with all features, then retrain dropping each feature group. Measure CV MAE increase.

### Function: `feature_ablation(model_factory, X, y, feature_groups, n_splits=5)`

Returns DataFrame with group name, full MAE, ablated MAE, delta.

### COVID-specific ablation

Additional comparison: train with vs. without COVID-era data entirely, to quantify lockdown distortion.

## Visualization Additions (evaluate.py)

- `plot_cv_results(cv_results_dict)` — grouped bar chart of mean MAE/RMSE/R² with error bars (std)
- `plot_ablation_results(ablation_df)` — horizontal bar chart of MAE increase per dropped group
- `plot_improvement_steps(steps_df)` — line/bar chart showing MAE improvement across steps 0-4

## Files Modified

| File | Changes |
|------|---------|
| `src/preprocessing.py` | Add COVID features, U of T calendar, remove `school_in_session` |
| `src/train.py` | Update FEATURE_COLS, add FEATURE_GROUPS, add `cross_validate_model()`, `tune_model()`, `feature_ablation()` |
| `src/evaluate.py` | Add `plot_cv_results()`, `plot_ablation_results()`, `plot_improvement_steps()` |
| `src/models.py` | No changes needed |
| `src/data_loader.py` | No changes needed (already updated for new data) |
