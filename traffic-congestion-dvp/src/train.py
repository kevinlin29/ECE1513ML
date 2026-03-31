"""
train.py — Training pipeline for traffic speed prediction models.

Provides utilities for fitting individual models, training the full model
suite, hyperparameter tuning, cross-validation, feature ablation, and
persisting / loading fitted models via joblib.

Scope: Predict avg_speed on streets near U of T campus using City of
Toronto midblock speed bin data (2020-2025).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

try:
    from src.models import (
        get_baseline_model,
        get_linear_regression,
        get_random_forest,
        get_xgboost,
    )
except ImportError:
    from models import (
        get_baseline_model,
        get_linear_regression,
        get_random_forest,
        get_xgboost,
    )

# ----------------------------------------------------------------------
# Column definitions
# ----------------------------------------------------------------------

TARGET_COL = "avg_speed"

FEATURE_COLS = [
    # Temporal
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    # Weather
    "temp",
    "humidity",
    "visibility",
    "wind_speed",
    "is_raining",
    "is_snowing",
    # Calendar
    "is_holiday",
    "is_long_weekend",
    "uoft_classes",
    "uoft_exams",
    "uoft_reading_week",
    # COVID
    "is_covid_lockdown",
    "is_covid_era",
    # Lag / autoregressive
    "speed_lag_1",
    "speed_lag_4",
    "speed_lag_8",
    "speed_rolling_4",
    # Spatial / categorical
    "latitude",
    "longitude",
    "location_encoded",
    "direction_encoded",
]

# Columns that represent categorical entities and need embeddings in neural nets
EMBEDDING_COLS = {
    "location_encoded": {"name": "location", "embed_dim": 16},
    "direction_encoded": {"name": "direction", "embed_dim": 4},
}

# Everything in FEATURE_COLS that is NOT an embedding column
CONTINUOUS_COLS = [c for c in FEATURE_COLS if c not in EMBEDDING_COLS]

FEATURE_GROUPS = {
    "temporal": ["hour_of_day", "day_of_week", "month", "is_weekend", "is_rush_hour"],
    "spatial": ["latitude", "longitude", "location_encoded", "direction_encoded"],
    "weather": ["temp", "humidity", "visibility", "wind_speed", "is_raining", "is_snowing"],
    "calendar": ["is_holiday", "is_long_weekend", "uoft_classes", "uoft_exams", "uoft_reading_week"],
    "covid": ["is_covid_lockdown", "is_covid_era"],
    "lag": ["speed_lag_1", "speed_lag_4", "speed_lag_8", "speed_rolling_4"],
}

# Columns the HistoricalAverageModel needs from the raw DataFrame (before
# location encoding) so it can group by the human-readable location name.
BASELINE_GROUP_COLS = ["location_name", "hour_of_day", "day_of_week"]

# Hyperparameter search spaces for RandomizedSearchCV
RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 15, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2", 0.5, 0.8],
}

XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [4, 6, 8, 10, 12],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [0.5, 1.0, 2.0],
}


# ----------------------------------------------------------------------
# Label encoding helper
# ----------------------------------------------------------------------

def encode_location(df, encoder=None):
    """Label-encode the ``location_name`` column into ``location_encoded``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``location_name`` column.
    encoder : LabelEncoder or None
        If provided, uses this encoder (e.g. one fitted on the training set).
        If None, a new encoder is fitted on ``df``.

    Returns
    -------
    tuple[pd.DataFrame, LabelEncoder]
        The DataFrame with a new ``location_encoded`` column and the
        encoder used.
    """
    df = df.copy()
    if encoder is None:
        encoder = LabelEncoder()
        df["location_encoded"] = encoder.fit_transform(df["location_name"])
    else:
        df["location_encoded"] = encoder.transform(df["location_name"])
    return df, encoder


def encode_direction(df, encoder=None):
    """Label-encode the ``direction`` column into ``direction_encoded``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``direction`` column.
    encoder : LabelEncoder or None
        If provided, uses this encoder (e.g. one fitted on the training set).
        If None, a new encoder is fitted on ``df``.

    Returns
    -------
    tuple[pd.DataFrame, LabelEncoder]
        The DataFrame with a new ``direction_encoded`` column and the
        encoder used.
    """
    df = df.copy()
    df["direction"] = df["direction"].fillna("UNKNOWN")
    if encoder is None:
        encoder = LabelEncoder()
        df["direction_encoded"] = encoder.fit_transform(df["direction"])
    else:
        # Handle unseen directions at test time
        known = set(encoder.classes_)
        df["direction"] = df["direction"].apply(lambda x: x if x in known else "UNKNOWN")
        df["direction_encoded"] = encoder.transform(df["direction"])
    return df, encoder


# ----------------------------------------------------------------------
# Training functions
# ----------------------------------------------------------------------

def train_model(model, X_train, y_train):
    """Fit a single model on the training data.

    Parameters
    ----------
    model : estimator
        Any object that implements a scikit-learn-compatible ``fit`` method.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : array-like
        Training target values.

    Returns
    -------
    model
        The fitted model (same object, mutated in place).
    """
    model.fit(X_train, y_train)
    return model


def train_all_models(df_train):
    """Train every model in the standard suite and return them with metadata.

    The function handles label encoding for ``location_name`` internally
    and passes the appropriate feature sets to each model.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data containing all ``FEATURE_COLS`` (except
        ``location_encoded``, which is computed here), ``location_name``,
        and ``TARGET_COL``.

    Returns
    -------
    dict
        Keys:
        - ``"models"``: dict[str, estimator] — fitted model instances.
        - ``"label_encoder"``: LabelEncoder — fitted on training locations.
        - ``"feature_cols"``: list[str] — features used by non-baseline models.
    """
    # Encode locations.
    df_train, le = encode_location(df_train)

    y_train = df_train[TARGET_COL]

    # -- Baseline model uses raw location_name for grouping ---------------
    baseline_features = df_train[BASELINE_GROUP_COLS + [
        c for c in FEATURE_COLS if c not in BASELINE_GROUP_COLS
    ]]

    # -- Other models use numeric features only ---------------------------
    X_train = df_train[FEATURE_COLS]

    models = {
        "Historical Average": get_baseline_model(),
        "Linear Regression": get_linear_regression(),
        "Random Forest": get_random_forest(),
        "XGBoost": get_xgboost(),
    }

    fitted = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        if name == "Historical Average":
            fitted[name] = train_model(model, baseline_features, y_train)
        else:
            fitted[name] = train_model(model, X_train, y_train)
        print(f"  {name} training complete.")

    return {
        "models": fitted,
        "label_encoder": le,
        "feature_cols": FEATURE_COLS,
    }


# ----------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------

def cross_validate_model(model, X, y, n_splits=5):
    """Evaluate a model using TimeSeriesSplit cross-validation.

    Parameters
    ----------
    model : estimator
        Unfitted sklearn-compatible model (will be cloned internally).
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : array-like
        Target values.
    n_splits : int
        Number of CV folds (default 5).

    Returns
    -------
    dict
        Keys: ``mae_mean``, ``mae_std``, ``rmse_mean``, ``rmse_std``,
        ``r2_mean``, ``r2_std``, ``fold_results`` (list of per-fold dicts).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_tr = np.asarray(y)[train_idx]
        y_val = np.asarray(y)[val_idx]

        from sklearn.base import clone
        m = clone(model)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_val)

        fold_results.append({
            "fold": fold_idx,
            "mae": mean_absolute_error(y_val, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
            "r2": r2_score(y_val, y_pred),
        })

    maes = [f["mae"] for f in fold_results]
    rmses = [f["rmse"] for f in fold_results]
    r2s = [f["r2"] for f in fold_results]

    result = {
        "mae_mean": np.mean(maes), "mae_std": np.std(maes),
        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
        "fold_results": fold_results,
    }
    print(f"  CV({n_splits}): MAE={result['mae_mean']:.3f}±{result['mae_std']:.3f}, "
          f"RMSE={result['rmse_mean']:.3f}±{result['rmse_std']:.3f}, "
          f"R²={result['r2_mean']:.3f}±{result['r2_std']:.3f}")
    return result


# ----------------------------------------------------------------------
# Hyperparameter tuning
# ----------------------------------------------------------------------

def tune_model(model, param_distributions, X, y, n_iter=50, n_splits=5, n_jobs=-1):
    """Tune hyperparameters using RandomizedSearchCV with TimeSeriesSplit.

    Parameters
    ----------
    model : estimator
        Unfitted sklearn-compatible model.
    param_distributions : dict
        Parameter search space (passed to RandomizedSearchCV).
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : array-like
        Target values.
    n_iter : int
        Number of random parameter combinations to try (default 50).
    n_splits : int
        Number of CV folds (default 5).
    n_jobs : int
        Number of parallel jobs for CV (-1 = all cores, 1 = sequential).
        Use 1 when the model itself uses GPU.

    Returns
    -------
    dict
        Keys: ``best_estimator``, ``best_params``, ``best_score`` (neg MAE),
        ``cv_results`` (DataFrame of all tried combinations).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X, y)

    results = {
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": -search.best_score_,  # convert neg MAE to positive
        "cv_results": pd.DataFrame(search.cv_results_),
    }

    print(f"  Best MAE: {results['best_score']:.3f}")
    print(f"  Best params: {results['best_params']}")
    return results


# ----------------------------------------------------------------------
# Feature ablation
# ----------------------------------------------------------------------

def feature_ablation(model, X, y, feature_groups=None, n_splits=5):
    """Drop-one-group ablation study using TimeSeriesSplit CV.

    Trains the model with all features, then retrains dropping each
    feature group to measure the performance impact.

    Parameters
    ----------
    model : estimator
        Unfitted sklearn-compatible model (will be cloned for each run).
    X : pd.DataFrame
        Full feature matrix (must have named columns).
    y : array-like
        Target values.
    feature_groups : dict or None
        Mapping of group name to list of column names. If None, uses
        the module-level ``FEATURE_GROUPS``.
    n_splits : int
        Number of CV folds (default 5).

    Returns
    -------
    pd.DataFrame
        One row per group + a row for the full model, with columns:
        ``group``, ``mae_mean``, ``mae_std``, ``delta_mae``.
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS

    print("Feature ablation: evaluating full model ...")
    full_cv = cross_validate_model(model, X, y, n_splits=n_splits)
    full_mae = full_cv["mae_mean"]

    rows = [{"group": "all_features", "mae_mean": full_mae,
             "mae_std": full_cv["mae_std"], "delta_mae": 0.0}]

    for group_name, group_cols in feature_groups.items():
        drop_cols = [c for c in group_cols if c in X.columns]
        if not drop_cols:
            continue
        X_ablated = X.drop(columns=drop_cols)
        print(f"  Dropping '{group_name}' ({len(drop_cols)} features) ...")
        ablated_cv = cross_validate_model(model, X_ablated, y, n_splits=n_splits)
        rows.append({
            "group": group_name,
            "mae_mean": ablated_cv["mae_mean"],
            "mae_std": ablated_cv["mae_std"],
            "delta_mae": ablated_cv["mae_mean"] - full_mae,
        })

    result_df = pd.DataFrame(rows)
    print(f"\nAblation results:\n{result_df.to_string(index=False)}")
    return result_df


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------

def save_model(model, filepath):
    """Persist a fitted model to disk using joblib.

    Parameters
    ----------
    model : estimator
        Fitted model to save.
    filepath : str
        Destination path (e.g. ``models/random_forest.joblib``).
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a previously saved model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model file.

    Returns
    -------
    estimator
        The loaded model.
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a small synthetic dataset for demonstration purposes.
    np.random.seed(42)
    n_samples = 500

    locations = ["College St", "Bloor St W", "Spadina Ave", "University Ave", "Dundas St W"]

    df = pd.DataFrame({
        "location_name": np.random.choice(locations, n_samples),
        "hour_of_day": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "month": np.random.randint(1, 13, n_samples),
        "is_weekend": np.random.randint(0, 2, n_samples),
        "is_rush_hour": np.random.randint(0, 2, n_samples),
        "temp": np.random.uniform(-10, 35, n_samples),
        "humidity": np.random.uniform(30, 100, n_samples),
        "visibility": np.random.uniform(1, 25, n_samples),
        "wind_speed": np.random.uniform(0, 50, n_samples),
        "is_raining": np.random.randint(0, 2, n_samples),
        "is_snowing": np.random.randint(0, 2, n_samples),
        "is_holiday": np.random.randint(0, 2, n_samples),
        "is_long_weekend": np.random.randint(0, 2, n_samples),
        "uoft_classes": np.random.randint(0, 2, n_samples),
        "uoft_exams": np.random.randint(0, 2, n_samples),
        "uoft_reading_week": np.random.randint(0, 2, n_samples),
        "is_covid_lockdown": np.random.randint(0, 2, n_samples),
        "is_covid_era": np.random.randint(0, 2, n_samples),
    })

    # Synthetic target: speed influenced by hour and rain.
    df["avg_speed"] = (
        45
        - 0.5 * np.abs(df["hour_of_day"] - 8)
        - 3.0 * df["is_raining"]
        + np.random.normal(0, 3, n_samples)
    )

    # Train all models.
    result = train_all_models(df)
    fitted_models = result["models"]

    # Save and reload one model as a demo.
    save_model(fitted_models["Random Forest"], "models/random_forest.joblib")
    loaded_rf = load_model("models/random_forest.joblib")

    # Quick sanity check.
    df_encoded, _ = encode_location(df, result["label_encoder"])
    preds = loaded_rf.predict(df_encoded[FEATURE_COLS].head(5))
    print(f"\nSample predictions from reloaded Random Forest:\n{preds}")
