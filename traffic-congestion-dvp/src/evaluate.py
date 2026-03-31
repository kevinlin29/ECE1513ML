"""
evaluate.py — Evaluation and visualization for traffic speed prediction models.

Provides functions to compute regression metrics, compare multiple models,
and generate publication-ready plots (scatter, bar chart, heatmap, etc.).

Scope: Predict avg_speed on streets near U of T campus using City of
Toronto midblock speed bin data (2020-2024).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Evaluate a fitted model on a test set.

    Parameters
    ----------
    model : estimator
        Fitted model with a ``predict`` method.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : array-like
        True target values.

    Returns
    -------
    dict
        Dictionary with keys ``MAE``, ``RMSE``, and ``R2``.
    """
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }


def evaluate_all_models(models_dict, X_test, y_test):
    """Evaluate every model and return a comparison DataFrame.

    Parameters
    ----------
    models_dict : dict[str, estimator]
        Mapping of model name to fitted model.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : array-like
        True target values.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by model name with columns MAE, RMSE, R2.
    """
    rows = {}
    for name, model in models_dict.items():
        rows[name] = evaluate_model(model, X_test, y_test)
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------

def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual"):
    """Scatter plot of predicted values against ground truth.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolors="none")

    # Perfect-prediction reference line.
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Ideal")

    ax.set_xlabel("Actual Speed (km/h)")
    ax.set_ylabel("Predicted Speed (km/h)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_model_comparison(results_df):
    """Grouped bar chart comparing models on MAE, RMSE, and R2.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`evaluate_all_models`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["MAE", "RMSE", "R2"]
    colors = sns.color_palette("muted", n_colors=len(results_df))

    for ax, metric in zip(axes, metrics):
        results_df[metric].plot.bar(ax=ax, color=colors)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Model Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names):
    """Horizontal bar chart of feature importances.

    Works for models that expose a ``feature_importances_`` attribute
    (e.g. Random Forest, XGBoost).

    Parameters
    ----------
    model : estimator
        Fitted tree-based model.
    feature_names : list[str]
        Names corresponding to each feature column.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    AttributeError
        If the model does not have ``feature_importances_``.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    ax.barh(
        np.array(feature_names)[indices],
        importances[indices],
        color=sns.color_palette("viridis", n_colors=len(feature_names)),
    )
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig


def plot_congestion_heatmap(df, speed_col="avg_speed"):
    """Heatmap of average speed by hour_of_day and day_of_week.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``hour_of_day``, ``day_of_week``, and the
        column specified by ``speed_col``.
    speed_col : str, default="avg_speed"
        Name of the column holding speed values.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pivot = df.pivot_table(
        values=speed_col,
        index="hour_of_day",
        columns="day_of_week",
        aggfunc="mean",
    )

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    # Rename columns only if they are integer-coded 0-6.
    if set(pivot.columns).issubset(set(range(7))):
        pivot = pivot.rename(columns=dict(enumerate(day_labels)))

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(pivot, cmap="RdYlGn", annot=True, fmt=".1f", ax=ax)
    ax.set_title(f"Average {speed_col.replace('_', ' ').title()} by Hour and Day")
    ax.set_ylabel("Hour of Day")
    ax.set_xlabel("Day of Week")
    fig.tight_layout()
    return fig


def plot_congestion_heatmap_by_location(df, speed_col="avg_speed"):
    """Heatmap of average speed by location_name and hour_of_day.

    Useful for comparing congestion patterns across different streets
    near U of T campus.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``location_name``, ``hour_of_day``, and the
        column specified by ``speed_col``.
    speed_col : str, default="avg_speed"
        Name of the column holding speed values.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pivot = df.pivot_table(
        values=speed_col,
        index="location_name",
        columns="hour_of_day",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6)))
    sns.heatmap(pivot, cmap="RdYlGn", annot=True, fmt=".1f", ax=ax)
    ax.set_title(f"Average {speed_col.replace('_', ' ').title()} by Location and Hour")
    ax.set_ylabel("Location")
    ax.set_xlabel("Hour of Day")
    fig.tight_layout()
    return fig


def plot_error_by_hour(y_true, y_pred, hours):
    """Bar chart of MAE broken down by hour of day.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    hours : array-like
        Hour label for each sample (0-23).

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    hours = np.asarray(hours)

    errors = np.abs(y_true - y_pred)
    df = pd.DataFrame({"hour_of_day": hours, "abs_error": errors})
    hourly_mae = df.groupby("hour_of_day")["abs_error"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hourly_mae.index, hourly_mae.values, color=sns.color_palette("coolwarm", n_colors=len(hourly_mae)))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Prediction Error by Hour")
    ax.set_xticks(range(0, 24))
    fig.tight_layout()
    return fig


def plot_error_by_location(y_true, y_pred, locations):
    """Horizontal bar chart of MAE broken down by location.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    locations : array-like
        Location name for each sample.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = np.abs(y_true - y_pred)
    df = pd.DataFrame({"location_name": locations, "abs_error": errors})
    loc_mae = df.groupby("location_name")["abs_error"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4, len(loc_mae) * 0.4)))
    ax.barh(loc_mae.index, loc_mae.values, color=sns.color_palette("muted", n_colors=len(loc_mae)))
    ax.set_xlabel("Mean Absolute Error")
    ax.set_ylabel("Location")
    ax.set_title("Prediction Error by Location")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Cross-validation & ablation plots
# ------------------------------------------------------------------

def plot_cv_results(cv_results_dict):
    """Grouped bar chart of CV results with error bars for multiple models.

    Parameters
    ----------
    cv_results_dict : dict[str, dict]
        Mapping of model name to dict with keys ``mae_mean``, ``mae_std``,
        ``rmse_mean``, ``rmse_std``, ``r2_mean``, ``r2_std`` (output of
        :func:`train.cross_validate_model`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(cv_results_dict.keys())
    metrics = ["mae", "rmse", "r2"]
    titles = ["MAE (km/h)", "RMSE (km/h)", "R²"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = sns.color_palette("muted", n_colors=len(names))

    for ax, metric, title in zip(axes, metrics, titles):
        means = [cv_results_dict[n][f"{metric}_mean"] for n in names]
        stds = [cv_results_dict[n][f"{metric}_std"] for n in names]
        bars = ax.bar(names, means, yerr=stds, color=colors, capsize=4)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Comparison (5-Fold TimeSeriesSplit CV)", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_ablation_results(ablation_df):
    """Horizontal bar chart of MAE increase when each feature group is dropped.

    Parameters
    ----------
    ablation_df : pd.DataFrame
        Output of :func:`train.feature_ablation` with columns ``group``,
        ``mae_mean``, ``delta_mae``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = ablation_df[ablation_df["group"] != "all_features"].copy()
    df = df.sort_values("delta_mae", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6)))
    colors = ["#d32f2f" if d > 0 else "#388e3c" for d in df["delta_mae"]]
    ax.barh(df["group"], df["delta_mae"], color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in MAE (km/h) when group is dropped")
    ax.set_title("Feature Ablation Study — Drop-One-Group")

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["delta_mae"], i,
                f" +{row['delta_mae']:.3f}" if row["delta_mae"] > 0 else f" {row['delta_mae']:.3f}",
                ha="left" if row["delta_mae"] >= 0 else "right",
                va="center", fontsize=9)

    fig.tight_layout()
    return fig


def plot_improvement_steps(steps_df):
    """Bar chart showing MAE improvement across incremental steps.

    Parameters
    ----------
    steps_df : pd.DataFrame
        Must have columns ``step``, ``description``, ``mae_mean``,
        ``mae_std``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("Blues_d", n_colors=len(steps_df))

    bars = ax.bar(steps_df["description"], steps_df["mae_mean"],
                  yerr=steps_df["mae_std"], color=colors, capsize=4)
    ax.set_ylabel("MAE (km/h)")
    ax.set_title("Incremental Model Improvement — CV MAE at Each Step")
    ax.tick_params(axis="x", rotation=25, labelsize=9)

    for bar, (_, row) in zip(bars, steps_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{row['mae_mean']:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# I/O helper
# ------------------------------------------------------------------

def save_figure(fig, filepath):
    """Save a matplotlib figure to disk.

    Creates parent directories if they do not exist.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filepath : str
        Destination path (e.g. ``figures/scatter.png``).
    """
    import os
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {filepath}")
