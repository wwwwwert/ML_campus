import matplotlib.pyplot as plt
import seaborn as sns
from typing import Iterable
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def add_line(
    ax: Axes,
    x_values: Iterable,
    y_values: Iterable,
    name: str,
    color: str,
) -> Axes:
    """Add a line plot to the axes."""
    ax.plot(x_values, y_values, color=color, label=name, linewidth=1.5)
    return ax


def update_layout(ax: Axes, fig: Figure):
    """Update the layout with time series specific settings."""
    ax.set_title("Time Series", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)

    # Set figure size (equivalent to height=600 in plotly)
    fig.set_size_inches(20, 8)

    # Enable grid for better readability
    ax.grid(True, alpha=0.3)

    # Format x-axis for dates if the index contains datetime
    if hasattr(ax.get_xaxis(), "set_major_formatter"):
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Tight layout to prevent label cutoff
    fig.tight_layout()

    return ax


def plot_time_series(time_series: pd.DataFrame, title: str = "Time Series"):
    """Create a basic time series plot supporting multidimensional data."""
    fig, ax = plt.subplots(figsize=(20, 8))

    # Find all value columns (value_0, value_1, value_2, etc.)
    value_cols = [col for col in time_series.columns if col.startswith('value_')]
    
    if not value_cols:
        raise ValueError("No value columns found. Expected columns starting with 'value_'")
    
    # Plot all value columns without legend labels
    for i, col in enumerate(value_cols):
        # Use matplotlib's default color cycle, don't add to legend
        ax.plot(time_series.index, time_series[col], linewidth=1.5)

    # Update layout
    update_layout(ax, fig)

    return ax


def add_confidence_interval(ax: Axes, forecast: pd.DataFrame):
    """Add forecast line with confidence interval."""
    ds = forecast.index

    # Add forecast line
    ax.plot(
        ds,
        forecast["expected"],
        color=(31 / 255, 119 / 255, 180 / 255, 0.8),
        label="Forecast",
        linewidth=2,
    )

    # Add confidence interval as filled area
    ax.fill_between(
        ds,
        forecast["lower"],
        forecast["upper"],
        color=(31 / 255, 119 / 255, 180 / 255),
        alpha=0.2,
        label="Confidence Interval",
    )

    return ax


def add_points(
    ax: Axes,
    x_values: Iterable,
    y_values: Iterable,
    name: str = "Anomalies",
    color: str = "red",
):
    """Add scatter points to the plot."""
    ax.scatter(
        x_values,
        y_values,
        color=color,
        label=name,
        s=50,  # equivalent to size=10 in plotly
        zorder=5,  # ensure points are on top
    )
    return ax


def add_anomalies(
    ax: Axes,
    time_series: pd.DataFrame,
    is_anomaly: np.ndarray,
    expected_values: np.array = None,
    expected_bounds: np.array = None,
):
    """Add anomaly visualization. For multidimensional data, draws red vertical lines at anomaly timestamps."""
    # Find all value columns to determine if this is multidimensional data
    value_cols = [col for col in time_series.columns if col.startswith('value_')]
    
    # Filter anomaly timestamps
    anomaly_mask = is_anomaly == 1
    anomaly_timestamps = time_series.index[anomaly_mask]
    
    if len(value_cols) > 1:
        # Multidimensional case: draw red vertical lines
        for timestamp in anomaly_timestamps:
            ax.axvline(x=timestamp, color='red', alpha=0.7, linewidth=1.5, linestyle='--', label='Anomaly' if timestamp == anomaly_timestamps[0] else "")
    else:
        # Single-dimensional case: use points (backward compatibility)
        anomaly_points = time_series[anomaly_mask]
        if not anomaly_points.empty:
            ax = add_points(
                ax, anomaly_points.index, anomaly_points["value_0"], "Anomalies", "red"
            )
    
    # Add confidence interval if provided
    if expected_values is not None and expected_bounds is not None:
        time_series_copy = time_series.copy()
        time_series_copy["expected"] = expected_values
        time_series_copy["upper"] = expected_bounds[:, 0]
        time_series_copy["lower"] = expected_bounds[:, 1]
        ax = add_confidence_interval(ax, time_series_copy)

    return ax


def create_seaborn_time_series(time_series: pd.DataFrame, title: str = "Time Series"):
    """Create a time series plot using seaborn style supporting multidimensional data."""
    # Set seaborn style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Find all value columns (value_0, value_1, value_2, etc.)
    value_cols = [col for col in time_series.columns if col.startswith('value_')]
    
    if not value_cols:
        raise ValueError("No value columns found. Expected columns starting with 'value_'")

    # Plot all value columns using seaborn
    time_series_reset = time_series.reset_index()
    index_name = time_series.index.name or "index"
    
    for col in value_cols:
        sns.lineplot(
            data=time_series_reset,
            x=index_name,
            y=col,
            ax=ax,
            linewidth=2,
            legend=False  # No legend labels as requested
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    fig.tight_layout()

    return fig, ax


def add_seaborn_confidence_interval(ax: Axes, forecast: pd.DataFrame):
    """Add confidence interval using seaborn style."""
    ds = forecast.index

    # Main forecast line
    sns.lineplot(
        x=ds,
        y=forecast["expected"],
        ax=ax,
        color="steelblue",
        linewidth=2,
        label="Forecast",
    )

    # Confidence interval
    ax.fill_between(
        ds,
        forecast["lower"],
        forecast["upper"],
        alpha=0.3,
        color="steelblue",
        label="Confidence Interval",
    )

    return ax


def add_seaborn_anomalies(
    ax: Axes,
    time_series: pd.DataFrame,
    is_anomaly: np.ndarray,
    expected_values: np.array = None,
    expected_bounds: np.array = None,
):
    """Add anomalies using seaborn style. For multidimensional data, draws red vertical lines at anomaly timestamps."""
    # Find all value columns to determine if this is multidimensional data
    value_cols = [col for col in time_series.columns if col.startswith('value_')]
    
    # Filter anomaly timestamps
    anomaly_mask = is_anomaly == 1
    anomaly_timestamps = time_series.index[anomaly_mask]
    
    if len(value_cols) > 1:
        # Multidimensional case: draw red vertical lines
        for timestamp in anomaly_timestamps:
            ax.axvline(x=timestamp, color='red', alpha=0.7, linewidth=1.5, label='Anomaly' if timestamp == anomaly_timestamps[0] else "")
    else:
        # Single-dimensional case: use scatter points (backward compatibility)
        anomaly_points = time_series[anomaly_mask]
        if not anomaly_points.empty:
            sns.scatterplot(
                x=anomaly_points.index,
                y=anomaly_points["value_0"],
                ax=ax,
                color="red",
                s=100,
                label="Anomalies",
                zorder=5,
            )

    # Add confidence interval if provided
    if expected_values is not None and expected_bounds is not None:
        time_series_copy = time_series.copy()
        time_series_copy["expected"] = expected_values
        time_series_copy["upper"] = expected_bounds[:, 0]
        time_series_copy["lower"] = expected_bounds[:, 1]
        ax = add_seaborn_confidence_interval(ax, time_series_copy)

    return ax
