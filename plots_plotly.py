import plotly.graph_objects as go
from typing import Iterable
import pandas as pd
import numpy as np


def add_line(
    fig: go.Figure,
    x_values: Iterable,
    y_values: Iterable,
    name: str,
    color: str,
) -> go.Figure:
    trace = go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        name=name,
        line={"color": color},
    )
    fig.add_trace(trace)
    return fig


def update_layout(fig: go.Figure):
    fig.update_layout(
        title="Time Series",
        xaxis_title="Time",
        yaxis_title="Value",
        height=600,
        hovermode="x unified",
        showlegend=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="day", step="day", stepmode="backward"),
                    dict(count=7, label="week", step="day", stepmode="backward"),
                    dict(count=1, label="month", step="month", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
            rangeslider={"visible": True},
            type="date",
        ),
        yaxis={"fixedrange": False},
    )
    return fig


def plot_time_series(time_series: pd.DataFrame, title: str = "Time Series"):
    """Create a basic time series plot supporting multidimensional data."""
    fig = go.Figure()
    
    # Find all value columns (value_0, value_1, value_2, etc.)
    value_cols = [col for col in time_series.columns if col.startswith('value_')]
    
    if not value_cols:
        raise ValueError("No value columns found. Expected columns starting with 'value_'")
    
    # Plot all value columns without legend labels
    for i, col in enumerate(value_cols):
        fig.add_trace(
            go.Scatter(
                x=time_series.index,
                y=time_series[col],
                mode="lines",
                showlegend=False,  # No legend labels as requested
                line=dict(width=1.5)
            )
        )
    
    update_layout(fig)
    return fig


def add_confidence_interval(fig: go.Figure, forecast: pd.DataFrame):
    ds = forecast.index
    fig.add_trace(
        go.Scatter(
            x=ds,
            y=forecast["expected"],
            mode="lines",
            name="Forecast",
            line=dict(color="rgba(31, 119, 180, 0.8)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ds,
            y=forecast["upper"],
            mode="lines",
            name="Upper Bound",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ds,
            y=forecast["lower"],
            mode="lines",
            name="Lower Bound",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(width=0),
            showlegend=False,
        )
    )
    return fig


def add_points(
    fig: go.Figure,
    x_values: Iterable,
    y_values: Iterable,
    name: str = "Anomalies",
    color: str = "red",
):
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            name=name,
            marker=dict(color=color, size=10),
        )
    )
    return fig


def add_anomalies(
    fig: go.Figure,
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
        for i, timestamp in enumerate(anomaly_timestamps):
            fig.add_vline(
                x=timestamp,
                line=dict(color="red", width=1.5, dash="dash"),
                opacity=0.7,
                name="Anomaly" if i == 0 else None,
                showlegend=(i == 0)  # Only show legend for first line
            )
    else:
        # Single-dimensional case: use points (backward compatibility)
        anomaly_points = time_series[anomaly_mask]
        if not anomaly_points.empty:
            fig = add_points(
                fig, anomaly_points.index, anomaly_points["value_0"], "Anomalies", "red"
            )
    
    # Add confidence interval if provided
    if expected_values is not None and expected_bounds is not None:
        time_series_copy = time_series.copy()
        time_series_copy["expected"] = expected_values
        time_series_copy["upper"] = expected_bounds[:, 0]
        time_series_copy["lower"] = expected_bounds[:, 1]
        fig = add_confidence_interval(fig, time_series_copy)

    return fig
