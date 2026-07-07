import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tsatools import detrend


def detect_seasonality_with_acf(ts, max_lag=None):
    """Определяет период сезонности используя автокорреляцию"""
    if max_lag is None:
        max_lag = min(len(ts) - 1, len(ts) // 2)  # 50% от длины ряда

    acf_values = acf(ts, nlags=max_lag, fft=True)

    start_lag = 10
    acf_values = acf_values[start_lag:]

    peaks = []
    for i in range(1, len(acf_values) - 1):
        if acf_values[i] > acf_values[i - 1] and acf_values[i] > acf_values[i + 1]:
            peaks.append((i + start_lag, acf_values[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)

    if not peaks:
        return 1

    return peaks[0][0]


def add_median_temporal_value(
    df: pd.DataFrame,
    time_col: str = "time",
    value_col: str = "value",
    freq: str = "daily",
) -> pd.Series:
    """
    Adds median value for each time period (by time of day or day of week + time) to each DataFrame element.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame with time series.
    time_col : str, optional
        Name of column with timestamps (default 'time').
    value_col : str, optional
        Name of column with values (default 'value').
    freq : str, optional
        Frequency for median calculation: 'daily' (time only) or 'weekly' (day of week + time) (default 'daily').

    Returns:
    -----------
    pd.Series
        Series with median values for each time period.
    """
    # Create DataFrame copy to avoid warnings
    df = df.copy()

    # Convert time column to datetime
    df["timestamp"] = pd.to_datetime(df[time_col])

    # Create grouping column depending on frequency
    if freq.lower() == "daily":
        df["time_group"] = df["timestamp"].dt.time
    elif freq.lower() == "weekly":
        # Combination of day of week and time, e.g.: (0, 12:30:00)
        df["time_group"] = list(zip(df["timestamp"].dt.dayofweek, df["timestamp"].dt.time))
    else:
        raise ValueError("Параметр freq должен быть 'daily' или 'weekly'")

    # Calculate median by groups
    median_by_group = df.groupby("time_group")[value_col].median().rename("median_value")

    # Join with original DataFrame
    result_series = df.join(median_by_group, on="time_group")["median_value"]

    return result_series


def add_std_temporal_value(
    df: pd.DataFrame,
    time_col: str = "time",
    value_col: str = "value",
    freq: str = "daily",
) -> pd.Series:
    """
    Adds standard deviation (std) for each time period
    (by time of day or day of week + time) to each DataFrame element.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame with time series.
    time_col : str, optional
        Name of column with timestamps (default 'time').
    value_col : str, optional
        Name of column with values (default 'value').
    freq : str, optional
        Frequency for calculation: 'daily' (time only) or 'weekly' (day of week + time) (default 'daily').

    Returns:
    -----------
    pd.Series
        Series with standard deviations for each time period.
    """
    # Create DataFrame copy to avoid warnings
    df = df.copy()

    # Convert time column to datetime
    df["timestamp"] = pd.to_datetime(df[time_col])

    # Create grouping column depending on frequency
    if freq.lower() == "daily":
        df["time_group"] = df["timestamp"].dt.time
    elif freq.lower() == "weekly":
        # Combination of day of week and time
        df["time_group"] = list(zip(df["timestamp"].dt.dayofweek, df["timestamp"].dt.time))
    else:
        raise ValueError("Параметр freq должен быть 'daily' или 'weekly'")

    # Calculate std by groups
    std_by_group = df.groupby("time_group")[value_col].std().rename("std_value")

    # Join with original DataFrame
    result_series = df.join(std_by_group, on="time_group")["std_value"]
    result_series = result_series.fillna(df["value"].std())

    return result_series


def get_observed_granularity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the observed time series granularity.

    Returns:
        Series with granularity statistics
    """
    value_counts = df.index.to_series().diff().value_counts()
    normalized_value_counts = df.index.to_series().diff().value_counts(normalize=True)
    result = pd.concat([value_counts, normalized_value_counts], axis=1)
    result.columns = ["count", "normalized count"]
    result.index = result.index.to_series().astype("str")
    return result


def median_smoothing(series: pd.Series, n_values: int) -> pd.Series:
    """Apply median smoothing with specified window size."""
    return series.rolling(window=n_values, center=True, min_periods=1).median()


def spectral_residual(values: pd.Series, window_size=3, padding_mode="autoreg", padding_size=10) -> pd.Series:
    """
    Apply spectral residual with specified window size.

    Args:
        values: Time series values
        window_size: Window size for spectral residual
    Returns:
        Spectral residual values

    About algorithm: https://opensource.salesforce.com/Merlion/v1.2.0/merlion.models.anomaly.html#module-merlion.models.anomaly.spectral_residual
    """
    if padding_mode != "autoreg":
        padded_values = np.pad(values, (padding_size, padding_size), mode=padding_mode)
    else:
        model = AutoReg(values, lags=10)
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=padding_size)

        padded_values = np.concatenate([values.values, forecast])

    transform = np.fft.fft(padded_values)
    log_amps = np.log(np.abs(transform))
    phases = np.angle(transform)

    avg_log_amps = pd.Series(log_amps).rolling(window_size, min_periods=1).mean().to_numpy()
    residuals = log_amps - avg_log_amps

    saliency_map = np.abs(np.fft.ifft(np.exp(residuals + 1j * phases)))

    if padding_size > 0:
        if padding_mode == "autoreg":
            saliency_map = saliency_map[:-padding_size]
        else:
            saliency_map = saliency_map[padding_size:-padding_size]
    return pd.Series(saliency_map, index=values.index)


def seasonal_component(series, period):
    series = np.asarray(series)
    n_periods = len(series) // period
    trimmed = series[: n_periods * period]
    reshaped = trimmed.reshape(-1, period)
    seasonality = np.median(reshaped, axis=0)
    repeated = np.tile(seasonality, n_periods + 1)[: len(series)]
    return repeated


def stl_decomposition(values: np.ndarray, n_steps: int = 5) -> pd.Series:
    """
    Apply STL decomposition with specified number of steps.

    Args:
        values: Time series values
        n_steps: Number of steps for STL decomposition
    Returns:
        STL decomposition values
    """
    values = detrend(values, order=1)
    for _ in range(n_steps):
        lag = detect_seasonality_with_acf(values)
        if lag == 1:
            break
        values = values - seasonal_component(values, lag)
        # values = STL(values, period=lag).fit().resid
    return values


def ewma(data, alpha):
    """
    Exponentially Weighted Moving Average without NaNs.
    Ignores NaNs in input; fills them forward.
    """
    ewma_arr = np.zeros_like(data)
    ewma_arr[0] = data[0]
    for t in range(1, len(data)):
        ewma_arr[t] = alpha * data[t] + (1 - alpha) * ewma_arr[t - 1]
    return ewma_arr


def adaptive_smoothing(
    values: np.ndarray,
    period: int,
    min_window: int = 1,
    max_window: int = 5,
    min_periods_required: int = 7,
) -> np.ndarray:
    """
    Apply phase-aware adaptive smoothing: heavier smoothing for unstable phases,
    minimal smoothing for stable phases.

    This helps reduce false positives in noisy/sparse periods (e.g., night hours
    for percentage metrics) while preserving resolution in stable periods.

    Args:
        values: Input time series values
        period: Seasonal period (e.g., 24 for hourly data with daily seasonality,
                1440 for minute data with daily seasonality)
        min_window: Minimum smoothing window for stable phases (default: 1 = no smoothing)
        max_window: Maximum smoothing window for unstable phases (default: 5)
        min_periods_required: Minimum number of complete periods required to estimate
                              phase confidence (default: 7)

    Returns:
        Smoothed values with phase-adaptive window sizes
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < period * min_periods_required:
        # Not enough data for adaptive smoothing, apply uniform light smoothing
        return pd.Series(values).rolling(window=min_window, min_periods=1, center=True).mean().values

    # Step 1: Estimate phase confidence based on variance stability across periods
    phase_confidence = np.ones(period)

    for phase in range(period):
        # Collect values at this phase across all periods
        phase_indices = np.arange(phase, n, period)
        phase_values = values[phase_indices]

        if len(phase_values) < 3:
            phase_confidence[phase] = 0.0
            continue

        # Compute per-period local variance (using rolling windows within each period)
        # For simplicity, compute variance of consecutive pairs of phase values
        diffs = np.diff(phase_values)
        if len(diffs) < 2:
            phase_confidence[phase] = 0.5
            continue

        # Measure instability: coefficient of variation of absolute differences
        mean_abs_diff = np.mean(np.abs(diffs))
        std_abs_diff = np.std(np.abs(diffs))

        if mean_abs_diff > 1e-10:
            instability = std_abs_diff / mean_abs_diff
        else:
            instability = 0.0

        # Convert instability to confidence (0 to 1)
        # High instability -> low confidence
        phase_confidence[phase] = 1.0 / (1.0 + instability)

    # Normalize confidence to [0, 1] range
    if phase_confidence.max() > phase_confidence.min():
        phase_confidence = (phase_confidence - phase_confidence.min()) / (
            phase_confidence.max() - phase_confidence.min()
        )
    else:
        phase_confidence[:] = 1.0

    # Step 2: Map confidence to smoothing window size
    # High confidence -> small window (min_window)
    # Low confidence -> large window (max_window)
    phase_windows = np.round(max_window - phase_confidence * (max_window - min_window)).astype(int)
    phase_windows = np.clip(phase_windows, min_window, max_window)

    # Step 3: Apply phase-dependent smoothing
    smoothed = values.copy()

    for phase in range(period):
        window = phase_windows[phase]
        if window <= 1:
            continue  # No smoothing needed

        # Get all indices for this phase
        phase_indices = np.arange(phase, n, period)

        # Apply rolling mean with the computed window size
        phase_values = values[phase_indices]
        smoothed_phase = pd.Series(phase_values).rolling(window=window, min_periods=1, center=True).mean().values

        smoothed[phase_indices] = smoothed_phase

    return smoothed


def compute_phase_confidence(
    values: np.ndarray,
    period: int,
    min_periods_required: int = 7,
) -> np.ndarray:
    """
    Compute confidence score for each phase in the seasonal period.

    Useful for diagnostics and understanding which phases are noisy.

    Args:
        values: Input time series values
        period: Seasonal period
        min_periods_required: Minimum periods to estimate confidence

    Returns:
        Array of confidence scores (0 to 1) for each phase
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < period * min_periods_required:
        return np.ones(period) * 0.5  # Unknown confidence

    phase_confidence = np.ones(period)

    for phase in range(period):
        phase_indices = np.arange(phase, n, period)
        phase_values = values[phase_indices]

        if len(phase_values) < 3:
            phase_confidence[phase] = 0.0
            continue

        diffs = np.diff(phase_values)
        if len(diffs) < 2:
            phase_confidence[phase] = 0.5
            continue

        mean_abs_diff = np.mean(np.abs(diffs))
        std_abs_diff = np.std(np.abs(diffs))

        if mean_abs_diff > 1e-10:
            instability = std_abs_diff / mean_abs_diff
        else:
            instability = 0.0

        phase_confidence[phase] = 1.0 / (1.0 + instability)

    # Normalize to [0, 1]
    if phase_confidence.max() > phase_confidence.min():
        phase_confidence = (phase_confidence - phase_confidence.min()) / (
            phase_confidence.max() - phase_confidence.min()
        )
    else:
        phase_confidence[:] = 1.0

    return phase_confidence
