import numpy as np
import pandas as pd
from typing import Dict, Any
from ..core import TimeSeriesWrapper
from ..core.utils import detect_seasonality_with_acf, detrend
from .base import BaseDetector, ModelResult
import scipy.stats as stats
from datetime import timedelta


class STLDetector(BaseDetector):
    """
    MEDIFF-inspired STL (Seasonal Trend decomposition) anomaly detection model.
    Based on robust median-based decomposition from the MEDIFF paper.
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "threshold": 3.0,
            "smoothing": 1,
            "seasonality": "auto",
            "seasonal_window": 3,  # hardcoded
            "dst_weight": 1.0,  # beta parameter for DST compensation
            "std_type": "mad",
        }

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params["threshold"] < 0:
            raise ValueError("threshold must be not less than 0")
        if params["seasonality"] not in ("auto", "hour", "day", "week"):
            raise ValueError("seasonality should be one of 'auto', 'hour', 'day', 'week'")
        if params["dst_weight"] < 0 or params["dst_weight"] > 1:
            raise ValueError("dst_weight should be between 0 and 1")
        if params["seasonal_window"] < 0:
            raise ValueError("seasonal_window should be non-negative")

    def _extract_seasonal_component(self, series: np.ndarray, period: int, window: int = 3) -> np.ndarray:
        """
        Extract seasonal component using median (Equation 4 in paper)
        """
        if period <= 1 or len(series) < period * 2:
            return np.zeros_like(series)

        n_periods = len(series) // period
        seasonal = np.zeros(period)

        # not vectorized version, todo fix
        for i in range(period):
            values = []
            for p in range(n_periods):
                idx = p * period + i
                if idx < len(series):
                    for w in range(-window, window + 1):
                        window_idx = p * period + ((i + w) % period)
                        if 0 <= window_idx < len(series):
                            values.append(series[window_idx])

            if values:
                seasonal[i] = np.median(values)

        result = np.tile(seasonal, (len(series) // period) + 1)[: len(series)]
        return result

    def _compute_mad(self, residual: np.ndarray) -> float:
        """Compute Median Absolute Deviation"""
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        return mad / stats.norm.ppf(0.75) if mad > 0 else 1e-8

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        values = time_series.time_series_pd.rolling(window=self.params["smoothing"], min_periods=1).mean()
        series = np.asarray(values)[:, 0]

        # Handle empty or too short series
        if len(series) < 3:
            return ModelResult(
                anomaly_scores=np.zeros(len(series)),
                is_anomaly=np.zeros(len(series), dtype=bool),
                expected_value=series.copy(),
                expected_bounds=np.column_stack((series - 1, series + 1)),
            )

        if self.params["seasonality"] == "auto":
            period = detect_seasonality_with_acf(series)
        elif self.params["seasonality"] in ("hour", "day", "week"):
            hours_map = {"hour": 1, "day": 24, "week": 7 * 24}
            time_delta = pd.to_timedelta(
                time_series.time_series_pd.index[1] - time_series.time_series_pd.index[0]
            ).total_seconds()
            period = int(timedelta(hours=hours_map[self.params["seasonality"]]).total_seconds() / time_delta)

        period = max(2, period)

        trend = series - detrend(series, 1)

        detrended = series - trend

        seasonal = self._extract_seasonal_component(detrended, period, self.params["seasonal_window"])

        # unused for a while, todo fix
        # if self.params["dst_weight"] < 1.0:
        #     seasonal_trend_window = min(30, len(detrended) // 4)
        #     seasonal_trend = self._moving_median(detrended, seasonal_trend_window)
        #     seasonal = self.params["dst_weight"] * seasonal + (1 - self.params["dst_weight"]) * seasonal_trend

        residual = series - trend - seasonal

        time_delta = pd.to_timedelta(
            time_series.time_series_pd.index[1] - time_series.time_series_pd.index[0]
        ).total_seconds()
        day_period = int(timedelta(hours=24).total_seconds() / time_delta)

        if day_period * 7 <= len(residual):
            std_dev = self.calculate_seasonal_std(residual, day_period)
        else:
            std_dev = self.calculate_std(residual)
        # std_dev = self.update_std_with_holidays()
        anomaly_scores = np.abs(residual) / std_dev

        expected = trend + seasonal
        expected_bounds = np.column_stack(
            (
                expected - self.params["threshold"] * std_dev,
                expected + self.params["threshold"] * std_dev,
            )
        )

        return ModelResult(
            anomaly_scores=anomaly_scores,
            is_anomaly=(anomaly_scores > self.params["threshold"]),
            expected_value=expected,
            expected_bounds=expected_bounds,
        )
