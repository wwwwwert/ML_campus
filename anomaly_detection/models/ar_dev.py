from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.tools import add_constant
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import timedelta
import scipy.stats as stats
from ..core import TimeSeriesWrapper
from ..core.utils import detect_seasonality_with_acf
from .base import BaseDetector, ModelResult


class ARDevDetector(BaseDetector):
    """
    AR (AutoRegressive) anomaly detection model. (development version)

    Implements autoregressive modeling for time series anomaly detection using
    residual analysis with z-score thresholding.
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "order": 30,
            "threshold": 3.0,
            "stable": True,
            "stable_sensitivity": 1.0,
            "order_decline_regime": True,
            "order_decline_step": 5,
            # Seasonal std estimation for bounds (like in STL)
            "seasonal_std": True,
            "seasonality": "day",  # 'auto', 'hour', 'day', 'week' or int
            "seasonal_std_clip": (0.5, 3.0),
        }

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params["order"] <= 0:
            raise ValueError("Autoregression order must be > 0")
        if params["threshold"] < 0:
            raise ValueError("threshold must be not less than 0")

    def _detect_period(self, time_series: TimeSeriesWrapper, series: np.ndarray) -> int:
        """Detect seasonal period based on params (like in STL)."""
        seasonality = self.params.get("seasonality", "auto")

        if seasonality == "auto":
            period = detect_seasonality_with_acf(series)
        elif isinstance(seasonality, int):
            period = seasonality
        elif seasonality in ("hour", "day", "week"):
            hours_map = {"hour": 1, "day": 24, "week": 7 * 24}
            time_delta = pd.to_timedelta(
                time_series.time_series_pd.index[1] - time_series.time_series_pd.index[0]
            ).total_seconds()
            period = int(timedelta(hours=hours_map[seasonality]).total_seconds() / time_delta)
        else:
            period = 1

        return max(1, period)

    def _calculate_mad(self, residual: np.ndarray) -> float:
        """Compute Median Absolute Deviation (always use median for seasonal std)."""
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        return mad / stats.norm.ppf(0.75) if mad > 0 else 1e-8

    def _check_insufficient_samples(self, time_series: TimeSeriesWrapper, order: int) -> bool:
        """
        Check if there are insufficient samples for AR model fitting.

        Args:
            time_series: Time series data
            order: AR model order

        Returns:
            True if samples are insufficient, False otherwise
        """
        n_samples = time_series.time_series_pd.shape[0]
        min_samples_required = order * 2 + 2
        return n_samples < min_samples_required

    def _get_fallback_result_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        """
        Return fallback result with zero scores for univariate time series with insufficient samples.

        Args:
            time_series: Time series data

        Returns:
            ModelResult with zero anomaly scores
        """
        n_samples = time_series.time_series_pd.shape[0]
        z_scores = np.zeros(n_samples)
        expected_value = np.array(time_series.time_series_pd["value_0"])
        expected_bounds = np.column_stack(
            (
                expected_value - self.params["threshold"],
                expected_value + self.params["threshold"],
            )
        )
        return ModelResult(
            anomaly_scores=z_scores,
            is_anomaly=(z_scores > self.params["threshold"]),
            expected_value=expected_value,
            expected_bounds=expected_bounds,
        )

    def _get_fallback_result_multivariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        """
        Return fallback result with zero scores for multivariate time series with insufficient samples.

        Args:
            time_series: Time series data

        Returns:
            ModelResult with zero anomaly scores
        """
        n_samples = time_series.time_series_pd.shape[0]
        z_scores = np.zeros(n_samples)
        expected_value = np.array(time_series.time_series_pd.values.T)
        return ModelResult(
            anomaly_scores=z_scores,
            is_anomaly=(z_scores > self.params["threshold"]),
            expected_value=expected_value,
            expected_bounds=None,
        )

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        order = self.params["order"]
        stable = self.params["stable"]

        if self._check_insufficient_samples(time_series, order):
            return self._get_fallback_result_univariate(time_series)

        current_order = order

        while True:
            model = AutoReg(time_series.time_series_pd["value_0"], lags=current_order)
            model_fit = model.fit()

            expected = np.concatenate((time_series.values[:current_order], model_fit.fittedvalues))
            residuals = time_series.values - expected
            residual_std = self.calculate_std(residuals) + 1e-8

            if stable:
                n = len(expected)

                data_mean = np.mean(time_series.values)
                data_std = np.std(time_series.values) + 1e-8

                normalized_expected = (np.array(time_series.values) - data_mean) / data_std

                stds = np.empty(n, dtype=np.float64)
                stds[:current_order] = residual_std / data_std

                ar_coeffs = model_fit.params.iloc[1 : current_order + 1].to_numpy()
                ar_const = model_fit.params.iloc[0]

                ar_coeffs_reversed = ar_coeffs[::-1]

                ar_const_norm = (ar_const - data_mean * (1 - np.sum(ar_coeffs))) / data_std

                current_std = residual_std / data_std
                base_std = current_std

                original_normalized = normalized_expected.copy()
                normalized_hidden = normalized_expected.copy()

                HALF_LOG_2PI = 0.9189385332046727

                for i in range(current_order, n):
                    prediction = ar_const_norm + normalized_hidden[i - current_order : i] @ ar_coeffs_reversed
                    val = original_normalized[i]

                    z = (val - prediction) / (current_std)
                    sigma_ratio = current_std / (base_std + 1e-12)
                    noise_adjusted_sensitivity = self.params["stable_sensitivity"] * (sigma_ratio)
                    anomaly_rate = (0.5 * z**2 + HALF_LOG_2PI) / noise_adjusted_sensitivity
                    inv_rate = 1 / (1 + anomaly_rate)

                    # update hidden and expected values
                    normalized_hidden[i] = prediction * (1 - inv_rate) + val * inv_rate
                    normalized_expected[i] = prediction

                    stds[i] = current_std

                    # for updating current_std assuming that we want values to stay inside 3-sigma bounds of new expected value in anomaly region
                    current_std += np.abs(normalized_expected[i] - normalized_expected[i - 1]) / 3.0 * (1 - inv_rate)
                    current_std = base_std * 0.1 + current_std * 0.9
                    # Protect against current_std becoming too small for constant series
                    current_std = max(current_std, 1e-12)

                expected = normalized_expected * data_std + data_mean
                residual_std = stds * data_std

            residuals = expected - time_series.values
            # Protect against division by zero for constant series
            if isinstance(residual_std, np.ndarray):
                residual_std = np.where(residual_std < 1e-12, 1e-12, residual_std)
            elif residual_std < 1e-12:
                residual_std = 1e-12

            # Apply seasonal std if enabled (uses residuals for bound estimation only)
            bounds_std = residual_std.copy() if isinstance(residual_std, np.ndarray) else residual_std
            if self.params.get("seasonal_std", False):
                period = self._detect_period(time_series, np.array(time_series.values))
                clip = self.params.get("seasonal_std_clip", (0.5, 3.0))

                n = len(residuals)
                if period * 7 < n:
                    # Calculate seasonal std and apply as modulation for bounds
                    seasonal_sigma = self.calculate_seasonal_std(residuals, period, clip)
                    seasonal_sigma = np.where(seasonal_sigma < 1e-12, 1e-12, seasonal_sigma)
                    bounds_std = seasonal_sigma

            # Ensure logical consistency: use same std for both z-scores and bounds
            # This ensures: is_anomaly ⟺ z_score > threshold ⟺ value out of bounds
            z_scores = np.abs(residuals / bounds_std)
            expected_bounds = np.column_stack(
                (
                    expected - bounds_std * self.params["threshold"],
                    expected + bounds_std * self.params["threshold"],
                )
            )

            if (
                np.mean(residuals**2) > np.mean((time_series.values - np.mean(time_series.values)) ** 2)
                and self.params["order_decline_regime"]
                and stable
            ):
                if current_order == 1:
                    current_order = order
                    stable = False
                current_order -= self.params["order_decline_step"]
                if current_order < 1:
                    current_order = 1
            else:
                break

        metadata = {}
        if self.params.get("seasonal_std", False):
            series = np.array(time_series.time_series_pd["value_0"])
            metadata["seasonal_period"] = self._detect_period(time_series, series)

        return ModelResult(
            anomaly_scores=z_scores,
            is_anomaly=(z_scores > self.params["threshold"]),
            expected_value=expected,
            expected_bounds=expected_bounds,
            metadata=metadata,
        )

    def _detect_multivariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        n_samples = time_series.time_series_pd.shape[0]
        n_series = time_series.time_series_pd.shape[1]
        order = self.params["order"]
        stable = self.params["stable"]

        if self._check_insufficient_samples(time_series, order):
            return self._get_fallback_result_multivariate(time_series)

        data = time_series.time_series_pd.values

        # Per-series normalization for numerical stability and scale invariance
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        stds = np.where(stds < 1e-10, 1.0, stds)
        data_norm = (data - means) / stds

        current_order = order

        while True:
            lagged_features = []

            for i in range(current_order, n_samples):
                lagged = data_norm[i - current_order : i].flatten()
                lagged_features.append(lagged)

            X = np.array(lagged_features)
            X = add_constant(X)

            predictions_all_norm = np.zeros_like(data_norm)
            predictions_all_norm[:current_order] = data_norm[:current_order]

            Y = data_norm[current_order:, :]
            X_pinv = np.linalg.pinv(X)
            B = X_pinv @ Y

            if stable:
                for i in range(current_order, n_samples):
                    cur_data_norm = np.concatenate(([1], predictions_all_norm[i - current_order : i, :].flatten()))
                    cur_prediction_norm = cur_data_norm @ B
                    expected_value_norm = data_norm[i, :].flatten()
                    residuals_norm = expected_value_norm - cur_prediction_norm
                    z_mag = np.linalg.norm(residuals_norm)
                    inv_rate = 1 / (1 + z_mag**2 / self.params["stable_sensitivity"])
                    predictions_all_norm[i, :] = cur_prediction_norm * (1 - inv_rate) + expected_value_norm * inv_rate
            else:
                preds_norm = X @ B
                predictions_all_norm[current_order:, :] = preds_norm

            # Denormalize predictions back to original scale
            predictions_all = predictions_all_norm * stds + means

            residuals = data[current_order:, :] - predictions_all[current_order:, :]

            resid_std = np.array([self.calculate_std(residuals[:, j]) for j in range(n_series)], dtype=float)
            resid_std = np.where(resid_std < 1e-12, 1e-12, resid_std)

            z_per_series = residuals / resid_std
            z_core = np.linalg.norm(z_per_series, axis=1)
            z_scores = np.concatenate([np.zeros(current_order, dtype=float), z_core])
            z_scores /= np.sqrt(n_series)  # normalize by the number of series

            expected_value = predictions_all.T

            z_scores = np.asarray(z_scores, dtype=float).reshape(-1)
            expected_value = np.asarray(expected_value, dtype=float)

            # Check if we need to reduce order
            data_mean_per_series = np.mean(data[current_order:, :], axis=0)
            data_variance = np.mean((data[current_order:, :] - data_mean_per_series) ** 2)
            residual_variance = np.mean(residuals**2)
            if residual_variance > data_variance and self.params["order_decline_regime"]:
                if current_order == 1:
                    current_order = order
                    stable = False
                current_order -= self.params["order_decline_step"]
                if current_order < 1:
                    current_order = 1
            else:
                break

        return ModelResult(
            anomaly_scores=z_scores,
            is_anomaly=(z_scores > self.params["threshold"]),
            expected_value=expected_value,
            expected_bounds=None,
        )