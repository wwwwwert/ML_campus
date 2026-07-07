from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from typing import Dict, Any
from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult


class ARDetector(BaseDetector):
    """
    AR (AutoRegressive) anomaly detection model.

    Implements autoregressive modeling for time series anomaly detection using
    residual analysis with z-score thresholding.
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "order": 20,
            "threshold": 3.0,
            "stable": True,
            "stable_sensitivity": 1.0,
            "order_decline_regime": True,
            "order_decline_step": 5,
        }

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params["order"] <= 0:
            raise ValueError("Autoregression order must be > 0")
        if params["threshold"] < 0:
            raise ValueError("threshold must be not less than 0")

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
                    # using hidden value as a prediction for the next value
                    # and expected value as a expected value for the next
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
            z_scores = np.abs(residuals / residual_std)
            expected_bounds = np.column_stack(
                (
                    expected - residual_std * self.params["threshold"],
                    expected + residual_std * self.params["threshold"],
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

        return ModelResult(
            anomaly_scores=z_scores,
            is_anomaly=(z_scores > self.params["threshold"]),
            expected_value=expected,
            expected_bounds=expected_bounds,
        )
