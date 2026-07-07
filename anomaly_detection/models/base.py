from abc import ABC, abstractmethod
from pydantic import BaseModel, field_validator
from typing import Dict, Any, Optional

import numpy as np
from scipy import stats
from statsmodels.robust.scale import qn_scale
from ..core import TimeSeriesWrapper


class ModelResult(BaseModel):
    """
    Class for storing the result of an anomaly detection.

    This class is used to store the result of an anomaly detection,
    including the anomaly scores.
    """

    anomaly_scores: Any
    is_anomaly: Any
    expected_value: Any = None
    expected_bounds: Any = None
    metadata: Dict[str, Any] = {}

    @field_validator("anomaly_scores", "is_anomaly")
    @classmethod
    def check_anomaly_scores_numpy_array(cls, v, info):
        if not isinstance(v, np.ndarray):
            raise TypeError(f"{info.field_name} must be a numpy.ndarray")
        if v.ndim != 1:
            raise ValueError(f"{info.field_name} must be a 1D array, but got {v.ndim}D array with shape {v.shape}")
        return v

    @field_validator("expected_value", "expected_bounds")
    @classmethod
    def check_expected_value_numpy_array(cls, v, info):
        if not isinstance(v, np.ndarray) and v is not None:
            raise TypeError(f"{info.field_name} must be a numpy.ndarray or None")
        return v


class BaseDetector(ABC):
    """
    Base class for anomaly detection models.

    This abstract class defines the interface that all anomaly detection
    models must implement.
    """

    def __init__(self, **kwargs):
        """
        Initialize the detector with model-specific parameters.

        Args:
            **kwargs: Model-specific parameters
        """
        self.params = {**self.get_default_params(), **kwargs}
        if "std_type" not in self.params:
            self.params["std_type"] = "default"
        self.validate_params(self.params)

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.
        Returns:
            Dictionary of default parameter values
        """
        pass

    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate the provided parameters.

        Args:
            params: Dictionary of parameters to validate

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    def _detect_multivariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        """
        Detect anomalies in multivariate time series.

        Args:
            time_series: Multivariate time series data

        Returns:
            ModelResult object containing detected anomalies and anomaly scores

        Raises:
            NotImplementedError: If the detector does not support multivariate time series
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support multivariate time series. "
            f"Received {time_series.n_series} series."
        )

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        """
        Detect anomalies in univariate time series.

        Args:
            time_series: Univariate time series data

        Returns:
            ModelResult object containing detected anomalies and anomaly scores

        Raises:
            NotImplementedError: If the detector does not implement univariate detection
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement univariate detection.")

    def __call__(self, time_series: TimeSeriesWrapper) -> ModelResult:
        if time_series.is_multivariate:
            return self._detect_multivariate(time_series)
        else:
            return self._detect_univariate(time_series)

    def calculate_std(self, residual: np.array) -> float:
        """
        Calculate the standard deviation of the residuals.

        Args:
            residual: Array of residuals

        Returns:
            Standard deviation of residuals
        """
        if self.params["std_type"] == "default":
            return np.sqrt(np.mean(residual**2))
        elif self.params["std_type"] == "default_robust":
            # Robust RMS using trimmed sample
            r = np.asarray(residual)
            abs_r = np.abs(r)

            if abs_r.size < 20:
                return np.std(r)

            q_lo, q_hi = 0.75, 0.98
            a_emp, b_emp = np.quantile(abs_r, [q_lo, q_hi])

            mask = (abs_r >= a_emp) & (abs_r <= b_emp)
            trimmed = r[mask]

            if trimmed.size < 10:
                return np.std(r)

            # raw (biased) estimate on truncated sample
            sigma_raw2 = np.mean(trimmed**2)

            # theoretical correction factor
            a = stats.norm.ppf(q_lo)
            b = stats.norm.ppf(q_hi)

            phi = stats.norm.pdf
            Phi = stats.norm.cdf

            C = ((-b * phi(b) + Phi(b)) - (-a * phi(a) + Phi(a))) / (Phi(b) - Phi(a))

            sigma = np.sqrt(sigma_raw2 / C)
            return sigma
        elif self.params["std_type"] == "mad":
            return np.median(np.abs(residual)) / stats.norm.ppf(0.75)
        elif self.params["std_type"] == "iqr":
            return np.subtract(*np.percentile(residual, [75, 25])) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
        elif self.params["std_type"] == "qn_scale":
            return qn_scale(residual)
        else:
            raise ValueError(f"Unknown std_type: {self.params['std_type']}")

    def calculate_seasonal_std(
        self, residual: np.array, period: int, clip: Optional[tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Calculate robust seasonal standard deviation resistant to outliers.

        For each phase in the seasonal cycle, estimates std using neighboring phases
        within a temporal window. Uses aggressive outlier filtering at two levels:
        1. Per-period filtering: exclude periods with anomalously high std
        2. Per-value filtering: trim extreme values using percentile-based approach

        This makes the estimate robust to:
        - Broken periods (all or most values anomalous)
        - Point outliers at the same phase across different periods

        Args:
            residual: Array of residuals
            period: Period of the seasonal component
            clip: (min, max) multipliers relative to overall std

        Returns:
            Array of seasonal std values (one per position), shape (len(residual),)
        """
        if clip is None:
            clip = (0.5, 3.0)

        statistical_evidence_bound = 500
        num_of_periods = len(residual) // period + 1
        window_size = max((statistical_evidence_bound // num_of_periods) // 2, 1) + 1
        values = []
        overall_std = self.calculate_std(residual)

        for i in range(period):
            # Collect residuals at phase i across all periods and phase neighbors
            phase_values_by_period = []
            for period_idx in range(num_of_periods):
                period_phase_values = []
                for j in range(-window_size, window_size + 1):
                    idx = period_idx * period + (i + j) % period
                    if 0 <= idx < len(residual):
                        period_phase_values.append(residual[idx])
                phase_values_by_period.append(period_phase_values)

            # Step 1: Calculate std for each period independently
            # This approach uses per-period medians, making it robust to broken periods
            period_stds = []
            for period_phase_values in phase_values_by_period:
                if len(period_phase_values) > 0:
                    # Calculate std for this period's phase values
                    period_stds.append(self.calculate_std(np.array(period_phase_values)))
                else:
                    period_stds.append(0)

            # Step 2: Use median std across periods (robust to broken periods)
            # The median is immune to entire periods being anomalous
            if len(period_stds) > 0:
                period_stds_arr = np.array(period_stds)
                # Filter out clearly broken periods (std > 80th percentile)
                # while keeping most normal periods
                percentile_80 = (
                    np.percentile(period_stds_arr[period_stds_arr > 0], 80)
                    if np.sum(period_stds_arr > 0) > 2
                    else np.max(period_stds_arr)
                )
                good_stds = period_stds_arr[period_stds_arr <= percentile_80]

                if len(good_stds) > 0:
                    # Use median of good periods' stds (very robust)
                    phase_std = np.median(good_stds)
                else:
                    phase_std = overall_std
            else:
                phase_std = overall_std

            values.append(max(phase_std, 1e-12))

        # Replicate seasonal std values to match time series length
        values_arr = np.array(values)
        values_tiled = np.tile(values_arr, num_of_periods)[: len(residual)]
        # Clip to reasonable bounds relative to overall std
        return np.clip(values_tiled, overall_std * clip[0], overall_std * clip[1])
