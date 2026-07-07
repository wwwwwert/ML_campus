from typing import Dict, Any

import pandas as pd
import numpy as np
from prophet import Prophet
from .base import BaseDetector, ModelResult
from ..core import TimeSeriesWrapper


class ProphetDetector(BaseDetector):
    """
    Facebook Prophet-based anomaly detection.

    This detector uses Prophet for forecasting and identifies anomalies
    as points that fall outside the prediction intervals.
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {"threshold": 3.0}

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params["threshold"] < 0:
            raise ValueError("threshold must be not less than 0")

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        model_params = {**self.params}
        del model_params["threshold"]
        del model_params["std_type"]

        df = pd.DataFrame({"ds": time_series.dates, "y": time_series.values})

        self.model = Prophet(**model_params)
        self.model.fit(df)

        forecast = self.model.predict(df)
        expected_value = forecast["yhat"].values

        residuals = time_series.values - expected_value
        residual_std = self.calculate_std(residuals)
        z_scores = np.abs(residuals / residual_std)

        expected_bounds = np.column_stack(
            (
                expected_value - residual_std * self.params["threshold"],
                expected_value + residual_std * self.params["threshold"],
            )
        )

        return ModelResult(
            anomaly_scores=z_scores,
            is_anomaly=(z_scores > self.params["threshold"]),
            expected_value=expected_value,
            expected_bounds=expected_bounds,
        )
