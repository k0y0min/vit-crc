"""
Temperature Scaling & Calibration Metrics Module
Implements:
1. Optimal Temperature Fitting T* via NLL minimization (Guo et al., ICML 2017)
2. Expected Calibration Error (ECE) computation across probability bins
3. Universal Logit/Log-Odds Temperature Scaling for VLM Confidence
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit as scipy_logit
from typing import Dict, Any, Tuple


class TemperatureScaler:
    """
    Post-hoc confidence temperature scaling:
        z = log(p / (1 - p))
        p_calibrated = sigmoid(z / T)
    Optimizes T* on calibration data to minimize Negative Log-Likelihood (NLL).
    """

    def __init__(self):
        self.temperature: float = 1.0

    def fit_from_confidences(
        self,
        confidences: np.ndarray,
        is_correct: np.ndarray,
        eps: float = 1e-5,
    ) -> float:
        """
        Fits optimal temperature T* on calibration confidences and binary correctness.
        Args:
            confidences: Shape (N,) predicted confidence scores in (0, 1).
            is_correct: Shape (N,) boolean array indicating correctness.
        """
        # Clip to prevent logit explosion
        p = np.clip(confidences, eps, 1.0 - eps)
        z = np.log(p / (1.0 - p)) # Log-odds / logits
        y = is_correct.astype(np.float64)

        def nll_objective(t: float) -> float:
            scaled_z = z / t
            # Stable log-sigmoid computation
            # loss = -(y * log_sigmoid(scaled_z) + (1 - y) * log_sigmoid(-scaled_z))
            p_scaled = expit(scaled_z)
            p_scaled = np.clip(p_scaled, 1e-12, 1.0 - 1e-12)
            nll = -np.mean(y * np.log(p_scaled) + (1.0 - y) * np.log(1.0 - p_scaled))
            return float(nll)

        res = minimize_scalar(nll_objective, bounds=(0.1, 5.0), method="bounded")
        self.temperature = float(res.x)
        print(f"[TemperatureScaler] Optimal Temperature fitted on calibration data: T* = {self.temperature:.4f}", flush=True)
        return self.temperature

    def scale_confidences(self, confidences: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Transforms raw confidence scores using fitted temperature T*.
        """
        p = np.clip(confidences, eps, 1.0 - eps)
        z = np.log(p / (1.0 - p))
        return expit(z / self.temperature)


def compute_ece(confidences: np.ndarray, is_correct: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for confidence predictions.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(is_correct[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return float(ece)
