"""Unit tests for Temperature Scaling and Expected Calibration Error (ECE)."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from temperature_scaling import TemperatureScaler, compute_ece


def test_temperature_scaling_optimization():
    """Verify that temperature scaling fits T* and scales probabilities monotonically."""
    np.random.seed(42)
    n = 200

    # Simulate overconfident probabilities
    true_labels = np.random.binomial(1, 0.7, size=n)
    overconfident_probs = np.clip(true_labels * 0.4 + 0.55 + np.random.normal(0, 0.05, size=n), 0.01, 0.99)

    scaler = TemperatureScaler()
    t_opt = scaler.fit_from_confidences(overconfident_probs, true_labels)

    assert t_opt > 0.0
    assert scaler.temperature == t_opt

    scaled_probs = scaler.scale_confidences(overconfident_probs)
    assert len(scaled_probs) == n
    assert np.all(scaled_probs >= 0.0) and np.all(scaled_probs <= 1.0)


def test_ece_computation():
    """Verify that well-calibrated predictions yield near-zero ECE."""
    n = 500
    probs = np.array([0.9] * 450 + [0.1] * 50)
    labels = np.array([1] * 405 + [0] * 45 + [1] * 5 + [0] * 45)

    ece = compute_ece(probs, labels, n_bins=10)
    assert ece < 0.05
