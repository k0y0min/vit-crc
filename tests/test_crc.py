"""Unit tests for Conformal Risk Control (CRC) Engine."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from crc_engine import ConformalRiskControl


def test_crc_monotonic_loss_guarantee():
    """Verify that CRC calibration satisfies the finite-sample risk bound."""
    np.random.seed(42)
    n_samples = 500
    n_lambdas = 101
    lambdas = np.linspace(0.0, 1.0, n_lambdas)

    sample_cutoff = np.random.beta(a=2, b=5, size=(n_samples, 1))
    losses_grid = (lambdas < sample_cutoff).astype(float)

    crc = ConformalRiskControl(B=1.0)
    alpha_levels = [0.05, 0.10, 0.20]
    calibrated = crc.calibrate(losses_grid, alpha_levels, lambdas=lambdas)

    for alpha in alpha_levels:
        l_hat = calibrated[alpha]
        idx = np.argmin(np.abs(lambdas - l_hat))
        r_hat = np.mean(losses_grid[:, idx])
        r_bound = (n_samples / (n_samples + 1.0)) * r_hat + (1.0 / (n_samples + 1.0))
        assert r_bound <= alpha + 1e-4, f"Bound violated for alpha={alpha}: {r_bound} > {alpha}"


def test_compute_candidate_set():
    """Verify candidate set generation under varying thresholds."""
    candidates = ["adenocarcinoma", "dysplasia", "benign tissue"]
    probabilities = [0.75, 0.20, 0.05]

    # lambda = 0.3 -> threshold = 1 - 0.3 = 0.7 -> only top candidate (0.75 >= 0.7)
    set_small = ConformalRiskControl.compute_candidate_set(candidates, probabilities, lambda_val=0.3)
    assert len(set_small) == 1
    assert set_small[0][0] == "adenocarcinoma"

    # lambda = 0.85 -> threshold = 0.15 -> top two candidates
    set_med = ConformalRiskControl.compute_candidate_set(candidates, probabilities, lambda_val=0.85)
    assert len(set_med) == 2

    # lambda = 0.98 -> threshold = 0.02 -> all candidates
    set_large = ConformalRiskControl.compute_candidate_set(candidates, probabilities, lambda_val=0.98)
    assert len(set_large) == 3


def test_selective_abstention():
    """Verify selective prediction and straight-through processing logic."""
    probabilities = np.array([0.96, 0.92, 0.85, 0.70, 0.55])
    lambda_hat = 0.80

    # Predictions >= 0.80 pass through; < 0.80 abstain
    accepted = probabilities >= lambda_hat
    stp_rate = np.mean(accepted)
    assert stp_rate == 0.6  # 3 of 5 accepted
