"""
Test suite for CRC Engine
"""
import numpy as np
import pytest
from crc_engine import ConformalRiskControl


def test_crc_monotonic_loss_guarantee():
    # Simulate calibration losses for n=500 samples across 100 lambda thresholds
    # Losses monotonically decrease from 1.0 to 0.0 as lambda increases from 0 to 1
    np.random.seed(42)
    n_samples = 500
    n_lambdas = 101
    lambdas = np.linspace(0.0, 1.0, n_lambdas)

    # Random thresholds for each sample where it becomes covered (loss drops from 1 to 0)
    sample_cutoff = np.random.beta(a=2, b=5, size=(n_samples, 1))
    losses_grid = (lambdas < sample_cutoff).astype(float)

    crc = ConformalRiskControl(B=1.0)
    alpha_levels = [0.05, 0.10, 0.20]
    calibrated = crc.calibrate(losses_grid, alpha_levels, lambdas=lambdas)

    print("\nCalibrated thresholds:", calibrated)

    for alpha in alpha_levels:
        l_hat = calibrated[alpha]
        idx = np.argmin(np.abs(lambdas - l_hat))
        r_hat = np.mean(losses_grid[:, idx])
        r_bound = (n_samples / (n_samples + 1.0)) * r_hat + (1.0 / (n_samples + 1.0))
        assert r_bound <= alpha + 1e-4, f"Bound violated for alpha={alpha}: {r_bound} > {alpha}"
        print(f"Alpha {alpha:0.2f} -> lambda_hat={l_hat:0.3f}, empirical_calib_risk={r_hat:0.4f}, theoretical_bound={r_bound:0.4f}")


def test_compute_candidate_set():
    candidates = ["adenocarcinoma", "dysplasia", "benign tissue"]
    probabilities = [0.75, 0.20, 0.05]

    # lambda = 0.3 -> threshold = 1 - 0.3 = 0.7 -> only first candidate (0.75 >= 0.7)
    set_small = ConformalRiskControl.compute_candidate_set(candidates, probabilities, lambda_val=0.3)
    assert len(set_small) == 1
    assert set_small[0][0] == "adenocarcinoma"

    # lambda = 0.85 -> threshold = 0.15 -> first two candidates
    set_med = ConformalRiskControl.compute_candidate_set(candidates, probabilities, lambda_val=0.85)
    assert len(set_med) == 2

    # lambda = 0.98 -> threshold = 0.02 -> all candidates
    set_large = ConformalRiskControl.compute_candidate_set(candidates, probabilities, lambda_val=0.98)
    assert len(set_large) == 3


if __name__ == "__main__":
    test_crc_monotonic_loss_guarantee()
    test_compute_candidate_set()
    print("All local CRC tests passed!")
