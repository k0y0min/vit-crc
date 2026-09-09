"""
Conformal Risk Control (CRC) Engine
Implements distribution-free finite-sample risk control (Angelopoulos et al., 2022/2024).

Provides:
1. Conformal Selective Abstention (Fail-Safe / Straight-Through Processing):
   - Mathematically bounds the overall error risk: E[L(lambda)] <= alpha
   - Measures Enterprise Straight-Through Processing (STP) rate / coverage
2. Conformal Prediction Sets C_lambda(X):
   - Mathematically bounds miscoverage risk: E[1 - I(Y in C_lambda)] <= alpha
   - Measures efficiency via average candidate set size |C_lambda|
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class ConformalRiskControl:
    """
    Finite-sample Conformal Risk Control for monotonic non-increasing loss functions.
    Theorem (Angelopoulos et al.):
        Let L_i(lambda) in [0, B] be non-increasing in lambda.
        Empirical mean risk: R_hat(lambda) = (1/n) * sum_{i=1}^n L_i(lambda)
        Conformal threshold:
            lambda_hat = inf { lambda in Lambda : (n / (n + 1)) * R_hat(lambda) + B / (n + 1) <= alpha }
        Guarantees on unseen exchangeable test data:
            E[L(lambda_hat)] <= alpha
    """

    def __init__(self, B: float = 1.0):
        self.B = B
        self.calibrated_thresholds: Dict[float, float] = {}

    def calibrate(
        self,
        losses_grid: np.ndarray,
        alpha_levels: List[float],
        lambdas: np.ndarray,
    ) -> Dict[float, float]:
        """
        Calibrate lambda thresholds for a list of target risk tolerances alpha.

        Args:
            losses_grid: np.ndarray of shape (n_calib, n_lambdas) containing
                         non-increasing loss for each sample across the lambda grid.
            alpha_levels: Target risk levels alpha (e.g. [0.01, 0.05, 0.10, 0.20]).
            lambdas: Grid of lambda values (shape: n_lambdas,).

        Returns:
            Dict mapping alpha -> calibrated lambda_hat.
        """
        n = losses_grid.shape[0]
        # Empirical mean risk across calibration samples
        r_hat = np.mean(losses_grid, axis=0)

        # Theoretical upper bound from CRC theorem:
        r_bound = (n / (n + 1.0)) * r_hat + (self.B / (n + 1.0))

        calibrated = {}
        for alpha in alpha_levels:
            valid_indices = np.where(r_bound <= alpha)[0]
            if len(valid_indices) > 0:
                # Infimum lambda that satisfies the bound
                idx = valid_indices[0]
                calibrated[float(alpha)] = float(lambdas[idx])
            else:
                # Conservative fallback to max lambda
                calibrated[float(alpha)] = float(lambdas[-1])

        self.calibrated_thresholds = calibrated
        return calibrated

    # -------------------------------------------------------------
    # Mode 1: Selective Abstention / Straight-Through Processing
    # -------------------------------------------------------------
    @staticmethod
    def build_abstention_loss_grid(
        confidences: np.ndarray,
        is_correct: np.ndarray,
        lambdas: np.ndarray,
    ) -> np.ndarray:
        """
        Loss for selective abstention:
        If confidence >= lambda: model accepts and predicts. Loss = 1 if incorrect, 0 if correct.
        If confidence < lambda: model abstains ([ABSTAIN]). Loss = 0 (error prevented).
        As lambda increases from 0 to 1, threshold tightens, more queries abstain,
        and unhandled errors monotonically decrease.
        Shape: (n_samples, len(lambdas))
        """
        n = len(confidences)
        n_l = len(lambdas)
        grid = np.zeros((n, n_l), dtype=np.float32)

        for j, lam in enumerate(lambdas):
            accepted = confidences >= lam
            # Error occurs only if accepted AND prediction was wrong
            grid[:, j] = (accepted & (~is_correct)).astype(np.float32)

        return grid

    @staticmethod
    def evaluate_abstention(
        confidences: np.ndarray,
        is_correct: np.ndarray,
        lambda_val: float,
    ) -> Dict[str, float]:
        """
        Evaluates selective abstention at a specific threshold lambda:
        - empirical_risk: errors / total_samples (overall bounded error risk)
        - conditional_error_rate: errors / accepted_samples
        - straight_through_rate: accepted_samples / total_samples (coverage / STP rate)
        """
        n = len(confidences)
        accepted = confidences >= lambda_val
        n_accepted = np.sum(accepted)
        n_errors = np.sum(accepted & (~is_correct))

        empirical_risk = float(n_errors / n) if n > 0 else 0.0
        cond_error_rate = float(n_errors / n_accepted) if n_accepted > 0 else 0.0
        stp_rate = float(n_accepted / n) if n > 0 else 0.0

        return {
            "empirical_risk": empirical_risk,
            "conditional_error_rate": cond_error_rate,
            "straight_through_rate": stp_rate,
            "n_accepted": int(n_accepted),
            "n_abstained": int(n - n_accepted),
            "n_errors": int(n_errors),
        }

    # -------------------------------------------------------------
    # Mode 2: Conformal Candidate Prediction Sets
    # -------------------------------------------------------------
    @staticmethod
    def compute_candidate_set(
        candidates: List[str],
        probabilities: List[float],
        lambda_val: float,
    ) -> List[Tuple[str, float]]:
        """
        Constructs prediction set C_lambda(X) = { y_k : p(y_k|X) >= 1 - lambda_val }.
        As lambda_val increases, 1 - lambda_val decreases, expanding the set.
        """
        threshold = 1.0 - lambda_val
        selected = [
            (cand, prob)
            for cand, prob in zip(candidates, probabilities)
            if prob >= threshold
        ]
        if not selected and len(candidates) > 0:
            selected = [(candidates[0], probabilities[0])]
        return selected

    @staticmethod
    def evaluate_set_coverage(
        prediction_sets: List[List[str]],
        ground_truths: List[str],
    ) -> Tuple[float, float]:
        """
        Returns: (empirical_coverage, avg_set_size)
        """
        n = len(ground_truths)
        if n == 0:
            return 0.0, 0.0

        hits = 0
        total_size = 0
        for pred_set, gt in zip(prediction_sets, ground_truths):
            gt_clean = gt.strip().lower()
            hit = any(gt_clean == p.strip().lower() or gt_clean in p.strip().lower() for p in pred_set)
            if hit:
                hits += 1
            total_size += len(pred_set)

        coverage = hits / n
        avg_size = total_size / n
        return coverage, avg_size
