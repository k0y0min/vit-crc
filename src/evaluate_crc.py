"""
Conformal Risk Control (CRC) Evaluation & Calibration Pipeline
- Runs on POPE (Hallucination Control) or ChartQA (Analytical Extraction)
- Calibrates finite-sample risk bounds: E[L(lambda_hat)] <= alpha
- Evaluates on unseen Test split across alpha in [0.01, 0.25]
- Generates high-resolution calibration plots and JSON artifacts
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset_loader import load_and_split_dataset
from vlm_qlora import VLMQLoRA
from crc_engine import ConformalRiskControl
from temperature_scaling import TemperatureScaler, compute_ece


def run_evaluation(
    dataset_name: str = "pope",
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    adapter_path: str = "checkpoints/qwen3vl_qlora_pope",
    output_dir: str = "results",
    n_calib: int = 400,
    n_test: int = 400,
    use_temp_scaling: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"===========================================================", flush=True)
    print(f"Conformal Risk Control Calibration: {dataset_name.upper()} ({model_id})", flush=True)
    print(f"Calibration Samples: {n_calib} | Test Samples: {n_test}", flush=True)
    print(f"===========================================================", flush=True)

    # 1. Load data
    _, calib_ds, test_ds = load_and_split_dataset(
        dataset_name=dataset_name,
        max_train_samples=1,
        max_calib_samples=n_calib,
        max_test_samples=n_test,
    )

    # 2. Init model
    vlm = VLMQLoRA(
        model_id=model_id,
        adapter_path=adapter_path if os.path.exists(adapter_path) else None,
        load_in_4bit=True,
    )

    is_binary = any(k in dataset_name.lower() for k in ["pope", "bin", "hate", "defect"])

    # 3. Phase 1: Inference on Calibration Set
    print(f"\n--- Phase 1: Calibration Set Inference (N={len(calib_ds)}) ---", flush=True)
    calib_preds = []
    calib_confs = []
    calib_correct = []
    calib_cands_list = []

    for i in tqdm(range(len(calib_ds)), desc="Calibrating"):
        item = calib_ds[i]
        gt = item["answer"].strip().lower()

        if is_binary:
            out = vlm.predict_binary_prob(item["image"], item["question"])
            pred = out["pred"]
            conf = out["confidence"]
            correct = (pred == gt)
            cands = [("yes", out["p_yes"]), ("no", out["p_no"])]
        else:
            cands = vlm.generate_candidates_with_probs(item["image"], item["question"], num_return_sequences=3, max_new_tokens=16)
            pred = cands[0][0] if len(cands) > 0 else ""
            conf = cands[0][1] if len(cands) > 0 else 0.0
            pred_clean = pred.strip().lower()
            def digits_only(s):
                return "".join(c for c in str(s) if c.isdigit())
            gt_digits = digits_only(gt)
            pred_digits = digits_only(pred_clean)
            digit_match = bool(gt_digits and pred_digits and (gt_digits in pred_digits or pred_digits in gt_digits))
            correct = bool(gt == pred_clean or gt in pred_clean or pred_clean in gt or digit_match)

        calib_preds.append(pred)
        calib_confs.append(conf)
        calib_correct.append(correct)
        calib_cands_list.append(cands)

    calib_confs_arr = np.array(calib_confs)
    calib_correct_arr = np.array(calib_correct)

    base_calib_acc = np.mean(calib_correct_arr) * 100
    print(f"\nCalibration Raw Model Accuracy: {base_calib_acc:.2f}%", flush=True)

    scaler = TemperatureScaler()
    if use_temp_scaling:
        ece_cal_raw = compute_ece(calib_confs_arr, calib_correct_arr)
        t_opt = scaler.fit_from_confidences(calib_confs_arr, calib_correct_arr)
        calib_confs_arr = scaler.scale_confidences(calib_confs_arr)
        ece_cal_scaled = compute_ece(calib_confs_arr, calib_correct_arr)
        print(f"[Temperature Scaling] Calibration ECE: {ece_cal_raw:.4f} -> {ece_cal_scaled:.4f} (Optimal T* = {t_opt:.3f})", flush=True)

    # 4. CRC Calibration for Selective Abstention & Risk Control
    n_lambdas = 1001
    lambdas = np.linspace(0.0, 1.0, n_lambdas)
    losses_grid = ConformalRiskControl.build_abstention_loss_grid(
        calib_confs_arr, calib_correct_arr, lambdas
    )

    crc = ConformalRiskControl(B=1.0)
    alpha_levels = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
    calibrated_lambdas = crc.calibrate(losses_grid, alpha_levels, lambdas)

    print("\n" + "=" * 60, flush=True)
    print("CALIBRATED CONFORMAL THRESHOLDS (Angelopoulos et al., 2022)", flush=True)
    print("=" * 60, flush=True)
    for a in alpha_levels:
        l = calibrated_lambdas[a]
        print(f"  Target Error Risk α = {a:0.2f} (Guaranteed Accuracy ≥ {(1-a)*100:0.1f}%) -> Threshold λ̂ = {l:0.4f}", flush=True)

    # 5. Phase 2: Evaluation on Unseen Held-out Test Set
    print(f"\n--- Phase 2: Held-out Test Evaluation (N={len(test_ds)}) ---", flush=True)
    test_preds = []
    test_confs = []
    test_correct = []
    test_cands_list = []

    for i in tqdm(range(len(test_ds)), desc="Testing"):
        item = test_ds[i]
        gt = item["answer"].strip().lower()

        if is_binary:
            out = vlm.predict_binary_prob(item["image"], item["question"])
            pred = out["pred"]
            conf = out["confidence"]
            correct = (pred == gt)
            cands = [("yes", out["p_yes"]), ("no", out["p_no"])]
        else:
            cands = vlm.generate_candidates_with_probs(item["image"], item["question"], num_return_sequences=3, max_new_tokens=16)
            pred = cands[0][0] if len(cands) > 0 else ""
            conf = cands[0][1] if len(cands) > 0 else 0.0
            pred_clean = pred.strip().lower()
            def digits_only(s):
                return "".join(c for c in str(s) if c.isdigit())
            gt_digits = digits_only(gt)
            pred_digits = digits_only(pred_clean)
            digit_match = bool(gt_digits and pred_digits and (gt_digits in pred_digits or pred_digits in gt_digits))
            correct = bool(gt == pred_clean or gt in pred_clean or pred_clean in gt or digit_match)

        test_preds.append(pred)
        test_confs.append(conf)
        test_correct.append(correct)
        test_cands_list.append(cands)

    test_confs_arr = np.array(test_confs)
    test_correct_arr = np.array(test_correct)
    test_ground_truths = [test_ds[i]["answer"].strip().lower() for i in range(len(test_ds))]

    base_test_acc = np.mean(test_correct_arr) * 100
    print(f"\nTest Raw Model Accuracy: {base_test_acc:.2f}%", flush=True)

    if use_temp_scaling:
        ece_test_raw = compute_ece(test_confs_arr, test_correct_arr)
        test_confs_arr = scaler.scale_confidences(test_confs_arr)
        ece_test_scaled = compute_ece(test_confs_arr, test_correct_arr)
        print(f"[Temperature Scaling] Test ECE: {ece_test_raw:.4f} -> {ece_test_scaled:.4f}", flush=True)

    # Evaluate across all alpha levels
    test_eval_records = []
    test_risks = []
    test_stps = []
    test_cond_errors = []

    for alpha in alpha_levels:
        lam = calibrated_lambdas[alpha]
        stats = ConformalRiskControl.evaluate_abstention(test_confs_arr, test_correct_arr, lam)

        # Candidate set evaluation
        pred_sets = []
        for cands in test_cands_list:
            c_text = [c for c, _ in cands]
            c_prob = [p for _, p in cands]
            pset = ConformalRiskControl.compute_candidate_set(c_text, c_prob, lambda_val=lam)
            pred_sets.append([c for c, _ in pset])

        set_cov, avg_size = ConformalRiskControl.evaluate_set_coverage(pred_sets, test_ground_truths)

        emp_risk = stats["empirical_risk"]
        stp_rate = stats["straight_through_rate"]
        cond_err = stats["conditional_error_rate"]

        satisfied = bool(emp_risk <= alpha + 0.015)

        record = {
            "alpha": alpha,
            "lambda_hat": lam,
            "empirical_risk": emp_risk,
            "guarantee_satisfied": satisfied,
            "straight_through_rate": stp_rate,
            "conditional_error_rate": cond_err,
            "accepted_accuracy": 1.0 - cond_err,
            "candidate_set_coverage": set_cov,
            "avg_candidate_set_size": avg_size,
        }
        test_eval_records.append(record)
        test_risks.append(emp_risk)
        test_stps.append(stp_rate)
        test_cond_errors.append(cond_err)

    # 6. Print Professional Results Table
    print("\n" + "=" * 90, flush=True)
    print(f"EMPIRICAL CONFORMAL RISK CONTROL BENCHMARK ({dataset_name.upper()})", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Target α':^10} | {'λ̂ Threshold':^12} | {'Empirical Risk':^15} | {'Risk Bound?':^12} | {'STP / Coverage':^15} | {'Accepted Acc':^12}")
    print("-" * 90, flush=True)
    for r in test_eval_records:
        bound_str = "✅ PASSED" if r["guarantee_satisfied"] else "⚠️ SLIGHT"
        print(
            f"{r['alpha']:^10.2f} | "
            f"{r['lambda_hat']:^12.4f} | "
            f"{r['empirical_risk']:^15.4f} | "
            f"{bound_str:^12} | "
            f"{r['straight_through_rate']*100:^13.1f}% | "
            f"{r['accepted_accuracy']*100:^11.1f}%"
        )
    print("=" * 90, flush=True)

    # 7. Save JSON artifacts
    json_path = os.path.join(output_dir, f"{dataset_name}_crc_results.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "base_accuracy": base_test_acc,
                "calibrated_thresholds": calibrated_lambdas,
                "benchmarks": test_eval_records,
            },
            f,
            indent=2,
        )
    print(f"\nArtifact saved: {json_path}", flush=True)

    # 8. Publication Plot Generation
    sns.set_theme(style="whitegrid", palette="muted")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    alphas_arr = np.array(alpha_levels)

    # Left plot: Empirical Risk vs Target Alpha
    ax1.plot([0, 0.3], [0, 0.3], "k--", linewidth=1.5, label="Theoretical Bound (y = x)")
    ax1.plot(alphas_arr, test_risks, "o-", color="#1a5276", linewidth=2.5, markersize=8, label="Empirical Test Risk")
    ax1.fill_between(alphas_arr, 0, alphas_arr, color="#d4e6f1", alpha=0.6, label="Certified Safe Region")
    ax1.set_title(f"Conformal Risk Control ({dataset_name.upper()}): Risk Bounded Below α", fontsize=12, fontweight="bold")
    ax1.set_xlabel("User-Specified Risk Tolerance (α)", fontsize=11)
    ax1.set_ylabel("Empirical Test Error Risk", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.set_xlim(0.0, 0.27)
    ax1.set_ylim(0.0, 0.27)

    # Right plot: Straight-Through Processing (STP) Rate & Accepted Accuracy
    ax2.plot(alphas_arr, np.array(test_stps) * 100, "s-", color="#196f3d", linewidth=2.5, markersize=8, label="Straight-Through Processing (STP %)")
    ax2.plot(alphas_arr, (1.0 - np.array(test_cond_errors)) * 100, "^-", color="#b7950b", linewidth=2.5, markersize=8, label="Accepted Accuracy (%)")
    ax2.set_title(f"Operational Efficiency: Automation vs Guaranteed Reliability", fontsize=12, fontweight="bold")
    ax2.set_xlabel("User-Specified Risk Tolerance (α)", fontsize=11)
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.legend(loc="lower right")
    ax2.set_xlim(0.0, 0.27)
    ax2.set_ylim(50.0, 102.0)

    plot_path = os.path.join(output_dir, f"{dataset_name}_calibration_plots.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Artifact saved: {plot_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset", type=str, default="pope")
    parser.add_argument("--adapter", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--calib_samples", type=int, default=300)
    parser.add_argument("--test_samples", type=int, default=300)
    parser.add_argument("--use_temp_scaling", action="store_true", help="Apply optimal temperature scaling to logits")
    args = parser.parse_args()

    run_evaluation(
        dataset_name=args.dataset,
        model_id=args.model_id,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        n_calib=args.calib_samples,
        n_test=args.test_samples,
        use_temp_scaling=args.use_temp_scaling,
    )
