"""
Comparative Model Scaling Analysis: Qwen3-VL-2B vs Qwen3-VL-4B
Extracts all Conformal Risk Control metrics and generates side-by-side markdown comparison tables.
"""

import os
import json
import glob

DATASETS = [
    ("defect", "Defect Detection", "Industrial QA"),
    ("cord", "CORD-v2", "Invoice Extraction"),
    ("pathvqa_bin", "PathVQA Binarized", "Clinical Biopsy Triage"),
    ("hateful", "Hateful Memes", "Trust & Safety Moderation"),
    ("pope", "POPE", "Hallucination Guard"),
    ("chartqa", "ChartQA", "Financial Analytics"),
]

def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def get_metric_at_alpha(data, target_alpha):
    if not data or "benchmarks" not in data:
        return None
    for b in data["benchmarks"]:
        if abs(b["alpha"] - target_alpha) < 1e-4:
            return b
    return None

def main():
    print("# 📊 Corporate Benchmark Suite: 2B vs 4B Model Scaling & CRC Reliability\n")

    # Table 1: 2B vs 4B Model Scaling Master Table
    print("### Master Scaling & Reliability Comparison (2B vs 4B)")
    print("| Domain & Vertical | Model Size | Base Acc | QLoRA Acc | Lift (Δ) | Guaranteed Risk (α=0.01) | STP Rate (@ α=0.01) | STP Rate (@ α=0.05) | Acc @ α=0.05 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for key, name, vertical in DATASETS:
        # Load 4B paths
        if key == "pope":
            p_4b_base = "results/pope_crc_results.json"
            p_4b_qlora = "results/pope_qlora/pope_crc_results.json"
        elif key == "chartqa":
            p_4b_base = "results/chartqa/chartqa_crc_results.json"
            p_4b_qlora = "results/chartqa_qlora/chartqa_crc_results.json"
        elif key == "pathvqa_bin":
            p_4b_base = "results/pathvqa_bin_base/pathvqa_binarized_crc_results.json"
            p_4b_qlora = "results/pathvqa_bin_qlora/pathvqa_binarized_crc_results.json"
        else:
            p_4b_base = f"results/{key}_base/{key}_crc_results.json"
            p_4b_qlora = f"results/{key}_qlora/{key}_crc_results.json"

        d_4b_base = load_json(p_4b_base)
        d_4b_qlora = load_json(p_4b_qlora)

        # Load 2B paths
        if key == "pathvqa_bin":
            p_2b_base = "results/2b_pathvqa_bin_base/pathvqa_binarized_crc_results.json"
            p_2b_qlora = "results/2b_pathvqa_bin_qlora/pathvqa_binarized_crc_results.json"
        else:
            p_2b_base = f"results/2b_{key}_base/{key}_crc_results.json"
            p_2b_qlora = f"results/2b_{key}_qlora/{key}_crc_results.json"

        d_2b_base = load_json(p_2b_base)
        d_2b_qlora = load_json(p_2b_qlora)

        # Process 2B row
        if d_2b_qlora:
            base_acc_2b = d_2b_base.get("base_accuracy", 0.0) if d_2b_base else 0.0
            qlora_acc_2b = d_2b_qlora.get("base_accuracy", 0.0)
            lift_2b = qlora_acc_2b - base_acc_2b
            m01_2b = get_metric_at_alpha(d_2b_qlora, 0.01)
            m05_2b = get_metric_at_alpha(d_2b_qlora, 0.05)

            risk01_2b = f"{m01_2b['empirical_risk']:.4f} {'✅' if m01_2b['guarantee_satisfied'] else '❌'}" if m01_2b else "N/A"
            stp01_2b = f"{m01_2b['straight_through_rate']*100:.1f}%" if m01_2b else "N/A"
            stp05_2b = f"{m05_2b['straight_through_rate']*100:.1f}%" if m05_2b else "N/A"
            acc05_2b = f"{m05_2b['accepted_accuracy']*100:.1f}%" if m05_2b else "N/A"

            print(f"| **{name}** ({vertical}) | **2B** | {base_acc_2b:.1f}% | **{qlora_acc_2b:.1f}%** | +{lift_2b:.1f}% | {risk01_2b} | {stp01_2b} | **{stp05_2b}** | {acc05_2b} |")

        # Process 4B row
        if d_4b_qlora:
            base_acc_4b = d_4b_base.get("base_accuracy", 0.0) if d_4b_base else 0.0
            qlora_acc_4b = d_4b_qlora.get("base_accuracy", 0.0)
            lift_4b = qlora_acc_4b - base_acc_4b
            m01_4b = get_metric_at_alpha(d_4b_qlora, 0.01)
            m05_4b = get_metric_at_alpha(d_4b_qlora, 0.05)

            risk01_4b = f"{m01_4b['empirical_risk']:.4f} {'✅' if m01_4b['guarantee_satisfied'] else '❌'}" if m01_4b else "N/A"
            stp01_4b = f"{m01_4b['straight_through_rate']*100:.1f}%" if m01_4b else "N/A"
            stp05_4b = f"{m05_4b['straight_through_rate']*100:.1f}%" if m05_4b else "N/A"
            acc05_4b = f"{m05_4b['accepted_accuracy']*100:.1f}%" if m05_4b else "N/A"

            print(f"| | **4B** | {base_acc_4b:.1f}% | **{qlora_acc_4b:.1f}%** | +{lift_4b:.1f}% | {risk01_4b} | {stp01_4b} | **{stp05_4b}** | {acc05_4b} |")

if __name__ == "__main__":
    main()
