#!/bin/bash
set -e

export PYTHONPATH=.:src:$PYTHONPATH
PYTHON=/home/ayush/vlm-crc-env/bin/python3
MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"

echo "=========================================================================="
echo " Starting Full 2B Parameter Suite: Conformal Risk Control & 4-bit QLoRA"
echo " Model: ${MODEL_ID}"
echo " Host: $(hostname) | Date: $(date)"
echo "=========================================================================="

# -------------------------------------------------------------------------
# 1. POPE (Multimodal Hallucination Guard)
# -------------------------------------------------------------------------
echo -e "\n>>> [1/6] Starting POPE (Hallucination Guard) Pipeline..."
mkdir -p results/2b_pope_base checkpoints/qwen3vl_2b_qlora_pope results/2b_pope_qlora

echo ">> 1a. POPE Zero-Shot Base CRC Evaluation..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset pope \
    --calib_samples 150 \
    --test_samples 150 \
    --output_dir results/2b_pope_base \
    > eval_2b_pope_base.log 2>&1

echo ">> 1b. POPE 4-bit QLoRA Fine-Tuning (1500 samples)..."
$PYTHON -u src/train.py \
    --model_id "$MODEL_ID" \
    --dataset pope \
    --epochs 1 \
    --samples 1500 \
    --batch_size 2 \
    --grad_accum 4 \
    --output_dir checkpoints/qwen3vl_2b_qlora_pope \
    > train_2b_pope.log 2>&1

echo ">> 1c. POPE Fine-Tuned CRC Evaluation..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset pope \
    --adapter checkpoints/qwen3vl_2b_qlora_pope \
    --calib_samples 150 \
    --test_samples 150 \
    --output_dir results/2b_pope_qlora \
    > eval_2b_pope_qlora.log 2>&1
echo ">> POPE Completed Successfully!"

# -------------------------------------------------------------------------
# 2. Defect Detection (Industrial Surface Inspection)
# -------------------------------------------------------------------------
echo -e "\n>>> [2/6] Starting Defect Detection Pipeline..."
mkdir -p results/2b_defect_base checkpoints/qwen3vl_2b_qlora_defect results/2b_defect_qlora

echo ">> 2a. Defect Detection Zero-Shot Base CRC Evaluation..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset defect \
    --calib_samples 200 \
    --test_samples 200 \
    --output_dir results/2b_defect_base \
    > eval_2b_defect_base.log 2>&1

echo ">> 2b. Defect Detection 4-bit QLoRA Fine-Tuning (1500 samples)..."
$PYTHON -u src/train.py \
    --model_id "$MODEL_ID" \
    --dataset defect \
    --epochs 1 \
    --samples 1500 \
    --batch_size 2 \
    --grad_accum 4 \
    --output_dir checkpoints/qwen3vl_2b_qlora_defect \
    > train_2b_defect.log 2>&1

echo ">> 2c. Defect Detection Fine-Tuned CRC Evaluation..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset defect \
    --adapter checkpoints/qwen3vl_2b_qlora_defect \
    --calib_samples 200 \
    --test_samples 200 \
    --output_dir results/2b_defect_qlora \
    > eval_2b_defect_qlora.log 2>&1
echo ">> Defect Detection Completed Successfully!"

# -------------------------------------------------------------------------
# 3. CORD-v2 (Fintech Invoice Totals)
# -------------------------------------------------------------------------
echo -e "\n>>> [3/6] Starting CORD-v2 Pipeline..."
mkdir -p results/2b_cord_base checkpoints/qwen3vl_2b_qlora_cord results/2b_cord_qlora

echo ">> 3a. CORD-v2 Zero-Shot Base CRC Evaluation..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset cord \
    --calib_samples 100 \
    --test_samples 100 \
    --output_dir results/2b_cord_base \
    > eval_2b_cord_base.log 2>&1

echo ">> 3b. CORD-v2 4-bit QLoRA Fine-Tuning (800 samples)..."
$PYTHON -u src/train.py \
    --model_id "$MODEL_ID" \
    --dataset cord \
    --epochs 1 \
    --samples 800 \
    --batch_size 2 \
    --grad_accum 4 \
    --output_dir checkpoints/qwen3vl_2b_qlora_cord \
    > train_2b_cord.log 2>&1

echo ">> 3c. CORD-v2 Fine-Tuned CRC Evaluation..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset cord \
    --adapter checkpoints/qwen3vl_2b_qlora_cord \
    --calib_samples 100 \
    --test_samples 100 \
    --output_dir results/2b_cord_qlora \
    > eval_2b_cord_qlora.log 2>&1
echo ">> CORD-v2 Completed Successfully!"

# -------------------------------------------------------------------------
# 4. PathVQA Binarized (Clinical Biopsy Triage)
# -------------------------------------------------------------------------
echo -e "\n>>> [4/6] Starting PathVQA Binarized Pipeline..."
mkdir -p results/2b_pathvqa_bin_base checkpoints/qwen3vl_2b_qlora_pathvqa_binarized results/2b_pathvqa_bin_qlora

echo ">> 4a. PathVQA Zero-Shot Base CRC Evaluation..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset pathvqa_binarized \
    --calib_samples 100 \
    --test_samples 100 \
    --output_dir results/2b_pathvqa_bin_base \
    > eval_2b_pathvqa_base.log 2>&1

echo ">> 4b. PathVQA 4-bit QLoRA Fine-Tuning (800 samples)..."
$PYTHON -u src/train.py \
    --model_id "$MODEL_ID" \
    --dataset pathvqa_binarized \
    --epochs 1 \
    --samples 800 \
    --batch_size 2 \
    --grad_accum 4 \
    --output_dir checkpoints/qwen3vl_2b_qlora_pathvqa_binarized \
    > train_2b_pathvqa.log 2>&1

echo ">> 4c. PathVQA Fine-Tuned CRC Evaluation..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset pathvqa_binarized \
    --adapter checkpoints/qwen3vl_2b_qlora_pathvqa_binarized \
    --calib_samples 100 \
    --test_samples 100 \
    --output_dir results/2b_pathvqa_bin_qlora \
    > eval_2b_pathvqa_qlora.log 2>&1
echo ">> PathVQA Binarized Completed Successfully!"

# -------------------------------------------------------------------------
# 5. Hateful Memes (Trust & Safety Content Moderation)
# -------------------------------------------------------------------------
echo -e "\n>>> [5/6] Starting Hateful Memes Pipeline..."
mkdir -p results/2b_hateful_base checkpoints/qwen3vl_2b_qlora_hateful results/2b_hateful_qlora

echo ">> 5a. Hateful Memes Zero-Shot Base CRC Evaluation..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset hateful \
    --calib_samples 200 \
    --test_samples 200 \
    --output_dir results/2b_hateful_base \
    > eval_2b_hateful_base.log 2>&1

echo ">> 5b. Hateful Memes 4-bit QLoRA Fine-Tuning (1000 samples)..."
$PYTHON -u src/train.py \
    --model_id "$MODEL_ID" \
    --dataset hateful \
    --epochs 1 \
    --samples 1000 \
    --batch_size 2 \
    --grad_accum 4 \
    --output_dir checkpoints/qwen3vl_2b_qlora_hateful \
    > train_2b_hateful.log 2>&1

echo ">> 5c. Hateful Memes Fine-Tuned CRC Evaluation..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset hateful \
    --adapter checkpoints/qwen3vl_2b_qlora_hateful \
    --calib_samples 200 \
    --test_samples 200 \
    --output_dir results/2b_hateful_qlora \
    > eval_2b_hateful_qlora.log 2>&1
echo ">> Hateful Memes Completed Successfully!"

# -------------------------------------------------------------------------
# 6. ChartQA (Financial Visual Reasoning & Analytics)
# -------------------------------------------------------------------------
echo -e "\n>>> [6/6] Starting ChartQA Pipeline..."
mkdir -p results/2b_chartqa_base checkpoints/qwen3vl_2b_qlora_chartqa results/2b_chartqa_qlora

echo ">> 6a. ChartQA Zero-Shot Base CRC Evaluation (with Temp Scaling)..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset chartqa \
    --calib_samples 100 \
    --test_samples 100 \
    --use_temp_scaling \
    --output_dir results/2b_chartqa_base \
    > eval_2b_chartqa_base.log 2>&1

echo ">> 6b. ChartQA 4-bit QLoRA Fine-Tuning (1500 samples)..."
$PYTHON -u src/train.py \
    --model_id "$MODEL_ID" \
    --dataset chartqa \
    --epochs 1 \
    --samples 1500 \
    --batch_size 2 \
    --grad_accum 4 \
    --output_dir checkpoints/qwen3vl_2b_qlora_chartqa \
    > train_2b_chartqa.log 2>&1

echo ">> 6c. ChartQA Fine-Tuned CRC Evaluation (with Temp Scaling)..."
$PYTHON -u src/evaluate_crc.py \
    --model_id "$MODEL_ID" \
    --dataset chartqa \
    --adapter checkpoints/qwen3vl_2b_qlora_chartqa \
    --calib_samples 100 \
    --test_samples 100 \
    --use_temp_scaling \
    --output_dir results/2b_chartqa_qlora \
    > eval_2b_chartqa_qlora.log 2>&1
echo ">> ChartQA Completed Successfully!"

echo -e "\n=========================================================================="
echo " All 6 Benchmark Domains Completed Successfully for 2B Model!"
echo " Date: $(date)"
echo "=========================================================================="
