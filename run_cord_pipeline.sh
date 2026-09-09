#!/bin/bash
set -e

echo "=== 1. Starting CORD-v2 Zero-Shot Baseline CRC Evaluation ==="
mkdir -p results/cord_base
/home/ayush/vlm-crc-env/bin/python3 -u src/evaluate_crc.py \
    --dataset cord \
    --calib_samples 100 \
    --test_samples 100 \
    --output_dir results/cord_base \
    > eval_cord_base.log 2>&1

echo "=== 2. Starting CORD-v2 QLoRA Fine-Tuning (800 samples, 100 steps) ==="
mkdir -p checkpoints/qwen3vl_qlora_cord
/home/ayush/vlm-crc-env/bin/python3 -u src/train.py \
    --dataset cord \
    --epochs 1 \
    --samples 800 \
    --batch_size 2 \
    --grad_accum 4 \
    --output_dir checkpoints/qwen3vl_qlora_cord \
    > train_cord.log 2>&1

echo "=== 3. Starting CORD-v2 Fine-Tuned CRC Evaluation ==="
mkdir -p results/cord_qlora
/home/ayush/vlm-crc-env/bin/python3 -u src/evaluate_crc.py \
    --dataset cord \
    --adapter checkpoints/qwen3vl_qlora_cord \
    --calib_samples 100 \
    --test_samples 100 \
    --output_dir results/cord_qlora \
    > eval_cord_qlora.log 2>&1

echo "=== CORD-v2 Pipeline Completed Successfully ==="
