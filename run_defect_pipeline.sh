#!/bin/bash
set -e

echo "=== Starting Defect Detection Fine-Tuning ==="
/home/ayush/vlm-crc-env/bin/python3 -u src/train.py \
    --dataset defect \
    --epochs 1 \
    --samples 1500 \
    --batch_size 2 \
    --grad_accum 4 \
    --output_dir checkpoints/qwen3vl_qlora_defect \
    > train_defect.log 2>&1

echo "=== Defect Fine-Tuning Complete. Starting Conformal Risk Control Evaluation ==="
/home/ayush/vlm-crc-env/bin/python3 -u src/evaluate_crc.py \
    --dataset defect \
    --adapter checkpoints/qwen3vl_qlora_defect \
    --calib_samples 200 \
    --test_samples 200 \
    --output_dir results/defect_qlora \
    > eval_defect_qlora.log 2>&1

echo "=== Defect QLoRA + CRC Evaluation Complete ==="
