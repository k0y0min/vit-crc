"""
QLoRA Fine-Tuning Pipeline for Qwen3-VL-4B-Instruct
- 4-bit NF4 Quantization with BitsAndBytes
- PEFT LoRA on projection and MLP layers
- Strict prompt masking (-100) so loss is computed exclusively on assistant response
- Optimized for NVIDIA L4 GPU (24 GB VRAM)
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from qwen_vl_utils import process_vision_info
from dataset_loader import load_and_split_dataset
from vlm_qlora import VLMQLoRA


def collate_fn(batch, processor, device="cuda"):
    messages_list = []
    for item in batch:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": f"Question: {item['question']}\nProvide a direct, concise answer."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["answer"]}
                ],
            },
        ]
        messages_list.append(messages)

    full_texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages_list
    ]
    prompt_texts = [
        processor.apply_chat_template(msg[:1], tokenize=False, add_generation_prompt=True)
        for msg in messages_list
    ]

    image_inputs = []
    video_inputs = []
    for msg in messages_list:
        imgs, vids = process_vision_info(msg)
        image_inputs.extend(imgs)
        if vids:
            video_inputs.extend(vids)

    inputs = processor(
        text=full_texts,
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask prompt tokens for each sample in the batch
    for i in range(len(batch)):
        prompt_inp = processor(
            text=[prompt_texts[i]],
            images=[image_inputs[i]],
            padding=False,
            return_tensors="pt",
        )
        prompt_len = prompt_inp["input_ids"].shape[1]
        labels[i, :prompt_len] = -100

    inputs["labels"] = labels
    return {k: v.to(device) for k, v in inputs.items()}


def train_qlora(
    dataset_name: str = "pope",
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    output_dir: str = "checkpoints/qwen3vl_qlora_pope",
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    learning_rate: float = 2e-4,
    max_train_samples: int = 600,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting QLoRA Fine-Tuning ===", flush=True)
    print(f"Model: {model_id} | Dataset: {dataset_name} | Samples: {max_train_samples} | Epochs: {epochs}", flush=True)
    print(f"Device: {device} | Output Checkpoint: {output_dir}", flush=True)

    # 1. Load data
    train_ds, _, _ = load_and_split_dataset(
        dataset_name=dataset_name,
        max_train_samples=max_train_samples,
        max_calib_samples=1,
        max_test_samples=1,
    )

    # 2. Init model in 4-bit NF4
    vlm = VLMQLoRA(model_id=model_id, load_in_4bit=True, device=device)
    model = vlm.setup_lora_for_training(r=16, lora_alpha=32)
    model.gradient_checkpointing_enable()
    processor = vlm.processor

    # 3. DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor, device),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.train()
    step = 0
    print(f"Total optimization steps: {total_steps} (Batches: {len(train_loader)})", flush=True)

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            running_loss += loss.item() * grad_accum_steps

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 5 == 0 or step == total_steps:
                    lr_curr = scheduler.get_last_lr()[0]
                    avg_loss = running_loss / (5.0 if step % 5 == 0 else (step % 5 or 1))
                    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    print(
                        f"Epoch [{epoch+1}/{epochs}] Step [{step}/{total_steps}] "
                        f"Loss: {avg_loss:.4f} | LR: {lr_curr:.6f} | VRAM: {vram_gb:.2f} GB",
                        flush=True,
                    )
                    running_loss = 0.0

    print(f"\nTraining finished! Saving LoRA adapter to {output_dir}...", flush=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Checkpoint saved successfully.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset", type=str, default="pope")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen3vl_qlora_pope")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_qlora(
        dataset_name=args.dataset,
        model_id=args.model_id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        max_train_samples=args.samples,
    )
