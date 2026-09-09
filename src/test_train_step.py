import torch
from dataset_loader import load_and_split_dataset
from vlm_qlora import VLMQLoRA
from train import collate_fn

print("Step 1: Loading 2 training samples...", flush=True)
train_ds, _, _ = load_and_split_dataset("pope", max_train_samples=2)

print("Step 2: Initializing VLMQLoRA...", flush=True)
vlm = VLMQLoRA(model_id="Qwen/Qwen3-VL-4B-Instruct", load_in_4bit=True)
model = vlm.setup_lora_for_training(r=16, lora_alpha=32)
model.gradient_checkpointing_enable()

print("Step 3: Collating batch...", flush=True)
batch = collate_fn([train_ds[0], train_ds[1]], vlm.processor, device=vlm.device)
print(f"Batch input_ids shape: {batch['input_ids'].shape}", flush=True)

print("Step 4: Forward pass...", flush=True)
outputs = model(**batch)
loss = outputs.loss
print(f"Loss computed: {loss.item():.4f}", flush=True)

print("Step 5: Backward pass...", flush=True)
loss.backward()
print("Backward pass successful! Gradients computed.", flush=True)
vram = torch.cuda.memory_allocated() / (1024 ** 3)
print(f"Peak VRAM used: {vram:.2f} GB / 23.03 GB", flush=True)
