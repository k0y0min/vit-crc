import torch
from dataset_loader import load_and_split_dataset
from vlm_qlora import VLMQLoRA
from qwen_vl_utils import process_vision_info

vlm = VLMQLoRA(model_id="Qwen/Qwen3-VL-4B-Instruct", load_in_4bit=True)
processor = vlm.processor
train_ds, _, _ = load_and_split_dataset("pope", max_train_samples=2)

item = train_ds[0]
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
        "content": [{"type": "text", "text": item["answer"]}],
    },
]

full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
prompt_text = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
imgs, _ = process_vision_info(messages)

inputs = processor(text=[full_text], images=imgs, padding=True, return_tensors="pt").to(vlm.device)
prompt_inputs = processor(text=[prompt_text], images=imgs, padding=True, return_tensors="pt").to(vlm.device)

labels = inputs["input_ids"].clone()
prompt_len = prompt_inputs["input_ids"].shape[1]
labels[:, :prompt_len] = -100

model = vlm.setup_lora_for_training(r=16, lora_alpha=32)
outputs = model(**inputs, labels=labels)
print(f"Full tokens: {inputs['input_ids'].shape[1]} | Prompt tokens: {prompt_len} | Target tokens: {inputs['input_ids'].shape[1] - prompt_len}")
print(f"Target loss on assistant tokens only: {outputs.loss.item():.4f}")
