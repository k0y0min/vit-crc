"""
VLM Loader & Inference Engine with 4-bit QLoRA
Supports Qwen3-VL-4B-Instruct with:
- BitsAndBytes 4-bit NF4 Quantization (under 3GB baseline VRAM)
- PEFT LoRA Adapters
- Fast logit-based confidence extraction for binary probing (POPE)
- Beam search with sequence probabilities for open-ended VQA (ChartQA)
"""

import os
import torch
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from qwen_vl_utils import process_vision_info


class VLMQLoRA:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.device = device

        print(f"Initializing VLM '{model_id}' (4-bit NF4: {load_in_4bit}) on {self.device}...", flush=True)

        self.processor = AutoProcessor.from_pretrained(model_id)

        bnb_config = None
        if load_in_4bit and "cuda" in self.device:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
            device_map="auto" if "cuda" in self.device else None,
        )

        # Cache common token IDs for fast binary evaluation
        self.tokenizer = self.processor.tokenizer
        self.yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[-1]
        self.no_id = self.tokenizer.encode("no", add_special_tokens=False)[-1]

        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading trained LoRA adapter from {adapter_path}...", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

    def setup_lora_for_training(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        """Prepares model for QLoRA fine-tuning."""
        self.model = prepare_model_for_kbit_training(self.model)
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        return self.model

    @torch.inference_mode()
    def predict_binary_prob(
        self,
        image: Image.Image,
        question: str,
    ) -> Dict[str, Any]:
        """
        Fast single-forward-pass evaluation for binary questions ('yes' vs 'no').
        Used for POPE hallucination probing.
        Returns:
            {
                'pred': 'yes' or 'no',
                'p_yes': float,
                'p_no': float,
                'confidence': float, # max(p_yes, p_no)
                'prob_dist': {'yes': p_yes, 'no': p_no}
            }
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{question} Answer only 'yes' or 'no'."},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model(**inputs)
        # Next token logits
        next_token_logits = outputs.logits[0, -1, [self.yes_id, self.no_id]]
        probs = torch.softmax(next_token_logits, dim=-1)
        p_yes = probs[0].item()
        p_no = probs[1].item()

        pred = "yes" if p_yes >= p_no else "no"
        conf = max(p_yes, p_no)

        return {
            "pred": pred,
            "p_yes": p_yes,
            "p_no": p_no,
            "confidence": conf,
            "prob_dist": {"yes": p_yes, "no": p_no},
        }

    @torch.inference_mode()
    def generate_candidates_with_probs(
        self,
        image: Image.Image,
        question: str,
        num_return_sequences: int = 5,
        max_new_tokens: int = 24,
    ) -> List[Tuple[str, float]]:
        """
        Generates candidate answers with beam search sequence log-probabilities.
        Used for open-ended VQA (ChartQA).
        Returns:
            Sorted list of (candidate_text, normalized_probability).
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {question}\nProvide a concise and direct answer."},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        num_beams = max(num_return_sequences, 5)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        sequences = outputs.sequences
        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = sequences[:, prompt_len:]
        decoded = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)

        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            log_probs = outputs.sequences_scores.cpu().numpy()
            probs = torch.softmax(torch.tensor(log_probs), dim=-1).tolist()
        else:
            probs = [1.0 / len(decoded)] * len(decoded)

        cand_dict: Dict[str, float] = {}
        for ans_text, prob in zip(decoded, probs):
            clean_ans = ans_text.strip().lower()
            if not clean_ans:
                continue
            cand_dict[clean_ans] = cand_dict.get(clean_ans, 0.0) + prob

        sorted_candidates = sorted(cand_dict.items(), key=lambda x: x[1], reverse=True)
        total_p = sum(p for _, p in sorted_candidates) or 1.0
        normalized = [(ans, p / total_p) for ans, p in sorted_candidates]

        return normalized


if __name__ == "__main__":
    vlm = VLMQLoRA()
    print("VLMQLoRA initialized successfully.")
