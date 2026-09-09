"""
Modular Dataset Loader for VLM Conformal Risk Control
Supports:
- POPE ('lmms-lab/POPE'): Object hallucination probing (Adversarial, Popular, Random)
- ChartQA ('HuggingFaceM4/ChartQA'): Enterprise analytical chart reasoning

Standardizes splits into:
- Train split: for QLoRA fine-tuning
- Calibration split: for CRC threshold fitting
- Test split: for empirical held-out statistical verification
"""

import os
import json
from typing import Tuple
from datasets import load_dataset, Dataset
from PIL import Image


class LazyTransformedDataset:
    """Zero-copy on-demand transformer to prevent OOM on high-resolution image datasets."""
    def __init__(self, hf_dataset, transform_fn):
        self.hf_dataset = hf_dataset
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        return self.transform_fn(self.hf_dataset[idx], idx)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


def load_and_split_dataset(
    dataset_name: str = "pope",
    max_train_samples: int = 1500,
    max_calib_samples: int = 500,
    max_test_samples: int = 500,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads dataset and returns standardized (train, calib, test) splits.
    Standardized schema:
        {
            'image': PIL.Image,
            'question': str,
            'answer': str,
            'id': str,
            'category': str (optional)
        }
    """
    name_lower = dataset_name.lower()
    print(f"Loading dataset '{dataset_name}' from Hugging Face...")

    if "pope" in name_lower:
        # POPE dataset: 9000 items in 'test' split (adversarial, popular, random)
        ds = load_dataset("lmms-lab/POPE", split="test")
        shuffled = ds.shuffle(seed=seed)

        total_needed = max_train_samples + max_calib_samples + max_test_samples
        if len(shuffled) < total_needed:
            # Scale proportionally if needed
            n_total = len(shuffled)
            n_train = int(n_total * 0.6)
            n_calib = int(n_total * 0.2)
            n_test = n_total - n_train - n_calib
        else:
            n_train = max_train_samples
            n_calib = max_calib_samples
            n_test = max_test_samples

        raw_train = shuffled.select(range(0, n_train))
        raw_calib = shuffled.select(range(n_train, n_train + n_calib))
        raw_test = shuffled.select(range(n_train + n_calib, n_train + n_calib + n_test))

        def standardize_pope(example, idx):
            img = example["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")

            q = str(example.get("question", "")).strip()
            ans = str(example.get("answer", "")).strip().lower()
            cat = str(example.get("category", "general"))

            return {
                "image": img,
                "question": q,
                "answer": ans,
                "id": f"pope_{idx}",
                "category": cat,
            }

        train_ds = raw_train.map(standardize_pope, with_indices=True)
        calib_ds = raw_calib.map(standardize_pope, with_indices=True)
        test_ds = raw_test.map(standardize_pope, with_indices=True)

    elif "chartqa" in name_lower:
        ds = load_dataset("HuggingFaceM4/ChartQA")
        raw_train = ds["train"]
        raw_val = ds["val"]
        raw_test = ds["test"]

        def standardize_chartqa(example, idx):
            img = example["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")

            q = str(example.get("query", "")).strip()
            raw_ans = example.get("label", [""])
            ans = str(raw_ans[0] if isinstance(raw_ans, list) and len(raw_ans) > 0 else raw_ans).strip().lower()

            return {
                "image": img,
                "question": q,
                "answer": ans,
                "id": f"chartqa_{idx}",
                "category": "chart",
            }

        n_train = min(len(raw_train), max_train_samples)
        n_calib = min(len(raw_val), max_calib_samples)
        n_test = min(len(raw_test), max_test_samples)

        train_ds = raw_train.shuffle(seed=seed).select(range(n_train)).map(standardize_chartqa, with_indices=True)
        calib_ds = raw_val.shuffle(seed=seed).select(range(n_calib)).map(standardize_chartqa, with_indices=True)
        test_ds = raw_test.shuffle(seed=seed).select(range(n_test)).map(standardize_chartqa, with_indices=True)

    elif "path" in name_lower and "vqa" in name_lower:
        ds = load_dataset("flaviagiammarino/path-vqa")
        raw_train = ds["train"]
        raw_val = ds["validation"]
        raw_test = ds["test"]

        is_bin_mode = "bin" in name_lower
        is_open_mode = "normal" in name_lower or "open" in name_lower

        if is_bin_mode or is_open_mode:
            target_set = {"yes", "no"}
            # Fast index filtering without decoding images
            def get_matching_indices(split):
                answers = split.select_columns(["answer"])["answer"]
                if is_bin_mode:
                    return [i for i, a in enumerate(answers) if str(a).strip().lower() in target_set]
                else:
                    return [i for i, a in enumerate(answers) if str(a).strip().lower() not in target_set]

            raw_train = raw_train.select(get_matching_indices(raw_train))
            raw_val = raw_val.select(get_matching_indices(raw_val))
            raw_test = raw_test.select(get_matching_indices(raw_test))

        def standardize_pathvqa(example, idx):
            img = example["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")
            return {
                "image": img,
                "question": str(example.get("question", "")).strip(),
                "answer": str(example.get("answer", "")).strip().lower(),
                "id": f"pathvqa_{idx}",
                "category": "pathology",
            }

        n_train = min(len(raw_train), max_train_samples)
        n_calib = min(len(raw_val), max_calib_samples)
        n_test = min(len(raw_test), max_test_samples)

        train_ds = raw_train.shuffle(seed=seed).select(range(n_train)).map(standardize_pathvqa, with_indices=True)
        calib_ds = raw_val.shuffle(seed=seed).select(range(n_calib)).map(standardize_pathvqa, with_indices=True)
        test_ds = raw_test.shuffle(seed=seed).select(range(n_test)).map(standardize_pathvqa, with_indices=True)

    elif "hateful" in name_lower:
        ds = load_dataset("Multimodal-Fatima/Hatefulmemes_train", split="train")
        shuffled = ds.shuffle(seed=seed)

        n_total = len(shuffled)
        n_train = min(max_train_samples, int(n_total * 0.7))
        n_calib = min(max_calib_samples, int(n_total * 0.15))
        n_test = min(max_test_samples, n_total - n_train - n_calib)

        raw_train = shuffled.select(range(0, n_train))
        raw_calib = shuffled.select(range(n_train, n_train + n_calib))
        raw_test = shuffled.select(range(n_train + n_calib, n_train + n_calib + n_test))

        def standardize_hateful(example, idx):
            img = example["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")

            meme_text = str(example.get("text", "")).strip()
            q = f"Does this multimodal meme contain hate speech, harassment, or policy violation? Meme caption: \"{meme_text}\""
            ans = "yes" if example.get("label") == 1 else "no"

            return {
                "image": img,
                "question": q,
                "answer": ans,
                "id": f"hateful_{idx}",
                "category": "safety",
            }

        train_ds = raw_train.map(standardize_hateful, with_indices=True)
        calib_ds = raw_calib.map(standardize_hateful, with_indices=True)
        test_ds = raw_test.map(standardize_hateful, with_indices=True)

    elif "defect" in name_lower or "mvtec" in name_lower:
        ds = load_dataset("iluvvatar/wood_surface_defects", split="train")
        shuffled = ds.shuffle(seed=seed)

        n_total = len(shuffled)
        n_train = min(max_train_samples, int(n_total * 0.7))
        n_calib = min(max_calib_samples, int(n_total * 0.15))
        n_test = min(max_test_samples, n_total - n_train - n_calib)

        raw_train = shuffled.select(range(0, n_train))
        raw_calib = shuffled.select(range(n_train, n_train + n_calib))
        raw_test = shuffled.select(range(n_train + n_calib, n_train + n_calib + n_test))

        def standardize_defect(example, idx):
            img = example["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")

            # Downsample high-res industrial camera scans (2800x1024) to max 1024 to optimize VRAM and RAM
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            objs = example.get("objects", [])
            has_defect = len(objs) > 0
            q = "Is there a physical surface defect or structural anomaly present on this manufacturing component?"
            ans = "yes" if has_defect else "no"

            return {
                "image": img,
                "question": q,
                "answer": ans,
                "id": f"defect_{idx}",
                "category": "defect",
            }

        train_ds = LazyTransformedDataset(raw_train, standardize_defect)
        calib_ds = LazyTransformedDataset(raw_calib, standardize_defect)
        test_ds = LazyTransformedDataset(raw_test, standardize_defect)

    elif "cord" in name_lower or "receipt" in name_lower or "sroie" in name_lower:
        ds = load_dataset("naver-clova-ix/cord-v2")
        raw_train = ds["train"]
        raw_val = ds["validation"]
        raw_test = ds["test"]

        def extract_total(gt_str):
            try:
                gt = json.loads(gt_str).get("gt_parse", {})
                tot = gt.get("total", {}).get("total_price", "")
                if not tot:
                    tot = gt.get("sub_total", {}).get("subtotal_price", "")
                return str(tot).strip()
            except Exception:
                return ""

        def standardize_cord(example, idx):
            img = example["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")

            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            gt_str = example.get("ground_truth", "{}")
            tot_ans = extract_total(gt_str)
            if not tot_ans:
                tot_ans = "unknown"

            q = "What is the total price on this receipt?"

            return {
                "image": img,
                "question": q,
                "answer": tot_ans,
                "id": f"cord_{idx}",
                "category": "fintech",
            }

        n_train = min(len(raw_train), max_train_samples)
        n_calib = min(len(raw_val), max_calib_samples)
        n_test = min(len(raw_test), max_test_samples)

        train_ds = LazyTransformedDataset(raw_train.shuffle(seed=seed).select(range(n_train)), standardize_cord)
        calib_ds = LazyTransformedDataset(raw_val.shuffle(seed=seed).select(range(n_calib)), standardize_cord)
        test_ds = LazyTransformedDataset(raw_test.shuffle(seed=seed).select(range(n_test)), standardize_cord)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from: 'pope', 'chartqa', 'pathvqa_binarized', 'pathvqa_normal', 'hateful', 'defect', 'cord'.")

    print(f"Dataset split complete: Train={len(train_ds)}, Calib={len(calib_ds)}, Test={len(test_ds)}")
    return train_ds, calib_ds, test_ds


if __name__ == "__main__":
    t, c, te = load_and_split_dataset("pope", max_train_samples=20, max_calib_samples=10, max_test_samples=10)
    print("POPE item:", t[0]["question"], "->", t[0]["answer"])
    tc, cc, tec = load_and_split_dataset("chartqa", max_train_samples=20, max_calib_samples=10, max_test_samples=10)
    print("ChartQA item:", tc[0]["question"], "->", tc[0]["answer"])
