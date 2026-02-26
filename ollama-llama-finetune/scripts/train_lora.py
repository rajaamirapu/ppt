#!/usr/bin/env python3
"""Fine-tune a Llama-family model with LoRA/QLoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Llama with LoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dataset", default=None, help="Optional dataset override")
    parser.add_argument("--base-model", default=None, help="Optional base model override")
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Merge LoRA adapter into base model after training",
    )
    return parser.parse_args()


def _torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_name, torch.bfloat16)


def load_jsonl_dataset(path: str) -> Dataset:
    records: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return Dataset.from_list(records)


def build_prompt(row: dict[str, str]) -> str:
    instruction = row.get("instruction", "").strip()
    input_text = row.get("input", "").strip()
    output = row.get("output", "").strip()

    if input_text:
        user_block = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"
    else:
        user_block = f"### Instruction:\n{instruction}"

    return f"{user_block}\n\n### Response:\n{output}"


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    base_model = args.base_model or config["base_model"]
    dataset_path = args.dataset or config["dataset_path"]
    output_dir = Path(config.get("output_dir", "outputs"))
    adapters_dir = output_dir / "adapters"
    merged_dir = output_dir / "merged"

    qconf = config.get("quantization", {})
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=qconf.get("load_in_4bit", True),
        bnb_4bit_quant_type=qconf.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=qconf.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=_torch_dtype(
            qconf.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
    )

    lconf = config["lora"]
    lora_config = LoraConfig(
        r=lconf.get("r", 16),
        lora_alpha=lconf.get("lora_alpha", 32),
        lora_dropout=lconf.get("lora_dropout", 0.05),
        target_modules=lconf.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    dataset = load_jsonl_dataset(dataset_path)

    tconf = config["training"]
    max_seq_length = int(tconf.get("max_seq_length", 1024))

    def tokenize(example: dict[str, str]) -> dict[str, Any]:
        text = build_prompt(example)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    adapters_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(adapters_dir),
        num_train_epochs=tconf.get("num_train_epochs", 1),
        per_device_train_batch_size=tconf.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=tconf.get("gradient_accumulation_steps", 8),
        learning_rate=tconf.get("learning_rate", 2e-4),
        warmup_ratio=tconf.get("warmup_ratio", 0.03),
        weight_decay=tconf.get("weight_decay", 0.01),
        max_grad_norm=tconf.get("max_grad_norm", 1.0),
        logging_steps=tconf.get("logging_steps", 10),
        save_steps=tconf.get("save_steps", 100),
        lr_scheduler_type=tconf.get("lr_scheduler_type", "cosine"),
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    trainer.train()

    model.save_pretrained(adapters_dir)
    tokenizer.save_pretrained(adapters_dir)

    if args.merge_adapter:
        merged_dir.mkdir(parents=True, exist_ok=True)
        base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
        merged_model = PeftModel.from_pretrained(base, adapters_dir).merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

    print(f"Adapter saved to: {adapters_dir}")


if __name__ == "__main__":
    main()
