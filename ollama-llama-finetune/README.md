# Ollama Llama Fine-Tuning Project

This project is a starter template for fine-tuning a **Llama-family model** with LoRA/QLoRA and packaging it for use in **Ollama**.

## What this project includes

- A reproducible training script using 🤗 Transformers + PEFT + bitsandbytes.
- Config file for training hyperparameters.
- Example JSONL instruction dataset.
- Utility script to convert adapter output into an Ollama-ready model workflow.
- `Modelfile` template to register your fine-tuned model in Ollama.

> Note: Ollama does not fine-tune models directly. You fine-tune externally, then import the merged/converted result into Ollama.

---

## Project structure

```text
ollama-llama-finetune/
├── configs/
│   └── train_config.yaml
├── data/
│   └── sample_instructions.jsonl
├── scripts/
│   ├── train_lora.py
│   └── prepare_ollama_model.py
├── Modelfile.template
├── requirements.txt
└── README.md
```

---

## 1) Setup

```bash
cd ollama-llama-finetune
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If using GPU, install the CUDA-compatible PyTorch build first from pytorch.org.

---

## 2) Prepare your dataset

The training script expects JSONL with these fields:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

`input` can be empty.

Use `data/sample_instructions.jsonl` as a reference.

---

## 3) Fine-tune

```bash
python scripts/train_lora.py --config configs/train_config.yaml
```

Training outputs:

- LoRA adapters in `outputs/adapters/`
- Tokenizer/config snapshots

Optional merge to full model:

```bash
python scripts/train_lora.py --config configs/train_config.yaml --merge-adapter
```

Merged model will be written to `outputs/merged/`.

---

## 4) Prepare for Ollama

1. Convert merged model to GGUF (if needed) with `llama.cpp` conversion tools.
2. Copy the GGUF path into a `Modelfile` based on `Modelfile.template`.
3. Build and run model in Ollama:

```bash
ollama create my-llama-ft -f Modelfile
ollama run my-llama-ft
```

If you already have a base Ollama model and only want behavior tuning via system prompt, you can skip GGUF conversion and use:

```text
FROM llama3.1
SYSTEM "Your specialized behavior instructions"
```

---

## 5) Practical tips

- Start with a small subset of data and 1 epoch to validate pipeline.
- Track eval loss and manually test prompts after each run.
- Keep LoRA rank/alpha modest for memory efficiency.
- Use quantized loading (`4-bit`) when GPU RAM is limited.

---

## 6) Example quick run

```bash
python scripts/train_lora.py \
  --config configs/train_config.yaml \
  --dataset data/sample_instructions.jsonl \
  --base-model meta-llama/Llama-3.1-8B-Instruct
```

Then package for Ollama:

```bash
python scripts/prepare_ollama_model.py \
  --merged-model-dir outputs/merged \
  --modelfile-out Modelfile
```

