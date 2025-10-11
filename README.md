# Frappe LLM Training Stack

This repository contains the data preparation, supervised fine-tuning, and serving utilities used to
align Qwen2.5 Coder 3B Instruct with Frappe Framework tasks.  The tooling targets 2025 era GPUs with
bfloat16 support and integrates modern practices such as Flash Attention 2, QLoRA, and TRL's
`SFTTrainer`.

## Layout

- `data/` – JSONL corpora used for instruction tuning and evaluation.
- `scripts/` – CLI utilities for training, serving, and RAG index creation.
- `models/` – Base checkpoints and fine-tuned adapters output by the pipeline.
- `logs/` – Training logs.
- `hf_cache/` – Optional Hugging Face cache directory.

## Quickstart

```bash
python -m scripts.train_sft --config config.example.json
```

After training, launch the FastAPI server:

```bash
uvicorn scripts.api:app --reload
```

## Indexing

Generate a retrieval index with:

```bash
python -m scripts.build_index --input data/raw/clean_code_corpus.jsonl --output index/frappe.faiss
```

## Notes

- Configuration files are serialised via `config.example.json` and the helpers in `scripts/config.py`.
- The training scripts assume that `peft`, `trl`, `bitsandbytes`, and `flash-attn` are available in
your environment.
