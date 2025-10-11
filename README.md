# Frappe LLM – Qwen2.5 Coder 3B Training Stack

This repository contains a production-ready pipeline for preparing data, running Axolotl QLoRA training, and serving a Frappe-focused coding assistant based on **Qwen2.5-Coder-3B-Instruct**. The project is tuned for a single NVIDIA GTX 1660 SUPER (6 GB, Turing) and follows conservative defaults that work on developer workstations without administrative privileges.

## Hardware + Precision Targets

- GPU: NVIDIA Turing (SM75) with 6 GB VRAM
- Precision: `fp16` (no `bf16`)
- Attention: PyTorch SDPA (`attn_implementation: sdpa`, Flash Attention disabled)
- Finetuning method: Axolotl QLoRA (NF4, LoRA rank 64)

## Repository Layout

| Path | Purpose |
| ---- | ------- |
| `configs/` | Version-controlled Axolotl configs (mirrored to `/srv/frappe-llm/configs/`). |
| `data/`, `prefs/`, `sft/` | Legacy datasets that can be normalised via `scripts/convert_to_messages.py`. |
| `output/` | Placeholder for evaluation artifacts (real training outputs live under `/srv/frappe-llm/output/`). |
| `prepared/` | Placeholder for Axolotl preprocessed datasets. |
| `scripts/` | Command line tooling for preprocessing, training, merging, dataset conversion, indexing, and serving. |
| `tests/` | Smoke tests that guard configuration drift. |
| `train/`, `logs/`, `models/`, `hf_cache/`, `venv_*` | Staging areas retained for completeness with README stubs. |

## Environment Setup (Ubuntu 24.04 + CUDA 12.4)

```bash
# 1) System prep
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv git

# 2) Python virtual environment
python3.11 -m venv ~/.venvs/frappe-llm
source ~/.venvs/frappe-llm/bin/activate

# 3) Upgrade pip + wheel
pip install --upgrade pip wheel setuptools

# 4) Install PyTorch (CUDA 12.4 build)
pip install --extra-index-url https://download.pytorch.org/whl/cu124 "torch==2.3.1" "torchvision==0.18.1" "torchaudio==2.3.1"

# 5) Axolotl + dependencies for QLoRA on Turing
pip install axolotl==0.4.0.post1 bitsandbytes==0.43.1 peft==0.11.1 trl==0.8.6 transformers==4.41.2 accelerate==0.31.0 fastapi==0.111.0 uvicorn[standard]==0.30.1 pydantic==2.7.4 sentence-transformers==2.7.0 faiss-cpu==1.8.0
```

## Canonical Dataset Location

All tooling expects the supervised fine-tuning dataset at:

```
/srv/frappe-llm/datasets/frappe_messages.json
```

The file must contain a JSON array where each element has a `messages` list in the OpenAI chat format (role/content objects).

### Converting Existing JSONL Corpora

Legacy corpora shipped with this repository can be normalised into the canonical dataset via:

```bash
python -m scripts.convert_to_messages \
  --output /srv/frappe-llm/datasets/frappe_messages.json \
  --instruction-jsonl data/frappe_instruct.jsonl \
  --sft-jsonl sft/frappe_sft.jsonl \
  --pairs-jsonl prefs/frappe_pairs.jsonl
```

The converter validates JSON structure, deduplicates on `(user prompt, assistant reply)`, and raises actionable errors when encountering malformed rows.

## Training Workflow (Axolotl QLoRA)

1. Copy the versioned config into place (already mirrored during repo checkout):
   - `/srv/frappe-llm/configs/qwen25-coder-3b-qlora-fp16-turing.yaml`
2. Inspect and, if necessary, edit dataset/output paths to suit your environment.
3. Run the pipeline:

```bash
# Preprocess -> populates /srv/frappe-llm/prepared
python -m scripts.train_sft preprocess --settings config.example.json

# Train -> writes checkpoints into /srv/frappe-llm/output/qwen25-coder-3b-frappe-qlora
python -m scripts.train_sft train --settings config.example.json

# Optional: merge LoRA adapters into full weights at .../merged
python -m scripts.train_sft merge --settings config.example.json
```

The commands verify that required files exist, log the exact Axolotl invocation, and surface stdout/stderr verbatim if any step fails.

## Serving with vLLM on GTX 1660 SUPER

After training you can serve either the adapter or merged weights:

```bash
pip install -U vllm

# Serve base + LoRA adapter
vllm serve Qwen/Qwen2.5-Coder-3B-Instruct \
  --enable-lora \
  --lora-modules '{"name":"frappe","path":"/srv/frappe-llm/output/qwen25-coder-3b-frappe-qlora"}' \
  --dtype float16 --max-model-len 4096

# OR serve merged weights
vllm serve /srv/frappe-llm/output/qwen25-coder-3b-frappe-qlora/merged --dtype float16
```

For local experimentation you may also load the LoRA adapter inside `scripts/api.py`, which defaults to binding on `127.0.0.1` and includes guidance on rate limiting.

## Testing & Developer Checks

Use the bundled smoke tests before pushing changes:

```bash
./devcheck.sh
```

This script compiles Python modules and runs `pytest`, covering configuration parsing plus dry-run validations for the preprocess/train/merge wrappers.

## Licensing Guidance

- Frappe Framework source data (MIT) is suitable for redistribution.
- ERPNext or other GPL-3 assets should remain internal unless cleared by legal review.

## Support

Open an issue if you encounter environment-specific problems or need additional automation around dataset preparation, Axolotl parameters, or deployment.
