# Frappe LLM Training & Alignment Stack

This repository contains the data preparation, supervised fine-tuning (SFT), preference optimization
(DPO/ORPO), and evaluation utilities used to align **Qwen2.5 Coder 3B Instruct** with Frappe
Framework tasks. The toolchain targets commodity 6 GB GPUs (for example, the RTX 1660 SUPER) using
4-bit QLoRA adapters and reproducible scripts.

## Repository Layout

- `configs/` – Axolotl configurations for SFT, DPO, and optional ORPO experiments.
- `data/` – Instruction corpora and preference datasets (see `data/dpo/README.md`).
- `models/` – Base checkpoints, SFT merges, and DPO adapters (ignored by git).
- `scripts/` – Python utilities and runnable shell scripts for environment setup and training.
- `tests/` – Pytest suite, including dataset and configuration validation.

## Quick Start

```bash
# Clone the repository to /srv/frappe-llm for path consistency
cd /srv
git clone https://github.com/frappe/frappe-llm.git
cd frappe-llm

# Create the Axolotl virtualenv (CUDA 12+, PyTorch wheels expected in the index)
make venv.ax

# Optional: create the evaluation and RAG virtualenvs
make venv.eval
make venv.rag

# Run the lightweight validation suite
pytest
```

The SFT merge referenced by all configs is expected at:
`/srv/frappe-llm/models/q25c3b-frappe-sft/merged`. If you do not already have the merged checkpoint,
follow your SFT pipeline to generate it before launching preference optimization.

## Training & Evaluation Matrix

| Stage | Config | Command | Notes |
| --- | --- | --- | --- |
| SFT (existing) | `configs/qwen25-coder-3b-sft-*.yaml` (see project history) | `python -m scripts.train_sft --config config.example.json` | Produces `/srv/frappe-llm/models/q25c3b-frappe-sft/merged` |
| DPO | `configs/qwen25-coder-3b-dpo-frappe.yaml` | `make train.dpo` | 4-bit QLoRA, batch size 1 × grad-accum 16 for 6 GB GPUs |
| ORPO (optional) | `configs/qwen25-coder-3b-orpo-frappe.yaml` | `AX_CONFIG=configs/qwen25-coder-3b-orpo-frappe.yaml axolotl train $AX_CONFIG` | Swap in if experimenting with ORPO |
| Merge (LoRA → full) | N/A | `make merge.dpo` | Writes to `/srv/frappe-llm/models/q25c3b-frappe-dpo/merged` |
| Evaluation (baseline) | N/A | `make eval.baseline` | Runs lm-eval on merged SFT checkpoint |
| Evaluation (DPO) | N/A | `make eval.dpo` | Runs lm-eval on merged DPO checkpoint |
| RAG index | `rag/index_build.py` | `make rag.index` | Builds FAISS + metadata under `/srv/frappe-llm/rag/index` |
| Serving | `serve/api.py` | `make serve` | Launches FastAPI with optional RAG context |

## DPO Preference Optimization

Supervised fine-tuning teaches the model **what** a reasonable answer looks like. Direct Preference
Optimization (DPO) teaches the model **which answer is better** when presented with competing
completions. This repository provides a production-ready workflow that layers DPO on top of the SFT
checkpoint without discarding your existing assets.

1. **Install tooling**
   ```bash
   make venv.ax
   make venv.eval  # optional, for evaluation only
   ```
2. **Validate the dataset schema**
   ```bash
   pytest tests/test_dpo_dataset_schema.py
   ```
3. **Train the DPO adapter**
   ```bash
   make train.dpo
   ```
   The script activates `/srv/frappe-llm/venvs/ax` and launches Axolotl with 4-bit QLoRA to keep VRAM
   under 6 GB. Adjust `gradient_accumulation_steps` in the config if you need different throughput.
4. **Merge the adapter**
   ```bash
   make merge.dpo
   ```
   This produces `/srv/frappe-llm/models/q25c3b-frappe-dpo/merged`, suitable for evaluation or
   deployment.
5. **Evaluate**
   ```bash
   make eval.baseline  # ARC-Easy + HellaSwag zero-shot on the SFT merge
   make eval.dpo       # same metrics on the DPO merge for apples-to-apples uplift
   ```

### Dataset Curation Tips

- Populate `data/dpo/frappe_pairs.jsonl` with real comparisons exported from your annotation tools.
- The bundled sample file demonstrates the simple prompt/choice schema. Chat-format records are also
  supported; see `data/dpo/README.md`.
- Keep counterexamples safe and realistic. Prefer highlighting missing migration steps or unsafe
  advice instead of inventing unrealistic failures.

### Deployment Notes

- AutoAWQ is **deprecated** in this stack. Prefer 4-bit loading with `bitsandbytes` during inference,
  or export GPTQ weights if you require on-disk quantization. AWQ-style exports via vLLM's
  `llm-compressor` can be evaluated later once the ecosystem stabilizes.
- Run `make rag.index` whenever your documentation or source repositories change materially. The
  manifest describes which paths were indexed.
- Launch `make serve` for local QA. Use `use_rag=true` in JSON payloads to inject retrieved context.
- Keep CUDA drivers at 12.x and install matching PyTorch wheels to ensure bitsandbytes operates in
  4-bit NF4 mode.

## Honest Evaluation

`lm-evaluation-harness` is wired for **honest**, zero-shot metrics on ARC-Easy and HellaSwag. Always
compare the SFT baseline against the DPO merge to validate uplift.

```bash
# Baseline (merged SFT)
bash scripts/eval_baseline.sh

# Preference-tuned model (merged DPO)
bash scripts/eval_dpo.sh
```

Record the resulting scores in deployment tickets and flag regressions greater than your agreed SLA.

## RAG 2.0 Pipeline

The `rag/` package provides end-to-end retrieval with dense embeddings and reranking:

1. `make venv.rag` to provision dependencies (`FlagEmbedding`, `faiss-cpu`, `transformers>=4.42`).
2. Build the index from vetted repositories:
   ```bash
   python rag/index_build.py --roots ../frappe,../erpnext,./docs --out /srv/frappe-llm/rag/index
   ```
3. Query the retriever in Python:
   ```python
   from rag.retriever import retrieve, pack_context
   passages = retrieve("How do I add a DocType field?", 8)
   context = pack_context(passages)
   ```
4. The FastAPI server (`make serve`) can prepend the packed context automatically when `use_rag=true`.

See `docs/RAG.md` for tuning guidance and governance notes.

## Minimal Serving

`serve/api.py` exposes a streaming `/query` endpoint backed by the merged DPO model. Requests are
rate-limited (5 per minute per client) and optionally inject RAG context. Configure the model path via
`FRAPPE_SERVE_MODEL` or follow the default `/srv/frappe-llm/models/q25c3b-frappe-dpo/merged`.

## Testing

All new automation is covered by pytest modules in `tests/`. The suite validates Axolotl configs,
ensures the merge script syntax is correct, guards against dataset schema drift, and now exercises the
RAG pipeline end-to-end with lightweight dummy embeddings.

Run `pytest` before pushing changes. Continuous integration executes `ruff` linting and the full test
suite (see `.github/workflows/ci.yml`).

## Full Runbook

- [Quantization guidance](docs/QUANTIZATION.md)
- [RAG operations](docs/RAG.md)
- [Operational governance checklist](docs/OPERATIONS.md)
