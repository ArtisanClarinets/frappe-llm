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

# Optional: create the evaluation virtualenv
make venv.eval

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
   make eval.baseline  # evaluate the original merged SFT checkpoint
   make eval.dpo       # evaluate the merged DPO checkpoint
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
- To serve merged checkpoints, reuse the existing serving utilities in `scripts/api.py`, pointing the
  loader to `/srv/frappe-llm/models/q25c3b-frappe-dpo/merged`.
- Keep CUDA drivers at 12.x and install matching PyTorch wheels to ensure bitsandbytes operates in
  4-bit NF4 mode.

## Testing

All new automation is covered by pytest modules in `tests/`. The suite validates Axolotl configs,
ensures the merge script syntax is correct, and guards against dataset schema drift.

Run `pytest` before pushing changes. Continuous integration should execute the same command.
