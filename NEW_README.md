# frappe-llm

**Enterprise-grade, GPU-efficient fine-tuning for Frappe / ERPNext aware code models (QLoRA + Axolotl), with SFT + DPO workflows, evaluation, and reproducible ops.**

Designed for small-VRAM cards (e.g., RTX 2060 6 GB / GTX 1660 SUPER) on Windows (WSL2) or Ubuntu servers. This repo standardizes **dataset format**, **Axolotl configs**, **training commands**, **LoRA merge/export**, and **evaluation**—so you can go from raw JSONL to a deployable adapter in a few copy-paste steps.

---

## TL;DR Quick Start

```bash
# 0) (WSL2/Ubuntu) System prep
sudo apt update && sudo apt install -y git python3.12-venv build-essential
python3.12 -m venv ~/.venvs/ax && source ~/.venvs/ax/bin/activate
pip install --upgrade pip wheel
pip install "axolotl==0.12.2" "bitsandbytes>=0.43.3" "transformers>=4.44" "accelerate>=0.33" datasets peft deepspeed

# 1) Clone and enter
sudo mkdir -p /srv/frappe-llm && sudo chown -R $USER:$USER /srv/frappe-llm
cd /srv/frappe-llm
git clone <your-origin> .   # or copy the repo files into /srv/frappe-llm

# 2) Set caches (saves VRAM/RAM & prevents redownloads)
mkdir -p /srv/frappe-llm/hf_cache
export HF_HOME=/srv/frappe-llm/hf_cache
export TRANSFORMERS_CACHE=/srv/frappe-llm/hf_cache

# 3) Put SFT data at data/sft/*.jsonl  (format below)
#    Put DPO data at data/dpo/*.jsonl  (format below)

# 4) Preprocess (creates tokenized shards)
axolotl preprocess /srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml

# 5) Train SFT (QLoRA, 4-bit, sdpa)
axolotl train /srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml

# 6) (Optional) DPO preference tuning
axolotl train /srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml

# 7) Merge LoRA to a full HF model, or export adapter
axolotl merge-lora /srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml

# 8) Evaluate
python scripts/eval_lm_harness.py --model_dir /srv/frappe-llm/models/finetune/q25c3b-frappe-sft-merged
```

---

## Why this repo

* **Small-VRAM friendly:** QLoRA + 4-bit NF4 + gradient checkpointing, tuned for sm_75 (Turing, e.g., RTX 2060/1660).
* **Enterprise hygiene:** deterministic runs, pinned configs, dataset specs, and audit trail of artifacts.
* **Frappe/ERPNext focus:** SFT and DPO formats tailored for framework code, DocTypes, controllers, server scripts, and ops workflows.

---

## Repository Layout

```
/srv/frappe-llm
├─ configs/                         # Axolotl YAMLs (SFT, DPO, merge)
│  ├─ qwen25-coder-3b-qlora-fp16-turing.yaml
│  ├─ qwen25-coder-3b-sft-frappe-instr.yaml
│  ├─ qwen25-coder-3b-sft-frappe-chat.yaml
│  ├─ qwen25-coder-3b-sft-code.yaml
│  ├─ qwen25-coder-3b-dpo.yaml
│  └─ qwen25-coder-3b-dpo-frappe.yaml
├─ data/
│  ├─ sft/                          # Your SFT JSONLs (instruction/input/output)
│  └─ dpo/                          # Your DPO JSONLs (prompt/chosen/rejected)
├─ datasets/                        # Optional: raw sources before curation
│  ├─ sft/
│  └─ dpo/
├─ hf_cache/                        # Hugging Face cache (HF_HOME, TRANSFORMERS_CACHE)
├─ models/
│  └─ finetune/                     # Output adapters/merged weights
├─ outputs/                         # Logs, tensorboard, checkpoints
└─ scripts/
   ├─ eval_lm_harness.py            # Harness wrapper (example below)
   ├─ preprocess.sh                 # (optional) command wrappers
   ├─ train_sft.sh
   ├─ train_dpo.sh
   ├─ merge_lora.sh
   └─ export_adapter.sh
```

> **Note:** If any of the above helper scripts aren’t in your repo yet, copy the inline examples from this README into `scripts/` verbatim.

---

## Hardware Profiles

* **Primary target:** RTX 2060 6 GB (sm_75), 128 GB system RAM, Ryzen 7 3700—**supported** with batch size 1 + grad accumulation.
* **Also works on:** GTX 1660 SUPER 6 GB (sm_75), similar knobs.
* **Larger GPUs:** Increase `micro_batch_size`, decrease `grad_accumulation_steps` for speed.

---

## Supported Base Models (tested)

* `Qwen/Qwen2.5-Coder-3B-Instruct` (recommended here; modern tokenizer, strong coding priors)
* You can adapt configs to other 3B–7B coder models; keep the **4-bit QLoRA** pattern.

---

## Dataset Standards

### Supervised Fine-Tuning (SFT) JSONL

Each line = one object. Fields:

* `instruction` *(string, required)*
* `input` *(string, may be empty)*
* `output` *(string, required: the ideal answer/completion)*

**Example:**

```json
{"instruction":"Write a Frappe v15 query builder example that filters active Customers by territory.",
 "input":"",
 "output":"from frappe.query_builder import DocType\nCustomer = DocType('Customer')\nq = (\n    frappe.qb.from_(Customer)\n    .select(Customer.name, Customer.territory)\n    .where((Customer.disabled == 0) & (Customer.territory == 'North America'))\n)\nrows = q.run()\n"}
```

**Quality rules (Fortune-500):**

* Grounded, runnable, non-trivial, no placeholders.
* Prefer **framework-correct** Frappe/ERPNext patterns (permissions, qb, valid hooks).
* Cover breadth: DocTypes, controllers, server scripts, REST, background jobs, permissions, tests, migrations, reports, print formats, desk pages.

### Direct Preference Optimization (DPO) JSONL

Each line = one object. Fields:

* `prompt` *(string, the user/task message)*
* `chosen` *(string, the preferred response)*
* `rejected` *(string, the inferior response)*

**Example:**

```json
{"prompt":"Create a migration-safe patch to add an index on Sales Invoice posting_date in Frappe v15.",
 "chosen":"import frappe\n\ndef execute():\n    if not frappe.db.has_index('Sales Invoice', 'posting_date'): \n        frappe.db.add_index('Sales Invoice', ['posting_date'])\n",
 "rejected":"# create index blindly\nfrappe.db.sql(\"CREATE INDEX idx_posting_date ON `tabSales Invoice` (posting_date)\")\n"}
```

---

## Axolotl Configs (Key Patterns)

This repo includes ready-to-run configs. Typical knobs for Turing (sm_75):

```yaml
# Example: configs/qwen25-coder-3b-sft-frappe-instr.yaml
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
trust_remote_code: true
chat_template: qwen_25

# GPU/precision
attn_implementation: sdpa
flash_attention: false
fp16: true
bf16: false

# QLoRA w/ 4-bit
load_in_4bit: true
adapter: qlora
lora_r: 64
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Data
datasets:
  - path: /srv/frappe-llm/data/sft/frappe_sft.jsonl
    type: completion
    field_instruction: instruction
    field_input: input
    field_output: output

val_set_size: 1000
output_dir: /srv/frappe-llm/models/finetune/q25c3b-frappe-sft
```

> Use the provided DPO configs for preference tuning; they expect `data/dpo/*.jsonl` pairs.

---

## Installation & Environment (Windows + WSL2 or Ubuntu)

1. **Windows NVIDIA driver** (latest Studio/Game Ready) → **Enable WSL GPU**.
2. **WSL2 Ubuntu 22.04** recommended. Open Ubuntu terminal.
3. **CUDA in WSL2:** recent Windows drivers expose CUDA automatically to WSL. Verify:

   ```bash
   nvidia-smi
   ```
4. **Python venv & deps** (see TL;DR). Use **bitsandbytes ≥0.43** for better sm_75 stability.
5. **Hugging Face auth** (if needed for gated models):

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

### Accelerate config (optional but recommended)

```bash
accelerate config default
# Accept defaults; single GPU, FP16. Axolotl will still honor its own config.
```

---

## Reproducible Training Workflows

### 1) Preprocess (tokenize & shard)

```bash
source ~/.venvs/ax/bin/activate
export HF_HOME=/srv/frappe-llm/hf_cache
export TRANSFORMERS_CACHE=/srv/frappe-llm/hf_cache

axolotl preprocess /srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml
```

### 2) SFT (QLoRA)

```bash
axolotl train /srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml
```

**Tips for 6 GB VRAM:**

* Keep `micro_batch_size: 1` in your config (or CLI).
* Use gradient checkpointing (Axolotl enables by default for QLoRA).
* Increase `grad_accumulation_steps` (e.g., 16–64) if OOM.
* Use `attn_implementation: sdpa`, `flash_attention: false`.

### 3) DPO (after SFT)

Ensure `data/dpo/*.jsonl` exist, then:

```bash
axolotl train /srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml
```

### 4) Merge LoRA → full model (optional) or export adapter

```bash
# Merge LoRA weights into a single HF folder for inference without PEFT
axolotl merge-lora /srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml

# Or keep PEFT adapter to load over base in 4-bit for low VRAM inference
```

**Output paths** land under `/srv/frappe-llm/models/finetune/…`.

---

## Evaluation (LM-Evaluation-Harness wrapper)

A small convenience wrapper is provided at `scripts/eval_lm_harness.py`. Example:

```bash
python scripts/eval_lm_harness.py \
  --model_dir /srv/frappe-llm/models/finetune/q25c3b-frappe-sft-merged \
  --tasks humaneval mbpp \
  --batch_size 1
```

**What to watch:**

* Track scores **before** and **after** DPO to verify preference gains.
* Keep a simple CSV log (date, commit, dataset hash, config hash, scores).

---

## Inference / Serving

**Low-VRAM (6 GB) local testing:**

* Load base in 4-bit + apply PEFT adapter for quick checks.
* Or run the merged model with low `max_seq_len` (e.g., 1024–2048), batch size 1.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = "/srv/frappe-llm/models/finetune/q25c3b-frappe-sft-merged"
tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain how to create an idempotent migration in Frappe v15 to add an index."
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Data Governance & Quality

* **Deduplicate** near-identical samples.
* **Redact secrets** (API keys, tokens).
* **Prefer runnable code** and accurate Frappe idioms (qb, permissions, hooks).
* **Balance topics**: framework core, ERPNext doctypes, tests, security, performance.
* **DPO** pairs: ensure clear, instructive *why chosen is better* (safety, correctness, style).

---

## Troubleshooting

| Symptom                       | Likely Cause                      | Fix                                                                                                                                |
| ----------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| CUDA OOM                      | Context too long; batch too big   | Lower `max_seq_len`, set `micro_batch_size: 1`, raise `grad_accumulation_steps`, enable gradient checkpointing (default in QLoRA). |
| BitsAndBytes error on Windows | Native Windows CUDA wheels fickle | Use **WSL2 Ubuntu**; keep `bitsandbytes>=0.43`.                                                                                    |
| Flash-Attn import errors      | Turing not ideal                  | Keep `flash_attention: false`, use `sdpa`.                                                                                         |
| Slow downloads                | No HF cache                       | Set `HF_HOME` and `TRANSFORMERS_CACHE`.                                                                                            |
| Poor SFT val loss             | Mixed quality data                | Tighten curation, add more diverse, *correct* samples, especially tests and migrations.                                            |
| DPO not improving             | Weak or noisy pairs               | Strengthen preference signal; ensure *clear inferiority* of rejected; widen coverage.                                              |

---

## Security & Compliance

* **No production secrets** in training data, configs, or logs.
* Review outputs for **hallucinated unsafe SQL** or **permission bypasses**; prefer `frappe.qb` with parameters.
* Keep an **artifact log** (dataset version → model version → evaluation → deployment date).

---

## Maintenance Playbook

* **Version pinning:** Track Axolotl / Transformers / PEFT versions in `configs/README.versions.md`.
* **Artifacts:** Store checkpoints and merged weights under `/srv/frappe-llm/models/finetune/…` with dates.
* **Changelogs:** Each training run adds an entry in `outputs/run_log.csv` with git commit, dataset hash, config path, metrics.

---

## Example Helper Scripts

> Copy these into `scripts/` if you don’t already have them.

**scripts/eval_lm_harness.py**

```python
#!/usr/bin/env python3
import argparse, subprocess, sys, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=["humaneval","mbpp"])
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    # Lazy install if missing
    try:
        import lm_eval  # noqa
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lm-eval==0.4.2"])

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.model_dir},dtype=float16",
        "--tasks", ",".join(args.tasks),
        "--batch_size", str(args.batch_size),
        "--write_out", "--log_samples"
    ]
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
```

**scripts/train_sft.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-/srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml}
source ~/.venvs/ax/bin/activate
export HF_HOME=/srv/frappe-llm/hf_cache
export TRANSFORMERS_CACHE=/srv/frappe-llm/hf_cache
axolotl preprocess "$CFG"
axolotl train "$CFG"
```

**scripts/train_dpo.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-/srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml}
source ~/.venvs/ax/bin/activate
export HF_HOME=/srv/frappe-llm/hf_cache
export TRANSFORMERS_CACHE=/srv/frappe-llm/hf_cache
axolotl train "$CFG"
```

**scripts/merge_lora.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-/srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml}
source ~/.venvs/ax/bin/activate
axolotl merge-lora "$CFG"
```

---

## Roadmap

* [ ] Add pre-commit hooks (black, ruff, yaml-lint) for dataset & config hygiene
* [ ] Optional vLLM/Ollama serving recipe for merged models
* [ ] CI pipeline to run a smoke eval on PRs (small task set)
* [ ] Dataset profiler (length histograms, dup finder, leakage checks)

---

## License

Specify your project’s license here (e.g., Apache-2.0 or MIT). Do **not** include datasets you don’t own or that contain sensitive data.

---

## Acknowledgments

Built on the shoulders of **Axolotl**, **Transformers**, **PEFT**, **bitsandbytes**, and the open-source LLM community.

---

### Appendix A — Sample Datasets

**SFT minimal file (`data/sft/frappe_sft.jsonl`):**

```json
{"instruction":"Add a before_insert hook to set naming_series on a DocType.","input":"DocType: Repair Order","output":"# hooks.py\nfixtures = []\n\ndoc_events = {\n  \"Repair Order\": {\n    \"before_insert\": \"repair_portal.repair.doctype.repair_order.repair_order.set_series\"\n  }\n}\n\n# repair_order.py\nimport frappe\n\ndef set_series(doc, method=None):\n    if not doc.naming_series:\n        doc.naming_series = \"RO-.YYYY.-.#####\"\n"}
```

**DPO minimal file (`data/dpo/frappe_dpo_pairs.jsonl`):**

```json
{"prompt":"Safely fetch active Items by item_group using Frappe Query Builder.",
 "chosen":"from frappe.query_builder import DocType\nItem = DocType('Item')\nq = (frappe.qb.from_(Item)\n     .select(Item.name)\n     .where((Item.disabled == 0) & (Item.item_group == 'Clarinet Parts')))\nrows = q.run()\n",
 "rejected":"# unsafe raw SQL\nrows = frappe.db.sql(\"SELECT name FROM `tabItem` WHERE item_group='Clarinet Parts' AND disabled=0\")\n"}
```

---

### Appendix B — Known-Good Axolotl Knobs for 6 GB VRAM

In your YAML or CLI:

* `load_in_4bit: true`
* `attn_implementation: sdpa`
* `flash_attention: false`
* `fp16: true`, `bf16: false`
* `gradient_checkpointing: true` (Axolotl enables automatically for QLoRA)
* `micro_batch_size: 1`
* `grad_accumulation_steps: 16` (raise to 32/64 if needed)
* `max_seq_len: 2048` (drop to 1536/1024 if OOM)

