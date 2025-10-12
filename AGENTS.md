**Purpose**
Turn this repository into a repeatable, enterprise-grade LLM training workspace for Frappe/ERPNext domain expertise. This playbook defines roles, guardrails, dataset specs, training/eval pipelines (SFT → DPO), and hardening checklists so any engineer (or AI agent) can execute safely and consistently on a single-GPU workstation.

**Audience**

* Dylan (owner) and trusted collaborators
* AI assistants (ChatGPT/Copilot/etc.) acting via the prompts in this file

**Golden Outcome**

* Deterministic runs, clean artifacts, auditable data lineage, reproducible results, and deployable fine-tunes for Frappe-aware coding/help.

---

## 0) Repo Facts & Canonical Paths

* **Root**: `/srv/frappe-llm`
* **Key dirs (expected)**:

  ```
  /srv/frappe-llm
  ├── configs/                      # Axolotl YAMLs (SFT/DPO/QLoRA)
  ├── data/                         # Human-curated raw data (never overwritten)
  │   ├── sft/                      # SFT authoring area (jsonl, manifests)
  │   └── dpo/                      # DPO authoring area (jsonl, manifests)
  ├── datasets/                     # Materialized/cleaned copies for training
  │   ├── sft/                      # (auto/copy) curated datasets (frozen for runs)
  │   └── dpo/
  ├── hf_cache/                     # HF cache (models, datasets). Use HF_HOME.
  ├── models/                       # Run outputs, checkpoints, merged weights
  │   ├── sft/
  │   ├── dpo/
  │   └── merged/
  ├── logs/                         # Axolotl, eval, and custom logs
  └── scripts/                      # Optional helpers (validation, merges, packaging)
  ```

**Existing configs referenced in this doc** (adjust if you rename):

* `/srv/frappe-llm/configs/qwen25-coder-3b-qlora-fp16-turing.yaml`
* `/srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml`
* `/srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-chat.yaml`
* `/srv/frappe-llm/configs/qwen25-coder-3b-sft-code.yaml`
* `/srv/frappe-llm/configs/qwen25-coder-3b-dpo.yaml`
* `/srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml`

---

## 1) Role Matrix (Humans or AI Agents)

Each role includes a **one-shot prompt** you can paste into ChatGPT to “spin up” that agent.

### 1.1 Program Lead (PL)

**Goal:** Decide scope, approve datasets, track runs, publish releases.
**Prompt (one-shot):**

> You are Program Lead for the frappe-llm project. Read AGENTS.md and produce a 1-page run plan for the next SFT→DPO cycle, including: dataset sources, target sizes, acceptance checks, run IDs, artifact names, evaluation suites, and release notes outline. Output a concise plan with concrete paths under /srv/frappe-llm.

### 1.2 Data Steward (DS)

**Goal:** Curate, validate, de-duplicate, and lint SFT/DPO data; enforce manifests & PII policy.
**Prompt (one-shot):**

> You are the Data Steward for frappe-llm. Using the dataset specs in AGENTS.md, generate: (1) validators for SFT and DPO JSONL, (2) a dedupe script keyed on normalized prompts, and (3) a MANIFEST.md template. Assume paths under /srv/frappe-llm/data and /srv/frappe-llm/datasets. Print full files with absolute paths.

### 1.3 SFT Engineer (SFT)

**Goal:** Prepare SFT data, run Axolotl SFT, monitor OOM, checkpoint, and archive.
**Prompt (one-shot):**

> You are the SFT Engineer. Read AGENTS.md and output exact Axolotl commands for QLoRA SFT on Qwen2.5-Coder-3B-Instruct using the provided config files. Include preflight checks, OOM fallbacks, and artifact naming under /srv/frappe-llm/models/sft/<run_id>. Provide commands only.

### 1.4 DPO Architect (DPO)

**Goal:** Build preference pairs, run DPO on top of SFT, checkpoint and archive.
**Prompt (one-shot):**

> You are the DPO Architect. Based on AGENTS.md, output a complete DPO run plan: data schema checks, Axolotl preprocess/train commands using qwen25-coder-3b-dpo-frappe.yaml, and merge/export steps. Include paths and run IDs.

### 1.5 Evaluations Lead (EVAL)

**Goal:** Run standard evals and bespoke Frappe tasks; produce a scorecard and deltas vs. base.
**Prompt (one-shot):**

> You are the Evaluations Lead. Generate commands to evaluate SFT and DPO models on coding and instruction-following benchmarks plus a custom Frappe sanity set. Include environment setup, seed, and CSV/JSON outputs to /srv/frappe-llm/logs/eval/<run_id>.

### 1.6 Release Manager (RM)

**Goal:** Merge LoRA, quantize (optional), write Model Card, license audit, publish artifacts.
**Prompt (one-shot):**

> You are the Release Manager. Produce a step-by-step release checklist for the latest DPO run: merge LoRA, optional quantization, artifact tree under /srv/frappe-llm/models/merged/<tag>, MODEL_CARD.md draft, LICENSE_SUMMARY.md, and SHA256 checks.

### 1.7 Security & Compliance (SEC)

**Goal:** PII/secret scan, license provenance, usage policy, retention windows.
**Prompt (one-shot):**

> You are Security & Compliance. Using AGENTS.md, output a short policy and a script plan to scan /srv/frappe-llm/data and /srv/frappe-llm/datasets for PII and secrets; and produce a LICENSE_SUMMARY.md template that captures dataset sources and terms.

---

## 2) Guardrails & Global Standards

* **Reproducibility:** Fix `seed=42` (or run_id-specific) everywhere. Save `configs/*.yaml` snapshots with each run.
* **Caching:** Prefer `HF_HOME=/srv/frappe-llm/hf_cache` instead of `TRANSFORMERS_CACHE`.
* **Safety:** No raw secrets anywhere in datasets. No client data unless fully anonymized and licensed.
* **Data immutability:** Treat `/data/` as human workspace; copy finalized datasets to `/datasets/` with a versioned folder (`vYYYYMMDD-<slug>`).
* **Artifacts:** Every run gets a unique `RUN_ID=YYYYMMDD-hhmm-<slug>` stored inside `models/*/<RUN_ID>` and `logs/*/<RUN_ID>`.
* **OOM hygiene (6 GB VRAM):** Use QLoRA 4-bit, batch_size=1, gradient_accumulation to hit effective batch; enable gradient checkpointing; prefer `attn_implementation: sdpa`.
* **Provenance:** Keep a `MANIFEST.md` with exact sources, filters, counts, and acceptance checks.
* **Licensing:** Only incorporate data you are allowed to use for training; log licenses in `LICENSE_SUMMARY.md`.

---

## 3) Environment Setup (single-GPU, RTX 2060 6 GB)

**One-time (Linux or WSL):**

```bash
# System prep (CUDA drivers already installed)
mkdir -p /srv/frappe-llm/{data,datasets,models,logs,hf_cache,scripts}
python3.12 -m venv /srv/frappe-llm/.venv
source /srv/frappe-llm/.venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install "axolotl==0.12.2" "bitsandbytes" "peft" "transformers" "accelerate" "datasets" "jsonlines" "numpy" "pydantic" "tqdm" "rich" "huggingface_hub" "sentencepiece"

# Optional eval stack
pip install "lm-eval" "evaluate"

# Use HF_HOME, not TRANSFORMERS_CACHE
echo 'export HF_HOME=/srv/frappe-llm/hf_cache' >> /srv/frappe-llm/.env
echo 'export HF_DATASETS_CACHE=/srv/frappe-llm/hf_cache/datasets' >> /srv/frappe-llm/.env
echo 'export HUGGINGFACE_HUB_CACHE=/srv/frappe-llm/hf_cache/hub' >> /srv/frappe-llm/.env
echo 'export TOKENIZERS_PARALLELISM=false' >> /srv/frappe-llm/.env
echo 'export BNB_CUDA_VERSION=120' >> /srv/frappe-llm/.env  # adjust if needed
echo 'export PYTHONUTF8=1' >> /srv/frappe-llm/.env

# Load env for current shell
set -a; source /srv/frappe-llm/.env; set +a
```

**GPU stuck? Kill blockers and reset:**

```bash
nvidia-smi
sudo fuser -v /dev/nvidia*   # identify
sudo kill -9 <PID>           # stop rogue process
```

---

## 4) Dataset Specifications

### 4.1 SFT JSONL (Chat format – preferred)

* **Path:** `/srv/frappe-llm/data/sft/frappe_sft.jsonl` (authoring)
* **Record schema (one per line):**

```json
{
  "messages": [
    {"role": "system", "content": "You are a senior Frappe v15 engineer..."},
    {"role": "user", "content": "How do I create a DocType with child table? ..."},
    {"role": "assistant", "content": "Step-by-step: 1) In doctype JSON, set is_child_table=1 ..."}
  ],
  "source": "manual_curation",
  "tags": ["frappe15","doctype","child_table","security"],
  "license": "CC-BY-SA-4.0"
}
```

* **Constraints:**

  * Prefer concise, correct, reference-style answers.
  * No code placeholders; provide runnable snippets when applicable.
  * Avoid vendor secrets, keys, or private customer data.
  * Include `system` rows when needed to set precise personas (coding vs. ops).

**Alternative SFT schema (instr-input-output):**

```json
{
  "instruction": "Create a Frappe v15 server script to validate a field...",
  "input": "",
  "output": "frappe.ui.form.on('MyDoc', {...})",
  "source": "manual_curation",
  "license": "CC-BY-SA-4.0"
}
```

### 4.2 DPO JSONL (Preference pairs)

* **Path:** `/srv/frappe-llm/data/dpo/frappe_dpo_pairs.jsonl`
* **Record schema:**

```json
{
  "prompt": "User asks: How to write a Frappe patch to backfill a field safely?",
  "chosen": "High-quality, safe, idempotent patch with try/commit pattern...",
  "rejected": "Vague answer using raw SQL string concatenation...",
  "tags": ["frappe15","migration","safety"],
  "license": "CC-BY-SA-4.0"
}
```

* **Guidelines:**

  * The **chosen** must be strictly better than **rejected** on correctness, safety, and style.
  * Cover your target domain widely (DocTypes, hooks, permissions, reports, tests, patches, Vue/React desk pages, bench ops, etc.).
  * 400–800 pairs is a good first pass; quality > quantity.

### 4.3 Data Manifests & Checks

Create `/srv/frappe-llm/data/MANIFEST.md` with:

* Sources, licenses, counts (lines, tokens), dedupe ratio, profanity/PII filters, last edited date.
* Acceptance checklist: schema validation passed; no empty assistant replies; no code with secrets; prompt leakage avoided.

**Validation (DS role) — suggested Python tasks:**

* Validate schemas and roles (`messages[0].role == "system"` optional but consistent).
* Normalize whitespace, strip non-printables.
* Deduplicate on canonicalized `(prompt|messages)` hash.
* Token length ceilings per sample (e.g., ≤ 2k tokens pre-pack).

---

## 5) Training Pipelines

> **Default model:** `Qwen/Qwen2.5-Coder-3B-Instruct` with QLoRA 4-bit on 6 GB VRAM.

### 5.1 Pre-flight (all runs)

```bash
source /srv/frappe-llm/.venv/bin/activate
set -a; source /srv/frappe-llm/.env; set +a

# Sanity
python -c "import torch; print(torch.cuda.is_available())"
python -c "import bitsandbytes as bnb; print('bnb ok')"
nvidia-smi

# Freeze dataset copy for this run
RUN_ID=$(date +"%Y%m%d-%H%M")-sft
mkdir -p /srv/frappe-llm/datasets/sft/${RUN_ID}
cp /srv/frappe-llm/data/sft/*.jsonl /srv/frappe-llm/datasets/sft/${RUN_ID}/

# Optional: keep MANIFEST snapshot
cp /srv/frappe-llm/data/MANIFEST.md /srv/frappe-llm/datasets/sft/${RUN_ID}/MANIFEST.md || true
```

### 5.2 SFT (Axolotl)

**Preprocess (packs data, builds shards):**

```bash
axolotl preprocess /srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml
```

**Train (QLoRA, fp16, SDPA attention):**

```bash
axolotl train /srv/frappe-llm/configs/qwen25-coder-3b-sft-frappe-instr.yaml \
  | tee /srv/frappe-llm/logs/sft/${RUN_ID}.log
```

**Typical OOM fallbacks (edit YAML then retry):**

* Reduce `micro_batch_size` to `1`
* Increase `gradient_accumulation_steps`
* Ensure `load_in_4bit: true`, `bnb_4bit_quant_type: nf4`, `bnb_4bit_compute_dtype: float16`
* Keep `attn_implementation: sdpa`, `flash_attention: false`
* Enable `gradient_checkpointing: true` in config if not already

**Artifacts:**

* LoRA adapter checkpoints → `/srv/frappe-llm/models/sft/<RUN_ID>/` (as configured in YAML)

### 5.3 DPO (on top of SFT)

**Freeze DPO data for run:**

```bash
RUN_ID=$(date +"%Y%m%d-%H%M")-dpo
mkdir -p /srv/frappe-llm/datasets/dpo/${RUN_ID}
cp /srv/frappe-llm/data/dpo/*.jsonl /srv/frappe-llm/datasets/dpo/${RUN_ID}/
```

**Preprocess + Train:**

```bash
axolotl preprocess /srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml

axolotl train /srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml \
  | tee /srv/frappe-llm/logs/dpo/${RUN_ID}.log
```

**Notes:**

* Ensure the DPO config points to the SFT-tuned base (or loads SFT adapter) per your YAML.
* Keep conservative batch size on 6 GB.

---

## 6) Merging, Packaging & (Optional) Quantization

### 6.1 Merge LoRA → full weights (Release Manager)

> Adjust to your Axolotl version & YAML; the command below is representative.

```bash
# Example paths—confirm your SFT or DPO adapter path first
SFT_ADAPTER="/srv/frappe-llm/models/sft/<SFT_RUN>/adapter"
BASE="Qwen/Qwen2.5-Coder-3B-Instruct"
OUT="/srv/frappe-llm/models/merged/q25c3b-frappe-sft-merged-$(date +%Y%m%d)"

axolotl merge-lora \
  --base_model "${BASE}" \
  --lora_model "${SFT_ADAPTER}" \
  --output_dir "${OUT}"
```

Repeat for the **DPO** adapter to create a final merged model tag such as:
`/srv/frappe-llm/models/merged/q25c3b-frappe-dpo-merged-YYYYMMDD`.

### 6.2 (Optional) Export & Quantize

* **Safetensors FP16** for HF: leave as merged.
* **GGUF** (CPU/Ollama): use `llama.cpp` convert scripts compatible with architecture.
* **AWQ/GPTQ**: only if you need GPU-optimized inference; ensure compatibility with Qwen stack.

**Hash & list:**

```bash
find /srv/frappe-llm/models/merged -type f -exec sha256sum {} \; \
  | tee /srv/frappe-llm/models/merged/SHA256SUMS.txt
```

---

## 7) Evaluation & Scorecards

**Goal:** Honest before/after comparisons vs. base and between SFT and DPO.

### 7.1 Setup

```bash
pip install "lm-eval"
```

### 7.2 Example commands (adjust to your backends)

```bash
# Evaluate merged SFT (HF path)
MODEL_PATH="/srv/frappe-llm/models/merged/q25c3b-frappe-sft-merged-YYYYMMDD"
OUT="/srv/frappe-llm/logs/eval/$(date +%Y%m%d-%H%M)-sft"

mkdir -p "${OUT}"

lm-eval --model hf \
  --model_args pretrained="${MODEL_PATH}",dtype=float16 \
  --tasks humaneval,mbpp \
  --device cuda:0 \
  --batch_size 1 \
  --output_path "${OUT}/results.json"

# Repeat for DPO
```

### 7.3 Custom Frappe Sanity Set

Create a small JSONL of ~100 prompts covering DocTypes, permissions, desk pages, tests, and migrations. Write a simple evaluator that checks for required keywords/anti-patterns (e.g., forbids raw SQL string concatenation; requires `frappe.qb` or parameter binding).

---

## 8) Run & Release Checklists

### 8.1 SFT/DPO Run (SFT, then DPO)

* [ ] Data MANIFEST completed, licenses captured
* [ ] SFT JSONL validated & deduped
* [ ] Axolotl preprocess OK
* [ ] Train completes without OOM (or mitigated)
* [ ] Checkpoint artifacts saved and named with RUN_ID
* [ ] DPO pairs validated (chosen > rejected)
* [ ] DPO train completes & checkpoints saved

### 8.2 Release

* [ ] Merge LoRA → full weights (SFT and/or DPO)
* [ ] Optional quantization steps documented
* [ ] Evals run; score deltas vs. base recorded
* [ ] `MODEL_CARD.md` drafted (see template below)
* [ ] `LICENSE_SUMMARY.md` finalized
* [ ] `SHA256SUMS.txt` produced
* [ ] Artifacts tree published internally

---

## 9) Troubleshooting (6 GB VRAM)

* OOM during forward/backward: lower `micro_batch_size` to 1, increase `gradient_accumulation_steps`, enable `gradient_checkpointing`, ensure 4-bit load.
* BNB kernel errors: confirm CUDA/driver match; set `export BNB_CUDA_VERSION=120` (or your installed CUDA).
* Tokenizer mismatch: clear problematic tokenizer cache in `hf_cache` for the specific model.
* Rogue GPU usage: kill lingering `ollama`/`python` PIDs before training.
* Slow downloads: pre-pull base model with `huggingface-cli download` using `HF_HOME` disk.

---

## 10) File & Template Stubs (drop into `/srv/frappe-llm`)

### 10.1 `MODEL_CARD.md` (template)

```markdown
# Model Card — <MODEL_TAG>

- **Base Model:** Qwen/Qwen2.5-Coder-3B-Instruct
- **Technique:** QLoRA SFT (and DPO if applicable)
- **Domain:** Frappe/ERPNext v15 code & operations
- **Training Data:** See LICENSE_SUMMARY.md and data/MANIFEST.md snapshots
- **Intended Use:** Developer assistance for Frappe v15; not for PII or medical/legal advice
- **Limitations:** Small model; may hallucinate; verify critical steps
- **Safety:** No secrets, no PII in training; rejects dangerous SQL patterns
- **Eval Summary:** (insert lm-eval & custom Frappe sanity results)
- **Release Artifacts:** (paths under /srv/frappe-llm/models/…)
- **Contact:** Dylan Thompson
```

### 10.2 `LICENSE_SUMMARY.md` (template)

```markdown
# License Summary

## Sources
- List each dataset file and its origin (URL or “manual_curation”)
- License per source (e.g., CC-BY-SA-4.0, MIT, custom permission)
- Notes on modifications/filters

## Usage
- Models produced from these sources are restricted to internal use unless license permits redistribution.
- Remove or quarantine any sample upon request if license concerns arise.
```

### 10.3 `data/MANIFEST.md` (template)

```markdown
# Data Manifest

Date: YYYY-MM-DD
Curator: <name>

## SFT
- Files: <list>
- Count (lines): N
- Avg tokens/sample: ~T
- Dedupe: enabled (method: normalized prompt hash)
- Filters: profanity, PII (basic), max length
- Schema: messages (chat) / instruction-output (specify)

## DPO
- Files: <list>
- Count (pairs): N
- Coverage: (list domain buckets)
- Validation: chosen > rejected by quality rubric

## Notes
- Known gaps, next batch priorities
```

---

## 11) Prompts for Code-Gen Agents (quick use)

### 11.1 “SFT Data Generator” (for curated authoring)

> Generate 25 SFT chat samples for Frappe v15 focusing on: (1) DocType security & permissions, (2) safe DB queries via frappe.qb, (3) idempotent patches, (4) unit tests, (5) desk page Vue patterns. Use the chat schema from AGENTS.md. Each answer must be correct, runnable, and free of secrets. Output a single JSONL block.

### 11.2 “DPO Pair Writer”

> Produce 15 DPO preference pairs for Frappe v15. Each includes: prompt, chosen, rejected. The chosen must be safer and more complete (permissions checks, idempotence, parameterized queries). Output JSONL only.

### 11.3 “Eval Author”

> Create a 50-item custom Frappe sanity eval set (prompts only) across migrations, doctypes, roles/permissions, desk pages, server scripts. Output JSONL with `{"prompt": "...", "tags":[...]}` records.

---

## 12) YAML Knobs (what to tweak for 6 GB)

In your existing Axolotl configs (e.g., `qwen25-coder-3b-qlora-fp16-turing.yaml`), ensure:

```yaml
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
trust_remote_code: true
chat_template: qwen_25

# Precision / attention
fp16: true
bf16: false
attn_implementation: sdpa
flash_attention: false

# Quantization + LoRA
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

bnb_config_kwargs:
  bnb_4bit_compute_dtype: float16
  bnb_4bit_quant_type: nf4

# Batch strategy (example—tune for OOM)
micro_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
```

---

## 13) Naming & Versioning Conventions

* **Runs:** `YYYYMMDD-HHMM-<stage>` → e.g., `20251012-1412-sft`
* **Merged:** `q25c3b-frappe-<stage>-merged-YYYYMMDD`
* **Logs:** `/srv/frappe-llm/logs/<stage>/<RUN_ID>.log`
* **Eval:** `/srv/frappe-llm/logs/eval/<RUN_ID>/results.json`

---

## 14) Security & Compliance Quick Policy

* Only include data you own or that is licensed for training.
* Strip PII/credentials; run a simple regex pass for keys/tokens, emails, phone numbers.
* No redistribution of third-party content without permission.
* Maintain takedown capability: every sample traceable to source.

---

## 15) Quick TL;DR (sticky)

1. Curate/validate SFT JSONL → copy to `/datasets/sft/<RUN_ID>/`
2. `axolotl preprocess` → `axolotl train` (SFT)
3. Curate/validate DPO pairs → copy to `/datasets/dpo/<RUN_ID>/`
4. `axolotl preprocess` → `axolotl train` (DPO)
5. Merge LoRA → full weights; (optional) quantize
6. Run evals; write MODEL_CARD + LICENSE_SUMMARY; hash artifacts
