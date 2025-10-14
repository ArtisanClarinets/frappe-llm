#!/usr/bin/env bash
set -euo pipefail
source /srv/venvs/lmeval/bin/activate
lm_eval --model hf \
  --model_args pretrained=/srv/frappe-llm/models/q25c3b-frappe-sft/merged,dtype=float16 \
  --tasks arc_easy,hellaswag --num_fewshot 0
