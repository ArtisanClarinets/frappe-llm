#!/usr/bin/env bash
set -euo pipefail
cd /srv/frappe-llm
source /srv/frappe-llm/venvs/ax/bin/activate
axolotl merge-lora /srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml \
  --lora-model-dir /srv/frappe-llm/models/q25c3b-frappe-dpo
