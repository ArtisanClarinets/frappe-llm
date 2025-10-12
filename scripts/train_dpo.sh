#!/usr/bin/env bash
set -euo pipefail
cd /srv/frappe-llm
source /srv/frappe-llm/venvs/ax/bin/activate
axolotl train /srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml
