#!/usr/bin/env bash
set -euo pipefail
python -m venv /srv/frappe-llm/venvs/ax
source /srv/frappe-llm/venvs/ax/bin/activate
pip install -U pip wheel
pip install -U -r requirements-axolotl.txt
