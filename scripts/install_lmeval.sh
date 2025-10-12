#!/usr/bin/env bash
set -euo pipefail
python -m venv /srv/venvs/lmeval
source /srv/venvs/lmeval/bin/activate
pip install -U pip wheel
pip install -U -r requirements-eval.txt
