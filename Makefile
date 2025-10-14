SHELL := /bin/bash

.PHONY: venv.ax venv.eval venv.rag train.dpo merge.dpo eval.baseline eval.dpo rag.index serve

venv.ax:
	bash scripts/install_ax.sh

venv.eval:
	bash scripts/install_lmeval.sh

venv.rag:
	python -m venv /srv/venvs/rag && source /srv/venvs/rag/bin/activate && pip install -U pip wheel && pip install -U -r requirements-rag.txt

train.dpo:
	source /srv/frappe-llm/venvs/ax/bin/activate && axolotl train /srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml

merge.dpo:
	source /srv/frappe-llm/venvs/ax/bin/activate && axolotl merge-lora /srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml --lora-model-dir /srv/frappe-llm/models/q25c3b-frappe-dpo

eval.baseline:
	source /srv/venvs/lmeval/bin/activate && bash /srv/frappe-llm/scripts/eval_baseline.sh

eval.dpo:
	source /srv/venvs/lmeval/bin/activate && bash /srv/frappe-llm/scripts/eval_dpo.sh

rag.index:
	python rag/index_build.py --roots ../frappe,../erpnext,./docs --out /srv/frappe-llm/rag/index

serve:
	bash /srv/frappe-llm/scripts/serve.sh
