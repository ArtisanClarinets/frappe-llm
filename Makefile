.PHONY: venv.ax venv.eval train.dpo merge.dpo eval.baseline eval.dpo

venv.ax:
bash scripts/install_ax.sh

venv.eval:
bash scripts/install_lmeval.sh

train.dpo:
bash scripts/train_dpo.sh

merge.dpo:
bash scripts/merge_dpo.sh

eval.baseline:
source /srv/venvs/lmeval/bin/activate && \
lm_eval --model hf --model_args pretrained=/srv/frappe-llm/models/q25c3b-frappe-sft/merged --tasks gsm8k --device cuda

eval.dpo:
source /srv/venvs/lmeval/bin/activate && \
lm_eval --model hf --model_args pretrained=/srv/frappe-llm/models/q25c3b-frappe-dpo/merged --tasks gsm8k --device cuda
