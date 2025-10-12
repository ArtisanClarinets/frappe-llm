"""Supervised fine-tuning pipeline for the Frappe LLM project.

The script is intentionally modular so it can be orchestrated by
workflows (GitHub Actions, Dagster, Airflow) while remaining runnable from
the command line.  It focuses on SFT for Qwen2.5 Coder 3B Instruct with
PEFT/LoRA and training ergonomics that are practical in 2025 (Flash
Attention 2, paged optimisers, and gradient checkpointing).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence, TYPE_CHECKING

try:  # pragma: no cover - optional dependency for CLI usage
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from datasets import Dataset, DatasetDict
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig as PeftLoraConfig  # type: ignore
    from peft import TaskType  # type: ignore
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore
    from trl import SFTTrainer

try:  # pragma: no cover - optional dependencies for the CLI
    from datasets import Dataset, DatasetDict
except ModuleNotFoundError:  # pragma: no cover
    Dataset = DatasetDict = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependencies for the CLI
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
except ModuleNotFoundError:  # pragma: no cover
    AutoModelForCausalLM = AutoTokenizer = TrainingArguments = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependencies for the CLI
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model, prepare_model_for_kbit_training
except ModuleNotFoundError:  # pragma: no cover
    PeftLoraConfig = TaskType = get_peft_model = prepare_model_for_kbit_training = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependencies for the CLI
    from trl import SFTTrainer
except ModuleNotFoundError:  # pragma: no cover
    SFTTrainer = None  # type: ignore[assignment]

from .config import ExperimentConfig, TrainingArgumentsConfig, load_config

AXOLOTL_BIN = os.environ.get("AXOLOTL_BIN", "axolotl")
VALID_SUBCOMMANDS = {"preprocess", "train", "merge"}


def build_axolotl_command(subcommand: str, config_path: Path) -> List[str]:
    """Construct the Axolotl CLI command for the requested subcommand."""

    if subcommand not in VALID_SUBCOMMANDS:
        raise ValueError(f"Unknown subcommand: {subcommand}")

    resolved = Path(config_path).resolve()
    cli_subcommand = "merge-lora" if subcommand == "merge" else subcommand
    return [AXOLOTL_BIN, cli_subcommand, str(resolved)]


def execute_subcommand(subcommand: str, config_path: Path) -> None:
    """Run the Axolotl CLI with the desired action for supervised fine-tuning."""

    command = build_axolotl_command(subcommand, config_path)
    subprocess.run(command, check=True)

LOGGER = logging.getLogger(__name__)


def read_jsonl(path: Path, max_samples: int | None = None) -> List[dict]:
    """Read a JSONL file into a list of dictionaries."""

    samples: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def convert_chat_to_text(example: dict) -> dict:
    """Normalise chat messages and extract prompt/response summaries."""

    messages: Sequence[dict] = example.get("messages", [])
    if not messages:
        raise ValueError("Each training example must include a 'messages' list.")

    normalised: List[dict] = []
    prompt_parts: List[str] = []
    response = ""
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in {"system", "user", "assistant"}:
            continue
        normalised.append({"role": role, "content": content})
        if role in {"system", "user"}:
            prompt_parts.append(content)
        elif role == "assistant":
            response = content

    prompt = "\n\n".join(part for part in prompt_parts if part)
    return {"messages": normalised, "prompt": prompt, "response": response}


def make_dataset(samples: Iterable[dict]) -> Dataset:
    """Create a :class:`datasets.Dataset` from raw message dicts."""

    if Dataset is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError("The `datasets` package is required. Install it with `pip install datasets`.")

    converted = [convert_chat_to_text(sample) for sample in samples]
    return Dataset.from_list(converted)


def build_dataset_dict(cfg: ExperimentConfig) -> DatasetDict:
    """Assemble the train/eval datasets from configured JSONL sources."""

    if DatasetDict is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError("The `datasets` package is required. Install it with `pip install datasets`.")

    train_samples = read_jsonl(cfg.dataset.sft_path, cfg.dataset.max_samples)
    dataset_dict = {"train": make_dataset(train_samples)}

    if not cfg.evaluation.enable_eval:
        return DatasetDict(dataset_dict)

    if cfg.dataset.eval_path and cfg.dataset.eval_path.exists():
        eval_samples = read_jsonl(cfg.dataset.eval_path)
        dataset_dict["eval"] = make_dataset(eval_samples)
    else:
        split = dataset_dict["train"].train_test_split(test_size=0.05, seed=cfg.training.seed)
        dataset_dict["train"] = split["train"]
        dataset_dict["eval"] = split["test"]

    return DatasetDict(dataset_dict)


def ensure_tokenizer(tokenizer_path: str, cache_dir: Path | None, trust_remote_code: bool) -> AutoTokenizer:
    if AutoTokenizer is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError(
            "transformers is required to load the tokenizer. Install it with `pip install transformers`."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=trust_remote_code,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|extra_pad|>"})
    tokenizer.model_max_length = 4096
    return tokenizer


def configure_model(model_name: str, cfg: ExperimentConfig) -> AutoModelForCausalLM:
    """Load the base model and prepare it for LoRA training."""

    if AutoModelForCausalLM is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError(
            "transformers is required to load the model. Install it with `pip install transformers`."
        )
    if torch is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError("PyTorch is required to configure the training model. Install torch.")
    if None in (PeftLoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training):
        raise RuntimeError(
            "peft is required for LoRA fine-tuning. Install it with `pip install peft`."
        )

    torch_dtype = torch.bfloat16 if cfg.training.bf16 else torch.float16

    load_kwargs = dict(
        cache_dir=str(cfg.model.cache_dir) if cfg.model.cache_dir else None,
        trust_remote_code=cfg.model.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    if cfg.model.use_flash_attention:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    if cfg.model.load_in_4bit:
        load_kwargs.update(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if tokenizer_needs_resize := model.get_input_embeddings().weight.shape[0]:
        LOGGER.info("Model vocabulary size: %s", tokenizer_needs_resize)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.training.gradient_checkpointing)

    lora_cfg = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.lora.target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def create_training_arguments(cfg: TrainingArgumentsConfig, enable_eval: bool) -> TrainingArguments:
    if TrainingArguments is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError(
            "transformers is required to create training arguments. Install it with `pip install transformers`."
        )

    common_kwargs = dict(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        seed=cfg.seed,
        report_to=cfg.report_to,
        torch_compile=cfg.torch_compile,
        optim=cfg.optim,
    )

    if enable_eval:
        eval_kwargs = dict(evaluation_strategy="steps", eval_steps=cfg.eval_steps)
    else:
        eval_kwargs = dict(evaluation_strategy="no")

    return TrainingArguments(**common_kwargs, **eval_kwargs)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    if torch is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError("PyTorch is required to set the training seed. Install torch.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to a JSON/YAML configuration file")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override the maximum number of SFT samples for quick experiments",
    )
    args = parser.parse_args()

    app_cfg = load_config(args.config)
    if app_cfg.experiment is None:
        raise RuntimeError("Configuration file is missing experiment settings.")

    cfg = app_cfg.experiment
    if args.max_samples is not None:
        cfg.dataset.max_samples = args.max_samples

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    LOGGER.info("Loaded configuration: %s", json.dumps(asdict(cfg), indent=2, default=str))

    set_random_seed(cfg.training.seed)

    dataset = build_dataset_dict(cfg)
    enable_eval = cfg.evaluation.enable_eval and "eval" in dataset
    if enable_eval:
        LOGGER.info("Dataset sizes - train: %d, eval: %d", len(dataset["train"]), len(dataset["eval"]))
    else:
        LOGGER.info("Dataset size - train: %d", len(dataset["train"]))
    tokenizer = ensure_tokenizer(cfg.model.model_name_or_path, cfg.model.cache_dir, cfg.model.trust_remote_code)
    model = configure_model(cfg.model.model_name_or_path, cfg)

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    training_args = create_training_arguments(cfg.training, enable_eval)

    def formatting_func(batch: dict) -> List[str]:
        texts = []
        for messages in batch["messages"]:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return texts

    eval_dataset = dataset["eval"] if enable_eval else None

    if SFTTrainer is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError("trl is required for supervised fine-tuning. Install it with `pip install trl`.")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        max_seq_length=cfg.training.max_seq_length,
        packing=True,
        args=training_args,
        peft_config=None,
    )

    if torch.cuda.is_available():
        LOGGER.info("CUDA devices detected: %s", torch.cuda.device_count())

    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    LOGGER.info("Training complete. Artifacts saved to %s", training_args.output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
