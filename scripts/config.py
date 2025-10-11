"""Configuration models and utilities for the Frappe LLM training stack.

This module centralises every runtime configuration object that the
scripts in :mod:`scripts` consume.  Each dataclass is intentionally kept
small and serialisable to both JSON and YAML so that the configuration can
be stored in Git and versioned just like code.  The defaults aim to be
sensible for 2025 era hardware (A100/H100 class GPUs or modern consumer
GPUs with bfloat16 support) and the Qwen2.5 Coder 3B Instruct checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import json

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for datasets used in the pipeline."""

    instruction_path: Path
    sft_path: Path
    prefs_path: Optional[Path] = None
    eval_path: Optional[Path] = None
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        self.instruction_path = Path(self.instruction_path)
        self.sft_path = Path(self.sft_path)
        self.prefs_path = Path(self.prefs_path) if self.prefs_path else None
        self.eval_path = Path(self.eval_path) if self.eval_path else None


@dataclass(slots=True)
class ModelConfig:
    """Configuration for the base model and tokenizer."""

    model_name_or_path: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    cache_dir: Optional[Path] = Path("hf_cache")
    token: Optional[str] = None
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    load_in_4bit: bool = True

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir) if self.cache_dir else None


@dataclass(slots=True)
class LoraConfig:
    """Parameter-Efficient Fine-Tuning (PEFT) configuration."""

    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(slots=True)
class TrainingArgumentsConfig:
    """High level training arguments to feed into the Trainer classes."""

    output_dir: Path = Path("models/finetune/q25c3b-frappe-sft")
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    bf16: bool = True
    fp16: bool = False
    seed: int = 2025
    report_to: List[str] = field(default_factory=lambda: ["tensorboard", "wandb"])
    torch_compile: bool = True
    optim: str = "paged_adamw_32bit"
    max_seq_length: int = 4096

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


@dataclass(slots=True)
class EvaluationConfig:
    """Runtime options for evaluation and synthetic test generation."""

    enable_eval: bool = True
    generation_max_new_tokens: int = 512
    generation_temperature: float = 0.2
    evaluation_examples: int = 64


@dataclass(slots=True)
class ServerConfig:
    """Configuration shared by the FastAPI inference service."""

    host: str = "0.0.0.0"
    port: int = 8000
    concurrency: int = 4
    request_timeout: int = 60


@dataclass(slots=True)
class ExperimentConfig:
    """Container dataclass aggregating the entire experiment setup."""

    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingArgumentsConfig = field(default_factory=TrainingArgumentsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_mapping(cls, mapping: Dict[str, object]) -> "ExperimentConfig":
        """Build an :class:`ExperimentConfig` from a raw mapping."""

        def build(subcls, key):
            data = mapping.get(key, {})
            if isinstance(data, dict):
                return subcls(**data)
            raise TypeError(f"Expected dict for '{key}', received {type(data)!r}")

        dataset_cfg = build(DatasetConfig, "dataset")
        model_cfg = build(ModelConfig, "model") if "model" in mapping else ModelConfig()
        lora_cfg = build(LoraConfig, "lora") if "lora" in mapping else LoraConfig()
        training_cfg = (
            build(TrainingArgumentsConfig, "training")
            if "training" in mapping
            else TrainingArgumentsConfig()
        )
        evaluation_cfg = (
            build(EvaluationConfig, "evaluation")
            if "evaluation" in mapping
            else EvaluationConfig()
        )
        server_cfg = build(ServerConfig, "server") if "server" in mapping else ServerConfig()
        return cls(
            dataset=dataset_cfg,
            model=model_cfg,
            lora=lora_cfg,
            training=training_cfg,
            evaluation=evaluation_cfg,
            server=server_cfg,
        )


def load_config(path: Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a JSON or YAML file."""

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to read YAML configuration files. Install it with `pip install pyyaml`."
            )
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

    if not isinstance(data, dict):
        raise TypeError("Top level configuration must be a JSON object / YAML mapping")

    return ExperimentConfig.from_mapping(data)


__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "LoraConfig",
    "TrainingArgumentsConfig",
    "EvaluationConfig",
    "ServerConfig",
    "ExperimentConfig",
    "load_config",
]
