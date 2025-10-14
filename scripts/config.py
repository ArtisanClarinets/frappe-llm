"""Configuration models and utilities for the Frappe LLM training stack."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:  # pragma: no cover - optional dependency at runtime
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Base experiment configuration (used by legacy training utilities)
# ---------------------------------------------------------------------------


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

    host: str = "127.0.0.1"
    port: int = 8000
    concurrency: int = 4
    request_timeout: int = 60

    def validate(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ValueError("Server port must be between 1 and 65535")


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


# ---------------------------------------------------------------------------
# Application configuration wrapper (used by tests and CLI entry points)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PathsConfig:
    """File-system layout for the application."""

    dataset_path: Path = Path("/srv/frappe-llm/datasets/frappe_messages.json")
    axolotl_config_path: Path = Path(
        "/srv/frappe-llm/configs/qwen25-coder-3b-qlora-fp16-turing.yaml"
    )
    output_dir: Path = Path("/srv/frappe-llm/output/qwen25-coder-3b-frappe-qlora")

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.axolotl_config_path = Path(self.axolotl_config_path)
        self.output_dir = Path(self.output_dir)

    def apply_env_overrides(self) -> None:
        overrides = {
            "FRAPPE_LLM_DATASET_PATH": "dataset_path",
            "FRAPPE_LLM_AXOLOTL_CONFIG": "axolotl_config_path",
            "FRAPPE_LLM_OUTPUT_DIR": "output_dir",
        }
        for env_var, attr in overrides.items():
            value = os.getenv(env_var)
            if value:
                setattr(self, attr, Path(value))


@dataclass(slots=True)
class AppConfig:
    """High-level application configuration exposed to entry points."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    experiment: Optional[ExperimentConfig] = None

    def validate(self) -> None:
        self.paths.apply_env_overrides()
        self.server.validate()


def _load_mapping(path: Path) -> Dict[str, object]:
    if not path.exists():  # pragma: no cover - defensive branch
        raise FileNotFoundError(path)

    if path.suffix in {".yml", ".yaml"}:
        if yaml is None:  # pragma: no cover - optional dependency guard
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
    return data


def load_config(path: Path | str) -> AppConfig:
    """Load an :class:`AppConfig` from a JSON or YAML file."""

    mapping = _load_mapping(Path(path))

    app_config = AppConfig()

    if "paths" in mapping:
        app_config.paths = PathsConfig(**mapping["paths"])
    if "server" in mapping:
        app_config.server = ServerConfig(**mapping["server"])

    experiment_keys = {"dataset", "model", "lora", "training", "evaluation"}
    if experiment_keys.intersection(mapping):
        app_config.experiment = ExperimentConfig.from_mapping(mapping)

    app_config.validate()
    return app_config


__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "LoraConfig",
    "TrainingArgumentsConfig",
    "EvaluationConfig",
    "ServerConfig",
    "ExperimentConfig",
    "PathsConfig",
    "AppConfig",
    "load_config",
]
