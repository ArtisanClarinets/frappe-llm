"""Runtime configuration helpers for the Frappe LLM training stack.

The configuration surface is intentionally small: a trio of
filesystem paths plus optional FastAPI server settings.  Every
consumer loads the same schema so that dataset, Axolotl YAML, and
output directories stay aligned across training and serving.

Environment variables override disk values for rapid experiments:

- ``FRAPPE_LLM_DATASET_PATH``
- ``FRAPPE_LLM_AXOLOTL_CONFIG``
- ``FRAPPE_LLM_OUTPUT_DIR``

The defaults match the Fortune-500-ready layout documented in the
project README.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
import os

try:  # Optional dependency for YAML configurations
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised via tests without PyYAML
    yaml = None  # type: ignore[assignment]

DEFAULT_DATASET_PATH = Path("/srv/frappe-llm/datasets/frappe_messages.json")
DEFAULT_AXOLOTL_CONFIG_PATH = Path("/srv/frappe-llm/configs/qwen25-coder-3b-qlora-fp16-turing.yaml")
DEFAULT_OUTPUT_DIR = Path("/srv/frappe-llm/output/qwen25-coder-3b-frappe-qlora")


@dataclass(slots=True)
class PathConfig:
    """Filesystem paths shared by training and serving."""

    dataset_path: Path = DEFAULT_DATASET_PATH
    axolotl_config_path: Path = DEFAULT_AXOLOTL_CONFIG_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path).expanduser()
        self.axolotl_config_path = Path(self.axolotl_config_path).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()

    def apply_env_overrides(self) -> None:
        dataset_override = os.environ.get("FRAPPE_LLM_DATASET_PATH")
        if dataset_override:
            self.dataset_path = Path(dataset_override).expanduser()

        axolotl_override = os.environ.get("FRAPPE_LLM_AXOLOTL_CONFIG")
        if axolotl_override:
            self.axolotl_config_path = Path(axolotl_override).expanduser()

        output_override = os.environ.get("FRAPPE_LLM_OUTPUT_DIR")
        if output_override:
            self.output_dir = Path(output_override).expanduser()

    def validate(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.dataset_path}. "
                "Place frappe_messages.json there or set FRAPPE_LLM_DATASET_PATH."
            )
        if not self.axolotl_config_path.exists():
            raise FileNotFoundError(
                f"Axolotl YAML not found: {self.axolotl_config_path}. "
                "Copy configs/qwen25-coder-3b-qlora-fp16-turing.yaml into place or set FRAPPE_LLM_AXOLOTL_CONFIG."
            )
        # Ensure the parent directory exists to avoid late failures
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ServerConfig:
    """FastAPI serving configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    concurrency: int = 4
    request_timeout: int = 60

    def validate(self) -> None:
        if not (0 < self.port < 65536):
            raise ValueError(f"Invalid port number: {self.port}")
        if self.concurrency <= 0:
            raise ValueError("Server concurrency must be positive")
        if self.request_timeout <= 0:
            raise ValueError("Request timeout must be positive")


@dataclass(slots=True)
class AppConfig:
    """Root configuration object consumed by scripts."""

    paths: PathConfig
    server: ServerConfig

    def validate(self) -> None:
        self.paths.validate()
        self.server.validate()


def _build_from_mapping(mapping: Dict[str, Any]) -> AppConfig:
    paths_data = mapping.get("paths", {})
    server_data = mapping.get("server", {})

    if not isinstance(paths_data, dict):
        raise TypeError("'paths' must be a mapping")
    if not isinstance(server_data, dict):
        raise TypeError("'server' must be a mapping")

    paths = PathConfig(**paths_data)
    paths.apply_env_overrides()
    server = ServerConfig(**server_data)
    config = AppConfig(paths=paths, server=server)
    config.validate()
    return config


def load_config(path: Path) -> AppConfig:
    """Load configuration from JSON or YAML, applying env overrides."""

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".yml", ".yaml"}:
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
        raise TypeError("Configuration file must contain a JSON/YAML object at the top level")

    return _build_from_mapping(data)


__all__ = ["PathConfig", "ServerConfig", "AppConfig", "load_config"]
