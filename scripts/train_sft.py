"""Axolotl wrapper with explicit preprocess, train, and merge steps."""
from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:  # Optional dependency for parsing Axolotl YAML
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
    yaml = None  # type: ignore[assignment]

from .config import AppConfig, load_config

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AxolotlSpec:
    config_path: Path
    dataset_paths: List[Path]
    output_dir: Path
    prepared_path: Path


def _read_axolotl_yaml(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML is required to parse Axolotl configs. Install with `pip install pyyaml`.")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - invalid YAML
        raise ValueError(f"Failed to parse Axolotl YAML at {path}: {exc}") from exc


def _resolve_axolotl_spec(app_cfg: AppConfig) -> AxolotlSpec:
    axo_cfg = _read_axolotl_yaml(app_cfg.paths.axolotl_config_path)

    dataset_entries: List[Path] = []
    for entry in axo_cfg.get("datasets", []):
        candidate = Path(entry.get("path", "")).expanduser()
        if candidate:
            dataset_entries.append(candidate)
    if not dataset_entries:
        dataset_entries.append(app_cfg.paths.dataset_path)

    prepared_path = Path(axo_cfg.get("dataset_prepared_path", "/srv/frappe-llm/prepared")).expanduser()
    output_dir = Path(axo_cfg.get("output_dir", app_cfg.paths.output_dir)).expanduser()

    # Enforce alignment with runtime config
    if app_cfg.paths.dataset_path not in dataset_entries:
        raise ValueError(
            "Axolotl YAML dataset path does not match runtime config. "
            f"Expected {app_cfg.paths.dataset_path}, found {dataset_entries}."
        )
    if output_dir != app_cfg.paths.output_dir:
        raise ValueError(
            "Axolotl YAML output_dir does not match runtime config. "
            f"Expected {app_cfg.paths.output_dir}, found {output_dir}."
        )

    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    return AxolotlSpec(
        config_path=app_cfg.paths.axolotl_config_path,
        dataset_paths=dataset_entries,
        output_dir=output_dir,
        prepared_path=prepared_path,
    )


def _run_axolotl(subcommand: str, spec: AxolotlSpec) -> None:
    command = ["axolotl", subcommand, str(spec.config_path)]
    LOGGER.info("Executing: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - command missing
        raise SystemExit("axolotl CLI not found. Install axolotl into your active virtualenv.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Axolotl command failed with exit code {exc.returncode}") from exc


def _ensure_dataset_exists(spec: AxolotlSpec) -> None:
    for path in spec.dataset_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset fragment missing: {path}. Run scripts.convert_to_messages to create frappe_messages.json."
            )


def handle_preprocess(app_cfg: AppConfig) -> None:
    spec = _resolve_axolotl_spec(app_cfg)
    _ensure_dataset_exists(spec)
    _run_axolotl("preprocess", spec)
    if not spec.prepared_path.exists():
        raise FileNotFoundError(
            f"Expected Axolotl to create prepared dataset at {spec.prepared_path}. Check preprocess logs for errors."
        )
    LOGGER.info("Preprocess complete: %s", spec.prepared_path)


def handle_train(app_cfg: AppConfig) -> None:
    spec = _resolve_axolotl_spec(app_cfg)
    if not spec.prepared_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset missing at {spec.prepared_path}. Run the preprocess stage first."
        )
    _run_axolotl("train", spec)
    if not spec.output_dir.exists():
        raise FileNotFoundError(
            f"Training finished without producing {spec.output_dir}. Inspect Axolotl output for failures."
        )
    LOGGER.info("Training artifacts available in %s", spec.output_dir)


def handle_merge(app_cfg: AppConfig) -> None:
    spec = _resolve_axolotl_spec(app_cfg)
    adapter_file = spec.output_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(
            f"LoRA adapter missing at {adapter_file}. Complete training before attempting merge-lora."
        )
    _run_axolotl("merge-lora", spec)
    merged_dir = spec.output_dir / "merged"
    if not merged_dir.exists():
        raise FileNotFoundError(
            f"Expected merged weights in {merged_dir}. Axolotl merge-lora did not finish successfully."
        )
    LOGGER.info("Merged checkpoint ready at %s", merged_dir)


def execute_subcommand(subcommand: str, settings_path: Path) -> None:
    app_cfg = load_config(settings_path)
    logging.getLogger("axolotl").setLevel(logging.WARNING)
    if subcommand == "preprocess":
        handle_preprocess(app_cfg)
    elif subcommand == "train":
        handle_train(app_cfg)
    elif subcommand == "merge":
        handle_merge(app_cfg)
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unknown subcommand: {subcommand}")


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preprocess", "train", "merge"], help="Pipeline stage to execute")
    parser.add_argument("--settings", type=Path, default=Path("config.example.json"), help="Path to runtime settings")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entry
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    LOGGER.info("Running '%s' with settings %s", args.command, args.settings)
    execute_subcommand(args.command, args.settings)
    LOGGER.info("%s stage finished successfully", args.command.capitalize())


if __name__ == "__main__":  # pragma: no cover
    main()
