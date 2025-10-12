"""High level orchestration utilities for the Frappe LLM project."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, load_config

LOGGER = logging.getLogger(__name__)


def run_command(command: list[str], cwd: Optional[Path] = None) -> None:
    """Execute a subprocess and stream its output to the console."""

    LOGGER.debug("Executing command: %s", " ".join(command))
    process = subprocess.Popen(command, cwd=str(cwd) if cwd else None)
    process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def launch_sft(config_path: Path, max_samples: Optional[int]) -> None:
    """Trigger the supervised fine-tuning run."""

    command = [sys.executable, "-m", "scripts.train_sft", "--config", str(config_path)]
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    run_command(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the training configuration file")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on training samples for smoke tests",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()

    app_cfg: AppConfig = load_config(args.config)
    if app_cfg.experiment is None:
        raise RuntimeError("Configuration file is missing experiment settings.")

    LOGGER.info("Starting SFT run for model %s", app_cfg.experiment.model.model_name_or_path)
    launch_sft(args.config, args.max_samples)


if __name__ == "__main__":  # pragma: no cover
    main()
