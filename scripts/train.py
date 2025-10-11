"""Convenience wrapper for the Axolotl training pipeline.

This script chains the `preprocess` and `train` subcommands from
``scripts.train_sft`` to provide a single-entry execution path.  It is
safe to run on developer workstations and logs each stage before
delegating to Axolotl.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import train_sft

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=Path("config.example.json"), help="Path to runtime settings")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip axolotl preprocess stage")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge-lora after training")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entry
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    LOGGER.info("Executing training pipeline with settings %s", args.settings)

    if not args.skip_preprocess:
        LOGGER.info("Running preprocess stage")
        train_sft.execute_subcommand("preprocess", args.settings)
    else:
        LOGGER.info("Skipping preprocess as requested")

    LOGGER.info("Running train stage")
    train_sft.execute_subcommand("train", args.settings)

    if not args.skip_merge:
        LOGGER.info("Running merge stage")
        train_sft.execute_subcommand("merge", args.settings)
    else:
        LOGGER.info("Skipping merge as requested")

    LOGGER.info("Training pipeline completed successfully")


if __name__ == "__main__":  # pragma: no cover
    main()
