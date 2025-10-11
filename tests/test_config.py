from __future__ import annotations

from pathlib import Path

import shutil

import pytest

from scripts.config import AppConfig, load_config


CANONICAL_CONFIG = Path("config.example.json")
CANONICAL_YAML = Path("configs/qwen25-coder-3b-qlora-fp16-turing.yaml")
DATASET_PATH = Path("/srv/frappe-llm/datasets/frappe_messages.json")
AXOLOTL_PATH = Path("/srv/frappe-llm/configs/qwen25-coder-3b-qlora-fp16-turing.yaml")
OUTPUT_DIR = Path("/srv/frappe-llm/output/qwen25-coder-3b-frappe-qlora")


def _ensure_canonical_layout() -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        DATASET_PATH.write_text("[]\n", encoding="utf-8")

    AXOLOTL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not AXOLOTL_PATH.exists():
        shutil.copy2(CANONICAL_YAML, AXOLOTL_PATH)

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)


def test_load_config_defaults() -> None:
    _ensure_canonical_layout()
    cfg: AppConfig = load_config(CANONICAL_CONFIG)
    assert cfg.paths.dataset_path == DATASET_PATH
    assert cfg.paths.axolotl_config_path == AXOLOTL_PATH
    assert cfg.paths.output_dir == OUTPUT_DIR
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 8000


def test_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]\n", encoding="utf-8")

    yaml_copy = tmp_path / "axolotl.yaml"
    yaml_copy.write_text(CANONICAL_YAML.read_text(encoding="utf-8"), encoding="utf-8")

    output_dir = tmp_path / "output"

    monkeypatch.setenv("FRAPPE_LLM_DATASET_PATH", str(dataset))
    monkeypatch.setenv("FRAPPE_LLM_AXOLOTL_CONFIG", str(yaml_copy))
    monkeypatch.setenv("FRAPPE_LLM_OUTPUT_DIR", str(output_dir))

    cfg = load_config(CANONICAL_CONFIG)
    assert cfg.paths.dataset_path == dataset
    assert cfg.paths.axolotl_config_path == yaml_copy
    assert cfg.paths.output_dir == output_dir


@pytest.mark.parametrize("bad_port", [0, 65536])
def test_invalid_server_port(monkeypatch: pytest.MonkeyPatch, bad_port: int, tmp_path: Path) -> None:
    _ensure_canonical_layout()
    config_copy = tmp_path / "config.json"
    config_copy.write_text(
        '{"paths": {"dataset_path": "' + str(DATASET_PATH) + '", "axolotl_config_path": "' + str(AXOLOTL_PATH) + '", "output_dir": "' + str(OUTPUT_DIR) + '"}, "server": {"host": "127.0.0.1", "port": ' + str(bad_port) + ', "concurrency": 4, "request_timeout": 60}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(config_copy)
