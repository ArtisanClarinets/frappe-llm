from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import pytest

from scripts import train_sft

DATASET_PATH = Path("/srv/frappe-llm/datasets/frappe_messages.json")
AXOLOTL_PATH = Path("/srv/frappe-llm/configs/qwen25-coder-3b-qlora-fp16-turing.yaml")
OUTPUT_DIR = Path("/srv/frappe-llm/output/qwen25-coder-3b-frappe-qlora")
PREPARED_PATH = Path("/srv/frappe-llm/prepared")


@pytest.fixture(autouse=True)
def _ensure_environment(tmp_path: Path) -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text("[]\n", encoding="utf-8")

    AXOLOTL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not AXOLOTL_PATH.exists():
        AXOLOTL_PATH.write_text(Path("configs/qwen25-coder-3b-qlora-fp16-turing.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    PREPARED_PATH.mkdir(parents=True, exist_ok=True)


class _Recorder:
    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def __call__(self, command: List[str], check: bool) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        if "preprocess" in command:
            PREPARED_PATH.mkdir(parents=True, exist_ok=True)
        if "train" in command:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
        if "merge-lora" in command:
            (OUTPUT_DIR / "merged").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 0)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(train_sft.subprocess, "run", rec)
    return rec


def test_preprocess_invokes_axolotl(recorder: _Recorder) -> None:
    train_sft.execute_subcommand("preprocess", Path("config.example.json"))
    assert any("preprocess" in call for call in recorder.calls)


def test_train_invokes_axolotl(recorder: _Recorder) -> None:
    train_sft.execute_subcommand("train", Path("config.example.json"))
    assert any("train" in call for call in recorder.calls)


def test_merge_invokes_axolotl(recorder: _Recorder) -> None:
    train_sft.execute_subcommand("preprocess", Path("config.example.json"))
    train_sft.execute_subcommand("train", Path("config.example.json"))
    train_sft.execute_subcommand("merge", Path("config.example.json"))
    assert any("merge-lora" in call for call in recorder.calls)
