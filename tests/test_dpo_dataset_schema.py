import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dpo"
SAMPLE_PATH = DATA_DIR / "frappe_pairs.sample.jsonl"
TRAIN_PATH = DATA_DIR / "frappe_pairs.jsonl"


@pytest.mark.parametrize("path", [SAMPLE_PATH, TRAIN_PATH])
def test_dataset_files_exist(path):
    assert path.exists(), f"Expected dataset at {path}"
    assert path.stat().st_size > 0, f"Dataset {path} is empty"


@pytest.mark.parametrize("path", [SAMPLE_PATH, TRAIN_PATH])
def test_dataset_schema(path):
    lines = path.read_text().strip().splitlines()
    assert lines, f"Dataset {path} must contain at least one record"
    for line in lines:
        record = json.loads(line)
        for field in ("prompt", "chosen", "rejected"):
            assert field in record, f"Missing field: {field}"
            value = record[field]
            assert isinstance(value, str), f"{field} must be a string"
            assert value.strip(), f"{field} must not be empty"
