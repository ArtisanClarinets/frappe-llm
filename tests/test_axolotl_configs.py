from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover
    pytest.skip("PyYAML is required for config validation", allow_module_level=True)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = [
    CONFIG_DIR / "qwen25-coder-3b-dpo-frappe.yaml",
    CONFIG_DIR / "qwen25-coder-3b-orpo-frappe.yaml",
]

REQUIRED_KEYS = {
    "base_model",
    "pretrained_model_name_or_path",
    "datasets",
    "lora_target_modules",
    "output_dir",
    "rl",
}
EXPECTED_LORA_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
}


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_config_keys(config_path):
    assert config_path.exists(), f"Missing config: {config_path}"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert isinstance(cfg, dict), "Config must deserialize to a dict"
    missing = REQUIRED_KEYS.difference(cfg)
    assert not missing, f"Missing required keys: {missing}"
    assert cfg["pretrained_model_name_or_path"].startswith("/srv/frappe-llm/"), (
        "pretrained_model_name_or_path must point to local SFT merge"
    )
    assert isinstance(cfg.get("datasets"), list) and cfg["datasets"], "Datasets must be defined"
    targets = set(cfg.get("lora_target_modules", []))
    assert EXPECTED_LORA_TARGETS.issubset(targets), "LoRA target modules incomplete"


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_dataset_path_alignment(config_path):
    cfg = yaml.safe_load(config_path.read_text())
    dataset_paths = [entry["path"] for entry in cfg.get("datasets", [])]
    for dataset_path in dataset_paths:
        assert dataset_path.startswith("/srv/frappe-llm/data/dpo/"), (
            "Dataset path must live under /srv/frappe-llm/data/dpo"
        )
