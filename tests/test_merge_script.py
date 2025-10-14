from pathlib import Path
import subprocess

MERGE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_dpo.sh"


def test_merge_script_syntax():
    result = subprocess.run(["bash", "-n", str(MERGE_SCRIPT)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_merge_script_invocation():
    script_body = MERGE_SCRIPT.read_text()
    assert "axolotl merge-lora /srv/frappe-llm/configs/qwen25-coder-3b-dpo-frappe.yaml" in script_body
    assert "--lora-model-dir /srv/frappe-llm/models/q25c3b-frappe-dpo" in script_body
