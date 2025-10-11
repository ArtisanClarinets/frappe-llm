"""Normalise assorted corpora into the canonical OpenAI messages format."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

DEFAULT_SYSTEM_PROMPT = "You are a Frappe v15 developer assistant."
LOGGER = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path} line {line_number}: {exc}") from exc


def _from_messages_jsonl(path: Path) -> List[dict]:
    records = []
    for sample in _load_jsonl(path):
        messages = sample.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Record in {path} is missing a 'messages' list: {sample}")
        _validate_messages(messages)
        records.append({"messages": messages})
    return records


def _from_sft_jsonl(path: Path, system_prompt: str) -> List[dict]:
    records = []
    for sample in _load_jsonl(path):
        instruction = sample.get("instruction")
        output = sample.get("output")
        user_input = sample.get("input") or ""
        if not instruction or not output:
            raise ValueError(f"Record in {path} is missing 'instruction' or 'output': {sample}")
        user_prompt = instruction.strip()
        if user_input.strip():
            user_prompt = f"{user_prompt}\n\n{user_input.strip()}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": output.strip()},
        ]
        _validate_messages(messages)
        records.append({"messages": messages})
    return records


def _from_pairs_jsonl(path: Path, system_prompt: str) -> List[dict]:
    records = []
    for sample in _load_jsonl(path):
        prompt = sample.get("prompt") or sample.get("question")
        chosen = sample.get("chosen") or sample.get("answer")
        if not prompt or not chosen:
            LOGGER.debug("Skipping preference record without prompt/chosen: %s", sample)
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": chosen.strip()},
        ]
        _validate_messages(messages)
        records.append({"messages": messages})
    return records


def _validate_messages(messages: Sequence[dict]) -> None:
    if not messages:
        raise ValueError("Messages list may not be empty")
    for message in messages:
        if "role" not in message or "content" not in message:
            raise ValueError(f"Invalid message entry: {message}")
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role '{message['role']}' in message: {message}")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError(f"Message content must be a non-empty string: {message}")


def _deduplicate(records: Iterable[dict]) -> List[dict]:
    seen: set[Tuple[str, str]] = set()
    unique: List[dict] = []
    for record in records:
        user_msg = next((m["content"].strip() for m in record["messages"] if m["role"] == "user"), "")
        assistant_msg = next((m["content"].strip() for m in record["messages"] if m["role"] == "assistant"), "")
        key = (user_msg, assistant_msg)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def convert(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    records: List[dict] = []

    if args.instruction_jsonl:
        LOGGER.info("Loading messages from %s", args.instruction_jsonl)
        records.extend(_from_messages_jsonl(args.instruction_jsonl))
    if args.sft_jsonl:
        LOGGER.info("Loading SFT instructions from %s", args.sft_jsonl)
        records.extend(_from_sft_jsonl(args.sft_jsonl, args.system_prompt))
    if args.pairs_jsonl:
        LOGGER.info("Loading preference data from %s", args.pairs_jsonl)
        records.extend(_from_pairs_jsonl(args.pairs_jsonl, args.system_prompt))

    if not records:
        raise SystemExit("No input records provided. Specify at least one JSONL source.")

    unique_records = _deduplicate(records)
    LOGGER.info("Writing %d unique conversations to %s", len(unique_records), args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(unique_records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    LOGGER.info("Dataset normalisation complete")


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Path to write frappe_messages.json")
    parser.add_argument("--instruction-jsonl", type=Path, help="JSONL file already in messages format")
    parser.add_argument("--sft-jsonl", type=Path, help="Instruction/response JSONL file")
    parser.add_argument("--pairs-jsonl", type=Path, help="Preference JSONL file with prompt/chosen")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to use when synthesising messages",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entry
    args = parse_args()
    convert(args)


if __name__ == "__main__":  # pragma: no cover
    main()
