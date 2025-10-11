"""Build a FAISS knowledge index for retrieval-augmented workflows.

The command operates independently from the Axolotl pipeline so that
search/RAG experiments never pollute `/srv/frappe-llm/output` or
`/srv/frappe-llm/prepared`.  Provide explicit paths and the script will
log every step before writing the resulting index.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)
FORBIDDEN_ROOTS = {Path("/srv/frappe-llm/output"), Path("/srv/frappe-llm/prepared")}


def read_documents(path: Path) -> List[str]:
    documents: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON on line {line_number} of {path}: {exc}") from exc
                documents.append(payload.get("text") or payload.get("prompt") or json.dumps(payload))
            else:
                documents.append(line)
    return documents


def embed_documents(model: SentenceTransformer, documents: Iterable[str]) -> np.ndarray:
    vectors = model.encode(
        list(documents),
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.astype(np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def _validate_output_path(path: Path) -> None:
    resolved = path.resolve()
    for forbidden in FORBIDDEN_ROOTS:
        if forbidden in resolved.parents or resolved == forbidden:
            raise ValueError(
                f"Refusing to write FAISS index inside protected directory {forbidden}. Choose a different output path."
            )


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to a JSONL/text corpus")
    parser.add_argument("--output", type=Path, required=True, help="Destination for the FAISS index")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Sentence transformer model for embeddings",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    LOGGER.info("Reading corpus from %s", args.input)

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    _validate_output_path(args.output)

    documents = read_documents(args.input)
    if not documents:
        raise ValueError("No documents found in input corpus")

    LOGGER.info("Loaded %d documents", len(documents))

    embedder = SentenceTransformer(args.model)
    LOGGER.info("Encoding documents with %s", args.model)
    vectors = embed_documents(embedder, documents)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    index = build_faiss_index(vectors)
    faiss.write_index(index, str(args.output))
    LOGGER.info("Saved FAISS index to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
