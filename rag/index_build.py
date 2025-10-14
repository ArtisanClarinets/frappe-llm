"""Utilities to build a FAISS retrieval index for the Frappe RAG pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - faiss is required at runtime
    raise RuntimeError("faiss is required to build the retrieval index") from exc

LOGGER = logging.getLogger(__name__)

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".json5",
    ".yml",
    ".yaml",
    ".md",
    ".txt",
    ".rst",
    ".ini",
    ".cfg",
    ".toml",
    ".sql",
}


@dataclass
class Passage:
    """Container for a single chunk of source text."""

    text: str
    metadata: Dict[str, object]


def load_embedder(model_name: str = "BAAI/bge-m3", device: str | None = None):
    """Instantiate the FlagEmbedding encoder used for dense retrieval."""

    from FlagEmbedding import BGEM3FlagModel  # type: ignore

    LOGGER.info("Loading BGEM3FlagModel %s", model_name)
    return BGEM3FlagModel(model_name, use_fp16=True, device=device)


def discover_files(roots: Sequence[Path]) -> Iterator[Path]:
    """Yield candidate files for indexing under the provided roots."""

    for root in roots:
        if not root.exists():
            LOGGER.debug("Skipping missing root %s", root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in TEXT_EXTENSIONS:
                yield path


def chunk_lines(lines: List[str], chunk_size: int, overlap: int) -> Iterator[Tuple[int, int, str]]:
    """Yield (start, end, text) tuples for the given lines."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    start = 0
    total_lines = len(lines)
    while start < total_lines:
        end = min(total_lines, start + chunk_size)
        chunk = "".join(lines[start:end]).strip()
        if chunk:
            yield start + 1, end, chunk
        if end == total_lines:
            break
        start = end - overlap
        if start < 0:
            start = 0


def collect_passages(roots: Sequence[Path], chunk_size: int, overlap: int) -> List[Passage]:
    """Collect all passages from the provided roots."""

    passages: List[Passage] = []
    for path in discover_files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            LOGGER.debug("Skipping non-UTF8 file %s", path)
            continue
        lines = [line if line.endswith("\n") else f"{line}\n" for line in text.splitlines()]
        for start_line, end_line, chunk in chunk_lines(lines, chunk_size, overlap):
            metadata = {
                "path": str(path.resolve()),
                "relative_path": str(path),
                "start_line": start_line,
                "end_line": end_line,
            }
            passages.append(Passage(text=chunk, metadata=metadata))
    LOGGER.info("Collected %d passages", len(passages))
    return passages


def _encode_batches(embedder, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
    """Encode texts into dense vectors using the provided embedder."""

    all_vectors: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        outputs = embedder.encode(batch, batch_size=len(batch))
        if isinstance(outputs, dict):
            vectors = outputs.get("dense_vecs")
            if vectors is None:
                vectors = outputs.get("sentence_embeddings")
        else:
            vectors = outputs
        if vectors is None:
            raise RuntimeError("Embedder returned no dense vectors")
        all_vectors.append(np.asarray(vectors, dtype="float32"))
    if not all_vectors:
        raise ValueError("No vectors produced")
    stacked = np.vstack(all_vectors)
    faiss.normalize_L2(stacked)
    return stacked


def build_index(
    roots: Sequence[Path],
    out_dir: Path,
    *,
    chunk_size: int = 120,
    overlap: int = 20,
    embedder=None,
) -> Path:
    """Build a FAISS index from the provided roots and return the output directory."""

    out_dir.mkdir(parents=True, exist_ok=True)
    passages = collect_passages(roots, chunk_size, overlap)
    if not passages:
        raise RuntimeError("No passages discovered for indexing")

    texts = [p.text for p in passages]
    embedder = embedder or load_embedder()
    vectors = _encode_batches(embedder, texts)

    dim = vectors.shape[1]
    LOGGER.info("Vectors computed with dimension %d", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = out_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    metadata_path = out_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for idx, passage in enumerate(passages):
            record = {
                "id": idx,
                "text": passage.text,
                **passage.metadata,
            }
            handle.write(json.dumps(record) + "\n")

    manifest = {
        "roots": [str(path) for path in roots],
        "num_passages": len(passages),
        "vector_dimension": dim,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Index written to %s", out_dir)
    return out_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RAG retrieval index")
    parser.add_argument(
        "--roots",
        type=str,
        default=None,
        help="Comma-separated directories to index. Defaults to ../frappe and ./docs if present.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/srv/frappe-llm/rag/index",
        help="Output directory for the FAISS index and metadata",
    )
    parser.add_argument("--chunk-size", type=int, default=120, help="Lines per chunk")
    parser.add_argument("--overlap", type=int, default=20, help="Line overlap between chunks")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="Embedding model name")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    if args.roots:
        roots = [Path(p).expanduser().resolve() for p in args.roots.split(",") if p]
    else:
        candidate_roots = [Path("../frappe"), Path("./docs")]
        roots = [path.resolve() for path in candidate_roots if path.exists()]
        if not roots:
            raise RuntimeError("No default roots found; specify --roots")

    out_dir = Path(args.out).expanduser().resolve()

    embedder = load_embedder(args.model)
    return build_index(roots, out_dir, chunk_size=args.chunk_size, overlap=args.overlap, embedder=embedder)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
