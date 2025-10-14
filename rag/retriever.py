"""Retrieval helpers that combine FAISS search with reranking."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("faiss is required for retrieval") from exc

from .index_build import load_embedder
from .rerank import Reranker

LOGGER = logging.getLogger(__name__)
DEFAULT_INDEX_DIR = Path("/srv/frappe-llm/rag/index")


@dataclass
class Passage:
    text: str
    metadata: Dict[str, object]
    vector_score: float


class RAGRetriever:
    """Load a FAISS index and perform retrieval with reranking."""

    def __init__(
        self,
        index_dir: Path | None = None,
        *,
        embedder=None,
        reranker: Reranker | None = None,
        initial_k: int = 50,
    ) -> None:
        self.index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        index_path = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.jsonl"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {self.index_dir}. Run rag/index_build.py first."
            )
        self.index = faiss.read_index(str(index_path))
        self.metadata = self._load_metadata(metadata_path)
        self.embedder = embedder or load_embedder()
        self.reranker = reranker or Reranker()
        self.initial_k = initial_k
        LOGGER.info("Loaded index with %d vectors", self.index.ntotal)

    @staticmethod
    def _load_metadata(path: Path) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.append(json.loads(line))
        return records

    def _embed_query(self, query: str) -> np.ndarray:
        outputs = self.embedder.encode([query], batch_size=1)
        if isinstance(outputs, dict):
            vectors = outputs.get("dense_vecs")
            if vectors is None:
                vectors = outputs.get("sentence_embeddings")
        else:
            vectors = outputs
        if vectors is None:
            raise RuntimeError("Embedder returned no vectors for query")
        vector = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(vector)
        return vector

    def search(self, query: str, *, top_k: int | None = None) -> List[Passage]:
        k = top_k or self.initial_k
        vector = self._embed_query(query)
        scores, indices = self.index.search(vector, k)
        hits: List[Passage] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx == -1:
                continue
            meta = dict(self.metadata[int(idx)])
            hits.append(
                Passage(
                    text=str(meta.pop("text", "")),
                    metadata=meta,
                    vector_score=float(score),
                )
            )
        return hits

    def retrieve(self, query: str, *, top_k: int = 8) -> List[Dict[str, object]]:
        candidates = [
            {
                "text": passage.text,
                **passage.metadata,
                "vector_score": passage.vector_score,
            }
            for passage in self.search(query)
        ]
        reranked = self.reranker.rerank(query, candidates, top_k=self.initial_k)
        results: List[Dict[str, object]] = []
        seen: set[tuple[str, int, int]] = set()
        for candidate in reranked:
            path = str(candidate.get("path") or candidate.get("relative_path"))
            key = (path, int(candidate.get("start_line", 0)), int(candidate.get("end_line", 0)))
            if key in seen:
                continue
            seen.add(key)
            results.append(candidate)
            if len(results) >= top_k:
                break
        return results


@lru_cache(maxsize=4)
def _cached_retriever(index_key: str | None) -> RAGRetriever:
    path = Path(index_key).resolve() if index_key else None
    return RAGRetriever(index_dir=path)


def get_retriever(index_dir: Path | None = None) -> RAGRetriever:
    index_key = str(Path(index_dir).resolve()) if index_dir else None
    return _cached_retriever(index_key)


def retrieve(query: str, k: int = 8, *, index_dir: Path | None = None, retriever: RAGRetriever | None = None) -> List[Dict[str, object]]:
    retriever = retriever or get_retriever(index_dir=index_dir)
    return retriever.retrieve(query, top_k=k)


def pack_context(passages: Sequence[Mapping[str, object]], max_tokens: int = 1024) -> str:
    segments: List[str] = []
    token_budget = 0
    for passage in passages:
        text = str(passage.get("text", "")).strip()
        if not text:
            continue
        citation = f"[{passage.get('relative_path', passage.get('path', ''))}:{passage.get('start_line')}-{passage.get('end_line')}]"
        segment = f"{citation}\n{text}"
        tokens = len(segment.split())
        if token_budget + tokens > max_tokens:
            break
        segments.append(segment)
        token_budget += tokens
    return "\n\n".join(segments)


__all__ = ["RAGRetriever", "retrieve", "pack_context"]
