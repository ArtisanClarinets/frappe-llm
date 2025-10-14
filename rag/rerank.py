"""Inference-time reranking utilities for the Frappe RAG pipeline."""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

try:  # pragma: no cover - optional dependency during import
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - handled lazily at runtime
    AutoModelForSequenceClassification = AutoTokenizer = Any  # type: ignore

try:  # pragma: no cover - optional dependency during import
    import torch
except ImportError:  # pragma: no cover - handled lazily at runtime
    torch = None  # type: ignore

class Reranker:
    """Thin wrapper around jinaai/jina-reranker-v3."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3",
        *,
        model: Optional[AutoModelForSequenceClassification] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        device_map: str | dict | None = "auto",
    ) -> None:
        if model is None or tokenizer is None:
            if torch is None:
                raise RuntimeError("PyTorch is required to load the default reranker model")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
        self.model = model.eval()
        self.tokenizer = tokenizer

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, object]],
        *,
        top_k: int = 8,
    ) -> List[Mapping[str, object]]:
        if not candidates:
            return []
        if torch is None:
            raise RuntimeError("PyTorch is required to run the reranker")

        texts = [str(candidate.get("text", "")) for candidate in candidates]
        inputs = self.tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits.squeeze(-1)
        scores = logits.detach().float().cpu().tolist()

        ranked: List[Mapping[str, object]] = []
        for score, candidate in sorted(zip(scores, candidates, strict=False), key=lambda item: item[0], reverse=True):
            enriched = dict(candidate)
            enriched["rerank_score"] = float(score)
            ranked.append(enriched)
        return ranked[:top_k]


__all__ = ["Reranker"]
