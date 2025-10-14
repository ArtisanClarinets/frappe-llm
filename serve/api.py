"""Minimal FastAPI serving stack with optional RAG context injection."""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Thread
from typing import Deque, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from rag.retriever import RAGRetriever, pack_context

LOGGER = logging.getLogger("serve")
logging.basicConfig(level=logging.INFO)

REQUEST_WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 5

MODEL_PATH = Path(os.environ.get("FRAPPE_SERVE_MODEL", "/srv/frappe-llm/models/q25c3b-frappe-dpo/merged"))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("FRAPPE_MAX_NEW_TOKENS", "512"))

app = FastAPI(title="Frappe LLM Service")

_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_rag: Optional[RAGRetriever] = None
_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)


def load_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model path {MODEL_PATH} not found")

    load_kwargs: Dict[str, object] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if os.environ.get("FRAPPE_DISABLE_4BIT") == "1":
        load_kwargs["torch_dtype"] = torch.float16
    else:
        try:
            import bitsandbytes  # type: ignore  # noqa: F401
        except ImportError:
            LOGGER.warning("bitsandbytes unavailable, falling back to float16 load")
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs.update(
                {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_use_double_quant": True,
                }
            )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), **load_kwargs)
    model.eval()
    return model, tokenizer


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    use_rag: bool = False
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, gt=0, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=1.5)


def check_rate_limit(identifier: str) -> None:
    now = time.time()
    bucket = _rate_buckets[identifier]
    while bucket and now - bucket[0] > REQUEST_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


def format_prompt(prompt: str, *, use_rag: bool, rag_context: str) -> str:
    if not use_rag or not rag_context:
        return prompt
    return (
        "You are a helpful assistant for the Frappe framework. Use the provided context when relevant.\n"
        f"Context:\n{rag_context}\n\nUser prompt: {prompt}"
    )


def stream_completion(prompt: str, *, max_new_tokens: int, temperature: float):
    assert _model is not None and _tokenizer is not None
    inputs = _tokenizer(prompt, return_tensors="pt")
    device = next(_model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "pad_token_id": _tokenizer.eos_token_id,
    }
    thread = Thread(target=_model.generate, kwargs=generation_kwargs)
    thread.start()
    for token in streamer:
        yield token
    thread.join()


@app.on_event("startup")
async def startup_event() -> None:
    global _model, _tokenizer, _rag
    _model, _tokenizer = load_model()
    try:
        _rag = RAGRetriever()
    except FileNotFoundError:
        LOGGER.warning("RAG index not found; continuing without retrieval support")
        _rag = None


@app.post("/query")
async def query_endpoint(payload: QueryRequest, request: Request) -> StreamingResponse:
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    rag_context = ""
    rag_tokens = 0
    if payload.use_rag:
        if _rag is None:
            raise HTTPException(status_code=503, detail="RAG index unavailable")
        passages = _rag.retrieve(payload.prompt, top_k=8)
        rag_context = pack_context(passages)
        rag_tokens = len(rag_context.split())

    prompt = format_prompt(payload.prompt, use_rag=payload.use_rag, rag_context=rag_context)
    prompt_tokens = len(prompt.split())

    start_time = time.perf_counter()

    def iterator():
        for chunk in stream_completion(
            prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
        ):
            yield chunk

    response = StreamingResponse(iterator(), media_type="text/plain")

    latency = time.perf_counter() - start_time
    LOGGER.info(
        "served prompt_len=%d rag_tokens=%d latency=%.2fs rag=%s", prompt_tokens, rag_tokens, latency, payload.use_rag
    )
    return response


__all__ = ["app"]
