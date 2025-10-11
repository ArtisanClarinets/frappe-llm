"""FastAPI based serving layer for Frappe LLM models."""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, TextIteratorStreamer

try:
    from peft import AutoPeftModelForCausalLM
except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
    raise SystemExit("peft is required to load LoRA adapters. Install with `pip install peft`.") from exc

from .config import ExperimentConfig, ServerConfig, load_config

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Frappe LLM Serving", version="1.0.0")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class GenerateResponse(BaseModel):
    completion: str


@lru_cache(maxsize=1)
def get_cfg() -> ExperimentConfig:
    config_path = Path("scripts/config.server.json")
    if not config_path.exists():
        raise RuntimeError("Server configuration missing. Create scripts/config.server.json")
    return load_config(config_path)


@lru_cache(maxsize=1)
def get_server_config() -> ServerConfig:
    return get_cfg().server


@lru_cache(maxsize=1)
def get_model_components() -> Dict[str, Any]:
    cfg = get_cfg()
    tokenizer = AutoTokenizer.from_pretrained(cfg.training.output_dir, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        cfg.training.output_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        is_trainable=False,
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


@app.on_event("startup")
async def on_startup() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    get_model_components()
    LOGGER.info("Model loaded and ready for inference")


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:  # pragma: no cover
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    components = get_model_components()
    tokenizer: AutoTokenizer = components["tokenizer"]
    model: AutoPeftModelForCausalLM = components["model"]

    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.temperature > 0,
    )

    loop = asyncio.get_event_loop()

    def threaded_generate() -> None:
        with torch.inference_mode():
            model.generate(**generation_kwargs)

    await loop.run_in_executor(None, threaded_generate)

    output_text = "".join(streamer)
    if not output_text:
        raise HTTPException(status_code=500, detail="Model returned empty completion")
    return GenerateResponse(completion=output_text)
