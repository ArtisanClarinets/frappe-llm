"""FastAPI-based inference API for local experimentation.

The service loads the Axolotl-produced LoRA adapter from the configured
output directory and exposes a `/generate` endpoint.  It is intended for
use behind an authenticating reverse proxy with rate limiting (e.g.,
Nginx `limit_req zone=llm burst=5 nodelay;`).  Production serving should
prefer vLLM as documented in the project README.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, TextIteratorStreamer

try:  # pragma: no cover - dependency hint for local testing
    from peft import AutoPeftModelForCausalLM
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("peft is required to load LoRA adapters. Install with `pip install peft`.") from exc

from .config import AppConfig, load_config

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Frappe LLM Serving", version="2.0.0")

_CONFIG_PATH = Path("scripts/config.server.json")


def configure_settings_path(path: Path) -> None:
    global _CONFIG_PATH
    _CONFIG_PATH = path
    get_cfg.cache_clear()
    get_server_config.cache_clear()
    get_model_components.cache_clear()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class GenerateResponse(BaseModel):
    completion: str


@lru_cache(maxsize=1)
def get_cfg() -> AppConfig:
    if not _CONFIG_PATH.exists():
        raise RuntimeError(f"Server configuration missing at {_CONFIG_PATH}." " Copy config.example.json or pass --settings.")
    cfg = load_config(_CONFIG_PATH)
    return cfg


@lru_cache(maxsize=1)
def get_server_config() -> Any:
    return get_cfg().server


@lru_cache(maxsize=1)
def get_model_components() -> Dict[str, Any]:
    cfg = get_cfg()
    adapter_dir = cfg.paths.output_dir
    if not adapter_dir.exists():
        raise RuntimeError(
            f"LoRA output directory not found: {adapter_dir}. Run training before starting the API."
        )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        is_trainable=False,
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


@app.on_event("startup")
async def on_startup() -> None:  # pragma: no cover - lifecycle hook
    LOGGER.info("Starting FastAPI server with config %s", _CONFIG_PATH)
    get_model_components()
    LOGGER.info("Model loaded and ready for inference")


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:  # pragma: no cover - simple endpoint
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


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=_CONFIG_PATH, help="Path to JSON/YAML runtime settings")
    parser.add_argument("--host", type=str, default=None, help="Override bind host")
    parser.add_argument("--port", type=int, default=None, help="Override bind port")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI wrapper
    import uvicorn

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    configure_settings_path(args.settings)
    server_cfg = get_server_config()
    host = args.host or server_cfg.host
    port = args.port or server_cfg.port

    LOGGER.info("Launching uvicorn on %s:%s", host, port)
    uvicorn.run("scripts.api:app", host=host, port=port, workers=server_cfg.concurrency)


if __name__ == "__main__":  # pragma: no cover
    main()
