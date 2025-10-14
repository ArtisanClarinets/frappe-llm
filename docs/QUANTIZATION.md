# Quantization Guidance

This repository targets **6 GB GPUs** (for example, RTX 1660 SUPER) by default. The recommended
approach is to rely on **load-time 4-bit quantization** through `bitsandbytes`, which keeps the
checkpoint footprint small without mutating the on-disk model.

## Recommended: 4-bit NF4 with bitsandbytes

* Ensure CUDA 12.x drivers and matching PyTorch wheels are installed before enabling 4-bit loads.
* At inference, call `AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True,
  bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)`.
* Training stacks (Axolotl) already ship with `bitsandbytes`; the serving API mirrors the same
  settings to stay within the VRAM budget.

## Optional: GPTQ export for offline sharing

If you must share a quantized checkpoint, prefer **GPTQ** exports via tools such as
[`AutoGPTQ`](https://github.com/AutoGPTQ/AutoGPTQ). Always validate the GPTQ artifact against the
baseline evaluation suite before releasing it.

## Deprecated: AutoAWQ

AutoAWQ is deprecated in this stack. On-disk AWQ-style exports introduce additional maintenance
burden and generally lag behind bitsandbytes' runtime kernels. If you require AWQ-like artifacts for
specialized runtimes, evaluate [vLLM's `llm-compressor`](https://github.com/vllm-project/llm-compressor)
separately—those workflows are intentionally out-of-scope for this repository to keep the alignment
stack simple and reproducible.
