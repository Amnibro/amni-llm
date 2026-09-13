# Amni-LLM Changelog

## v0.2.0-alpha — 2026-04-28 (CODE COMPLETE, NEEDS GPU VERIFICATION)

**Decisions locked:**
- Kernels derived clean-room from GGML spec text (no llama.cpp Vulkan source copy)
- Every kernel ships in two variants: `f16` (primary) + `f32` (fallback)
- GGUF parser supports V1, V2, V3 — universal coverage from day one
- Mobile profile adopts Amni-Haven `local_ai.html` battle-tested patterns

**Originals backed up** to `backups/<file>.v0.1.0.bak`.

**Code shipped (Node-tested, browser GPU dispatch unverified):**

- `backends/wgpu/gguf.js` — V1/V2/V3 parser (26/26 pass on real HF GGUFs incl. Qwen2.5-0.5B + Qwen3.5-0.8B)
- `backends/wgpu/runtime.js` — WGPURuntime: probe, software-adapter rejection (Amni-Haven pattern), buffer pool, pipeline cache, dispatch helper, readback, GPUPipelineError detection
- `backends/wgpu/kernels/` — 13 kernels (156/156 WGSL template tests pass): copy, get-rows, rmsnorm, swiglu, softmax, rope-neox, matmul-fp16, matmul-q4_k_m, matmul-q8_0, matmul-q5_0, attention-masked, ssm-conv1d, ssm-scan
- `backends/wgpu/tokenizer.js` — BPE with GPT-2 byte fallback, special-token aware (12/12 pass on real Qwen vocab)
- `backends/wgpu/sampler.js` — greedy / temperature / top-k / top-p / repetition penalty
- `backends/wgpu/kv-cache.js` + `ssm-state.js` — per-layer GPU buffer slabs
- `backends/wgpu/arch/qwen.js` — Qwen2/Qwen3/Llama/Mistral pure-transformer forward pass
- `backends/wgpu/arch/qwen35.js` — hybrid SSM+attention scaffold (SSM dispatch stub)
- `backends/wgpu/engine.js` — orchestration: probe → load GGUF → arch builder → tokenizer → forward pass → sample → decode
- `amni-llm.js` — backend selector: probes WGPU, peeks GGUF arch, tries wgpu engine, auto-falls back to wllama-CPU on `GPUPipelineError|webgpu-unavailable|shader-compile-failed`. `forceBackend: 'gpu'|'cpu'|'auto'` option.
- `tests/test-gguf.mjs`, `test-kernels.mjs`, `test-tokenizer.mjs`, `test-runtime.html`, `gguf-smoke.html` — Node + browser test runners

**Known unverified:**
- Numerical correctness of every kernel (no GPU run yet)
- End-to-end forward pass on a real model (needs browser test)
- Qwen3.5 SSM layer dispatch (`_dispatchSSMLayer` throws — needs tensor-name audit for `ssm_in_proj`/`ssm_x_proj`/`ssm_dt_proj`/`ssm_a_log`/`ssm_d`/`ssm_out_proj` patterns)

**Public API unchanged:** `createEngine`, `AmniLLMEngine`, `chatCompletions.create`, `chatStream` all preserved. Existing callers (Amni-Scient demo) work without modification — they just get GPU acceleration automatically when WebGPU is available.

## v0.1.0 — initial release

- WebLLM-compatible API surface (`createEngine`, `chatCompletions.create`, `chatStream`)
- 4 SOTA defaults (Qwen3.5 0.8B / 4B / 9B, Qwen2.5 Math 7B)
- Built-in HuggingFace search (`huggingface.co/api/models?library=gguf`)
- One-click install of any HF GGUF; localStorage-backed registry
- Arbitrary HF URL loading
- Local `.gguf` file loading
- Cross-origin isolation service worker for `SharedArrayBuffer` / multi-thread WASM
- Backend: wllama 2.4.0 (CPU SIMD WASM, llama.cpp port, MIT)
- Demo UI at `/lib/amni-llm/index.html`
