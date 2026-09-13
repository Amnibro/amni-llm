# Amni-LLM Architecture Map

Last updated: 2026-04-27

## Current state — v0.1.0 (CPU-only via wllama)

**Backend:** wllama 2.4.0 ESM build loaded from jsDelivr CDN. wllama is a llama.cpp WASM port (MIT). Runs CPU SIMD only. Per-token throughput: 3-8 t/s on a 7B Q4_K_M model on a modern desktop CPU. WebGPU is not used.

**Files:**

- `amni-llm.js` — `AmniLLMEngine` class, `createEngine()` factory, `COMPUTE_BACKEND` constant, error decoration for OOM hints, COI status helper.
- `registry.js` — `DEFAULT_MODELS` (4 SOTA picks), localStorage-backed installed-models store (`getInstalled` / `saveInstalled` / `removeInstalled`), HuggingFace search (`hfSearchModels`), HF GGUF file listing (`hfListGgufFiles`), URL resolver (`hfResolve`), quant detector.
- `index.html` — demo UI with five tabs: SOTA defaults, Installed, HF search, Custom URL, Local file. Chat panel with streaming, progress bar, COI status banner.
- `coi-register.js`, `coi-sw.js` — service worker that injects COOP/COEP headers so the page becomes cross-origin-isolated (enables `SharedArrayBuffer` and multi-thread WASM).
- `package.json` — `@wllama/wllama` ^2.4.0 declared as optional peer dep (loaded from CDN at runtime, not bundled).
- `LICENSE`, `README.md`, `.gitignore`.

**Public API surface (WebLLM-compatible):**

```
createEngine(modelSpec, opts) -> AmniLLMEngine
engine.chatCompletions.create({ messages, temperature, max_tokens, top_p, stop, stream }) -> Response | AsyncIterator
engine.chatStream(messages, opts) -> AsyncIterator<{ delta, raw }>
engine.unload() -> Promise<void>
engine.isReady() -> boolean
engine.getModelInfo() -> { id, name, hfUrl, sizeMB, ... }
COMPUTE_BACKEND -> { engine, version, kernel, license, source }
```

**Model loading paths:**

- Registry id (`'Qwen3.5-0.8B-Q4_K_M'`, etc.) — looked up in `DEFAULT_MODELS` or installed store.
- Direct HF URL (`{ url: '...' }`) — resolved verbatim.
- Local file (`{ file: File }`) — loaded into WASM heap, no upload.

## Planned state — v0.2.0 (WGSL universal GPU backend)

**Goal:** any GGUF that wllama loads today *also* runs on WebGPU at MLC-comparable speed (~15-30 t/s 7B Q4) with **no per-model compile, no hosted bundles, no registry maintainer in the path**. The point of difference vs. MLC is universal coverage; the point of difference vs. wllama-WASM is GPU acceleration.

**New backend layer:** `backends/wgpu/`

```
backends/
├── cpu-wllama.js        existing wllama path, kept as automatic fallback
└── wgpu/
    ├── runtime.js        WGPU device, adapter, buffer pool, command-queue dispatch loop, prefill/decode passes
    ├── gguf.js           GGUF parser: header, tensor table, ranged fetch (no full-file download until tensor is read)
    ├── tokenizer.js      BPE tokenizer; ports wllama's tokenizer config from GGUF metadata
    ├── sampler.js        temperature, top-k, top-p, repetition penalty, greedy
    ├── kv-cache.js       paged growable KV cache, fp16 storage, per-layer slabs
    ├── arch/
    │   ├── llama.js      Llama 2/3/4, Mistral, DeepSeek-distill (shared graph shape)
    │   ├── qwen.js       Qwen 2/3/3.5 (RoPE-NTK, attention sink quirks)
    │   └── gemma.js      Gemma 2/3 (logit softcap, sliding-window mask)
    └── kernels/
        matmul-q4_k_m.wgsl    most common GGUF quant — first kernel
        matmul-q8_0.wgsl
        matmul-q5_k.wgsl
        matmul-q6_k.wgsl
        matmul-fp16.wgsl      activations, embeddings, output projection
        attention-masked.wgsl masked QK^T → softmax → V
        softmax.wgsl          scaled, with mask
        rope-norm.wgsl        standard RoPE
        rope-neox.wgsl        NeoX-style RoPE (Qwen, GPT-NeoX)
        rmsnorm.wgsl          with epsilon
        silu.wgsl
        gelu.wgsl
        swiglu.wgsl
        copy.wgsl             dtype conversions (fp16 ↔ fp32, etc.)
        get-rows.wgsl         embedding lookup
        argmax.wgsl           greedy decode tail
```

**Engine selector (`amni-llm.js`):**

1. On `createEngine()`: probe `navigator.gpu` → `requestAdapter()` → `requestDevice()`.
2. Read GGUF header to identify architecture + quant.
3. If GPU available AND arch in supported set AND quant in supported set → `wgpu` backend.
4. Else → `cpu-wllama` backend (current behavior preserved).
5. Both backends expose identical `AmniLLMEngine` interface; caller code unaffected.

**`COMPUTE_BACKEND` becomes dynamic** — reflects which path the active engine took. Useful for the demo UI to show "Running on WebGPU (RTX 4070)" vs "Running on CPU SIMD (4 threads)".

## Phasing

- **v0.2.0-alpha** — GGUF parser, WGPU runtime skeleton, Q4_K_M matmul, single arch (Qwen3), greedy decode, no streaming. *Goal: prove first WGPU-generated token.*
- **v0.2.0-beta** — full attention path (RoPE + masked attn + softmax + KV cache), streaming, full sampler, Qwen3 + Llama archs. *Goal: SOTA defaults run end-to-end on GPU.*
- **v0.2.0** — auto GPU/CPU fallback, error decoration for adapter failures, demo UI shows backend, all 4 SOTA defaults work on both paths. *Goal: shippable.*
- **v0.3.0** — more archs (Gemma, Mistral explicit, DeepSeek explicit, Phi), more quants (Q5_K, Q6_K, IQ2/IQ3 modern small quants).
- **v0.4.0** — hybrid arch families (Mamba, RWKV, hybrid SSMs); opt-in per-model specialization compile pass for top-10 download leaders (the "v2 nice-to-have" ungated-MLC layer on top of universal kernels).

## Decisions (locked 2026-04-27)

1. **Kernel derivation: clean-room from GGML spec text.** No copying llama.cpp Vulkan kernel source. We read GGML block layouts (Q4_K_M = 256 weights / 8 sub-blocks / fp16 d+dmin scales / 6-bit q-scales / 4-bit weights = 144 bytes) and reimplement the dequant+matmul math in WGSL ourselves. Slower to write, cleaner licensing, full understanding of what we ship.
2. **Precision: f16 primary, f32 fallback.** Every kernel ships in two WGSL variants — `enable f16` for adapters that report `shader-f16` (most desktop), `f32` for those that don't (Android Chrome on older hardware, software adapters, etc.). Pipeline cache key includes the variant. f32 path is ~2× slower and uses ~2× memory but works everywhere WebGPU does.
3. **GGUF version coverage: all of V1, V2, V3.** Parser branches on the version field. V1 uses `u32` count fields and string lengths; V2/V3 use `u64`. V3 added new dtype codes (IQ-quants, BF16). Universal coverage is the whole reason this project exists — clamping to V3-only would re-introduce a gate.

## Mobile profile (reference: Amni-Haven `local_ai.html`)

Amni-Haven shipped WebLLM in an Android WebView and has battle-tested the failure surface. We adopt its patterns:

- **Adapter probing:** `navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })`, then read `adapter.info` (vendor / architecture / device / description). Reject software adapters (`swiftshader|llvmpipe|lavapipe|software|microsoft basic render`) — they're slower than wllama-CPU and waste compile time on shaders that never run fast.
- **Failure catch:** wrap engine init in a try/catch matching `/GPUPipelineError|CreateComputePipelines|VK_ERROR|Vulkan/i`. On match, automatically fall back to wllama-CPU and log "GPU rejected the compute shader, switched to CPU." Same routing-around-the-driver strategy that fixed Amni-Haven's Adreno+stock-WebView devices.
- **Stall detection:** if a single shader compile exceeds 3 minutes (Adreno cold-compile signature), surface a "shader compile stuck — fall back to CPU?" prompt rather than wedging the page.
- **Explicit override:** demo UI exposes a "CPU mode" button so users on known-broken devices skip the GPU attempt entirely.

## Reused vs. owned

- **Reused (compute):** none. wllama remains as fallback but the GPU path is fully ours.
- **Reused (loading):** HF URL conventions, GGUF format spec, GGUF tokenizer metadata layout.
- **Reused (mobile patterns):** adapter probing + auto-fallback strategy from Amni-Haven `local_ai.html` (proven on Android WebView).
- **Owned (new):** all WGSL kernels (clean-room from GGML spec), dispatch loop, KV cache, arch graph builders, GGUF parser (JS, V1+V2+V3), tokenizer wrapper, sampler, engine selector.

## Out of scope for v0.2.0

- Multi-GPU.
- Speculative decoding, draft models.
- Per-model compile pipeline (deferred to v0.4 as opt-in fast path).
- Vision/multimodal heads.
- LoRA hot-swap.
