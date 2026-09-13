# Checklist — Amni-LLM WGSL Universal GPU Backend (v0.2.0)

Source-of-truth for the WGSL universal port. Sequential. Check items off as completed. Do not claim the task done until all items are checked AND the user confirms the golden-path test.

## Pre-implementation (planning)

- [x] Scan/draft `architecture_map.md` (Amni-LLM had none — created)
- [x] Convene guardian council (`docs/guardian_councils/guardian_council_amni_llm_wgsl_universal.md`)
- [x] Draft this checklist
- [ ] **Phase 1 sign-off from Anthony before any kernel work begins**
- [ ] Backup originals: `amni-llm.js`, `registry.js`, `index.html` → `backups/<file>.v0.1.0.bak`

## Phase 1 — first WGPU-generated token (v0.2.0-alpha)

### GGUF parser (`backends/wgpu/gguf.js`)

- [ ] Magic check (`GGUF` 0x46554747)
- [ ] Version branch: V1 (u32 counts/strings), V2 (u64 counts/strings), V3 (current — same as V2 plus extended dtype enum)
- [ ] Metadata KV section (read all keys: `general.architecture`, `general.alignment`, `general.name`, `tokenizer.*`, `<arch>.*` rope/dim/layer/head fields). Value-type tags 0-12 (u8/i8/u16/i16/u32/i32/f32/bool/string/array/u64/i64/f64).
- [ ] Tensor table (name, dtype, shape, offset). Dim sizes are u32 in V1, u64 in V2/V3.
- [ ] Ranged-fetch helper: lazy `_extend()` grows the buffer in 4 MB chunks via HTTP `Range:` for URL sources, `File.slice()` for local files. Initial fetch covers metadata; tensor reads issue dedicated range requests.
- [ ] `readTensorBytes(name)` returns the raw block-quantized byte slab for a tensor.
- [ ] Quant code mapping: `GGML_TYPE_Q4_K`, `Q8_0`, `Q5_K`, `Q6_K`, `F16`, `F32`, `BF16` → JS enum + block-size table
- [ ] Tokenizer KV extraction: vocab, merges, BOS/EOS/PAD ids, chat template
- [ ] Unit test: parse Qwen3.5-0.8B-Q4_K_M.gguf header, assert version detected, tensor count matches metadata, arch === `qwen3`, quant hint Q4_K
- [ ] Negative test: V1 fixture parses (use a small TinyStories-style V1 GGUF if available); malformed magic raises clear error

### WGPU runtime (`backends/wgpu/runtime.js`)

- [ ] `initDevice()` — adapter + device with required limits (`maxBufferSize`, `maxStorageBufferBindingSize`, `maxComputeWorkgroupStorageSize` — query and report caps)
- [ ] Buffer pool: allocator with size classes, reuse across decode steps to avoid fragmentation
- [ ] Pipeline cache: WGSL → `GPUComputePipeline` keyed by `(kernel, workgroup_size, dtype variant)`
- [ ] Dispatch helper: `dispatch(pipeline, bindings, workgroups)` that records into a single `GPUCommandEncoder` per forward step
- [ ] Forward-step orchestrator: prefill (full sequence) vs. decode (one token), distinct dispatch shapes
- [ ] Mapped-staging readback for final logits → CPU sampler

### Q4_K_M matmul kernel (`kernels/matmul-q4_k_m.wgsl` + `.f32.wgsl`)

- [ ] Read GGML Q4_K_M block layout from spec text (NOT from llama.cpp source): 256 weights per superblock = 8 × 32-weight sub-blocks, fp16 d (super-scale), fp16 dmin (super-min), 12 bytes of 6-bit q-scales/q-mins, 128 bytes of 4-bit packed weights. Total 144 bytes / superblock.
- [ ] WGSL parser for the block struct (`uint32` storage, bit-extraction shifts/masks)
- [ ] Dequantize-and-multiply fused kernel (avoid materializing fp16 weight matrix — compute weight only when used)
- [ ] **Two variants compiled:** `f16` (`enable f16`) and `f32` fallback. Pipeline cache keys on adapter f16-support flag.
- [ ] Workgroup: 64 threads, one block per workgroup (subject to perf measurement)
- [ ] Correctness test: ingest a known Q4_K_M tensor + input vector, run kernel, compare output element-wise to wllama CPU reference. Spec: per-tensor max-relative-error < 1e-3 (per Sentinel's revision); not bit-equiv since reduction order differs.

### Attention path kernels

- [ ] `rmsnorm.wgsl` (with epsilon param)
- [ ] `rope-neox.wgsl` (Qwen/Llama/Mistral all use NeoX-style)
- [ ] `attention-masked.wgsl` (Q × K^T scaled by 1/√d_head, causal mask, softmax, × V — fused or 3-stage to be decided by perf measurement)
- [ ] `softmax.wgsl` (scaled, mask-aware, numerically stable — subtract max)
- [ ] `silu.wgsl`, `swiglu.wgsl` (Qwen's MLP uses SwiGLU)
- [ ] `get-rows.wgsl` (embedding lookup, fp16 output)
- [ ] `copy.wgsl` (fp32 ↔ fp16 conversion for boundary tensors)
- [ ] Per-kernel correctness test against wllama CPU reference (load same tensor, dispatch kernel, compare)

### Qwen3 arch builder (`arch/qwen.js`)

- [ ] Read `qwen3.*` GGUF metadata (block_count, embedding_length, attention.head_count, attention.head_count_kv, rope.freq_base, rope.dimension_count, feed_forward_length)
- [ ] Build forward graph: emb_lookup → for each layer: rmsnorm → qkv proj (matmul) → rope → kv-cache write → masked attn → out proj → residual → rmsnorm → swiglu MLP → residual; final rmsnorm → lm_head matmul
- [ ] Tied/untied embeddings (Qwen3 0.6B ties; larger sizes don't — read from metadata)
- [ ] GQA: head_count_kv < head_count → broadcast K/V over query heads in attention kernel

### Tokenizer (`tokenizer.js`)

- [ ] Port BPE tokenizer from GGUF tokenizer metadata (vocab + merges)
- [ ] Apply Qwen3 chat template (`<|im_start|>role\n...<|im_end|>`)
- [ ] Encode user/system messages → token ids
- [ ] Decode token id → text fragment (handles partial UTF-8 byte sequences across tokens)
- [ ] Test: encode/decode round-trip on known string

### Sampler (`sampler.js`)

- [ ] Greedy (argmax) — simplest path, used in alpha
- [ ] Temperature scale
- [ ] Top-k (sort + truncate)
- [ ] Top-p (cumulative dist truncation)
- [ ] Repetition penalty (per recent-token frequency)

### KV cache (`kv-cache.js`)

- [ ] Per-layer K and V slabs in GPU memory, fp16
- [ ] Growable: start at `n_ctx_seed=512`, grow by 2× to `n_ctx_max` (read from GGUF or capped at 4096 for alpha)
- [ ] Position write helper: `writeKV(layer, pos, k_tensor, v_tensor)`
- [ ] Position read helper: `readKV(layer, pos_range)` returns slabs sliced for current decode step

### Engine wrapper (`amni-llm.js` modification)

- [ ] GPU detection helper: `probeWebGPU()` → `{ available, adapter_info, limits, hasF16, isSoftware }` or `{ available: false, reason }`. Adopts Amni-Haven's adapter-probing pattern: rejects `swiftshader|llvmpipe|lavapipe|software|microsoft basic render` adapters as if they were missing.
- [ ] Backend selector inside `AmniLLMEngine.load()`: read GGUF arch + quant, decide wgpu vs. wllama, log decision. Order: (1) probe WebGPU, (2) check arch + quant supported, (3) try wgpu init, (4) on `/GPUPipelineError|CreateComputePipelines|VK_ERROR|Vulkan/i` mid-init → automatic wllama fallback with one-line user-visible status.
- [ ] Stall watchdog: if a single shader compile exceeds 180s, surface "shader compile stuck — switching to CPU" and fall back.
- [ ] Explicit `forceBackend: 'cpu'|'gpu'` option for users on known-good or known-broken devices.
- [ ] `COMPUTE_BACKEND` made dynamic — reflects active backend on `getModelInfo()` (`{ engine: 'amni-wgpu' | 'wllama-cpu', kernel, version, adapter }`).
- [ ] Same public API surface (no breaking changes for existing callers — confirmed by grep against `amni-scient-site/`)

### Phase 1 golden-path test

- [ ] Load Qwen3.5-0.8B-Q4_K_M on a WebGPU-capable browser
- [ ] Generate 64 tokens from prompt "Hello, "
- [ ] Verify: tokens match wllama-CPU output bit-for-bit on same seed (greedy is deterministic, so this is a hard equality check)
- [ ] Verify: t/s exceeds CPU baseline by ≥3× on the same machine
- [ ] Verify: no console errors, no GPU hangs, no OOM
- [ ] Run pylance/eslint (whichever is configured) — zero new errors
- [ ] Test on Chrome desktop, Edge, and at least one mobile browser if WebGPU is available there

### Documentation + cleanup

- [ ] Update `architecture_map.md` to mark v0.2.0-alpha shipped
- [ ] Append entry to `changelog.md` (date, version, summary)
- [ ] Update `README.md` — note WebGPU path is live, mention auto-fallback
- [ ] Bump `package.json` version

## Anthony's confirmation gate

- [ ] Anthony confirms phase 1 works end-to-end before this checklist is closed and v0.2.0-alpha is released

## Phase 2+ (deferred — not in this checklist version)

Will spawn `checklist_amni_llm_wgsl_universal_v2.md` for v0.2.0-beta (streaming, multi-arch, full sampler) once phase 1 is signed off.
