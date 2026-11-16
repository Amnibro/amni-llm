# Amni-LLM

Universal in-browser GGUF runtime. Load **any** GGUF model from HuggingFace (or a local file) and chat with it client-side, with no pre-compilation step gated by a registry maintainer.

**Why this exists.** [WebLLM (MLC)](https://github.com/mlc-ai/web-llm) requires every model to be specifically TVM-compiled to WebGPU shaders and published to its registry before it can run in the browser. New base models often wait weeks or months for that step. Amni-LLM uses GGUF, which has near-universal community coverage on HuggingFace within days of any release.

**Tradeoff.** llama.cpp WASM is currently CPU-SIMD-only (WebGPU support is experimental), so per-token throughput is ~3–8 tokens/sec on a 7B Q4 model on a modern desktop CPU, vs ~15–30 t/s for the same model on MLC's WebGPU stack. For interactive chat at small models, the gap is barely noticeable; for production workloads needing high throughput, MLC remains faster where it has the model.

**Honest stack.** Amni-LLM is a thin clean library on top of [wllama](https://github.com/ngxson/wllama) (MIT-licensed llama.cpp WASM port). We provide:

- Clean WebLLM-compatible API surface (`createEngine`, `chatCompletions.create`, `chatStream`)
- Curated defaults (Qwen 3.5 0.8B / 4B / 9B, IBM Granite 4.2 3B; Qwen 3.8 27B listed as desktop-only)
- **Built-in HuggingFace search**: queries `huggingface.co/api/models?library=gguf` live
- **One-click install**: pick a quantization from any HF GGUF repo; the file is downloaded once into browser storage (OPFS) and loads from the Installed tab next time, online or offline
- **Offline**: a service worker (`coi-sw.js`) caches the app shell and the vendored wllama runtime, so after one online visit the page opens and chats with installed/default models with no network
- Arbitrary HuggingFace URL loading (`{ url: '...' }`)
- Local `.gguf` file loading (`{ file: ... }`)
- Status + progress callbacks
- Browser demo at `/lib/amni-llm/index.html`

The transformer math is executed by wllama; the loader, registry, model selector, browser UI, search/install, and API surface are Amni-Scient's.

## Try it

Open [`/lib/amni-llm/`](./index.html) in the browser. Pick a SOTA default and press Load. To find anything else, switch to the **HF search** tab, type a model name (e.g. `qwen 3.5`, `llama 4`, `mistral nemo`), expand the results to list GGUF files, and press Install or Load.

## API

```js
import { createEngine } from '/lib/amni-llm/amni-llm.js';

// SOTA default
const e = await createEngine('Qwen3.5-4B-Q4_K_M');

// Any HuggingFace GGUF (the killer feature)
const e = await createEngine({
  url: 'https://huggingface.co/<org>/<repo>-GGUF/resolve/main/model.Q4_K_M.gguf'
});

// Local file
const e = await createEngine({ file: fileInput.files[0] });

// Chat (WebLLM-compatible)
const r = await e.chatCompletions.create({
  messages: [{role:'user', content:'Hello'}],
  temperature: 0.4
});
console.log(r.choices[0].message.content);

// Streaming
for await (const chunk of e.chatStream(messages, {temperature:0.4})) {
  process.stdout.write(chunk.delta);
}

await e.unload();
```

Use the `resolve/main` URL form. `blob/main` will not work (HuggingFace serves an HTML page, not the binary).

## SOTA defaults (as of 2026-04)

| Model | Size | Tier | Why |
|---|---|---|---|
| Qwen 3.5 0.8B Q4_K_M | 508 MB | mobile | Newest small SOTA available as GGUF (`unsloth/Qwen3.5-0.8B-GGUF`) |
| Qwen 3.5 4B Q4_K_M | 2.6 GB | balanced | Sweet spot for desktop chat (`unsloth/Qwen3.5-4B-GGUF`) |
| Qwen 3.5 9B Q4_K_M | 5.4 GB | reasoning | High-quality tier, needs 8 GB+ free RAM (`unsloth/Qwen3.5-9B-GGUF`) |
| Granite 4.2 3B Q4_K_M | 2.2 GB | balanced | IBM's official GGUF (`ibm-granite/granite-4.2-3b-GGUF`), SHA-256 pinned |
| Qwen 3.8 27B UD-Q4_K_M | 16.6 GB | desktop only | Listed, not loadable: the WASM runtime has a 4 GB memory cap and the smallest quant is 6.2 GB. Use desktop Amni-AI / native llama.cpp |

A math/engineering specialist default is intentionally absent until Qwen 3.5 Math GGUF or an equivalent ships and is verified to load reliably. For math work today, use the HF search tab to grab any current math GGUF that fits your memory budget.

Anything else: use the HF search tab. The point of Amni-LLM is that you are not limited to a curated list.

## License

Amni-LLM library code: same as the parent repo (Amni-Scient).
Compute backend (wllama / llama.cpp): MIT — see https://github.com/ngxson/wllama

## Offline

- `coi-sw.js` (scope `/lib/amni-llm/`) adds COOP/COEP headers and keeps the app shell in the `amni-llm-shell-<ver>` cache: pages and JS network-first with cache fallback, versioned `vendor/` files and the font cache-first. It never caches `.gguf` files, `/models/` paths, cross-origin requests, Range requests or partial responses.
- Model bytes are stored once, by the library: SHA-256-pinned defaults in Cache Storage `amni-llm-v1`; installed URLs in wllama's OPFS cache. The page asks for persistent storage and shows a clear error when the quota is too small.
- The wllama "compat" build that older browsers (no JSPI / memory64, e.g. Safari) need is vendored under `vendor/wllama-compat-3.13.0/` instead of loading from jsDelivr; it is cached on first use, not precached.
