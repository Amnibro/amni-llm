# Amni-LLM

Universal in-browser GGUF runtime. Load **any** GGUF model from HuggingFace (or a local file) and chat with it client-side, with no pre-compilation step gated by a registry maintainer.

**Why this exists.** [WebLLM (MLC)](https://github.com/mlc-ai/web-llm) requires every model to be specifically TVM-compiled to WebGPU shaders and published to its registry before it can run in the browser. New base models often wait weeks or months for that step. Amni-LLM uses GGUF, which has near-universal community coverage on HuggingFace within days of any release.

**Tradeoff.** llama.cpp WASM is currently CPU-SIMD-only (WebGPU support is experimental), so per-token throughput is ~3–8 tokens/sec on a 7B Q4 model on a modern desktop CPU, vs ~15–30 t/s for the same model on MLC's WebGPU stack. For interactive chat at small models, the gap is barely noticeable; for production workloads needing high throughput, MLC remains faster where it has the model.

**Honest stack.** Amni-LLM is a thin clean library on top of [wllama](https://github.com/ngxson/wllama) (MIT-licensed llama.cpp WASM port). We provide:

- Clean WebLLM-compatible API surface (`createEngine`, `chatCompletions.create`, `chatStream`)
- 4 SOTA defaults (Qwen3 0.6B, Qwen3 4B, DeepSeek-R1 Distill 7B, Qwen2.5 Math 7B) — small list, all current
- **Built-in HuggingFace search**: queries `huggingface.co/api/models?library=gguf` live
- **One-click install**: pick a quantization from any HF GGUF repo, save to local registry, load from the Installed tab next time
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
const e = await createEngine('Qwen3-4B-Q4_K_M');

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
| Qwen3 0.6B Q4_K_M | 480 MB | mobile | Latest Qwen, mobile-friendly |
| Qwen3 4B Q4_K_M | 2.5 GB | balanced | Sweet spot for desktop chat |
| DeepSeek-R1 Distill Qwen 7B Q4_K_M | 4.7 GB | reasoning | Strongest in-browser reasoning |
| Qwen2.5 Math 7B Q4_K_M | 4.7 GB | math | Engineering / math specialist |

Anything else: use the HF search tab. The point of Amni-LLM is that you are not limited to a curated list.

## License

Amni-LLM library code: same as the parent repo (Amni-Scient).
Compute backend (wllama / llama.cpp): MIT — see https://github.com/ngxson/wllama
