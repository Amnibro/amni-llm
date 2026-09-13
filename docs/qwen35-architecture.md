# Qwen3.5 Architecture (qwen35) — Reference

Derived clean-room from `transformers/models/qwen3_5/modular_qwen3_5.py` and `transformers/models/qwen3_next/modeling_qwen3_next.py` (HF transformers, Apache 2.0). Math reproduced; implementation original.

## Overview

Qwen3.5 is a hybrid of two layer types alternating per `full_attention_interval=4`:
- **`linear_attention`** (3 of every 4 layers): **Gated DeltaNet** — recurrent state-space layer with delta-rule update
- **`full_attention`** (every 4th layer): standard multi-head attention with **gated output** + **Q/K-Norm** + **partial RoPE**

Plus standard SwiGLU FFN and RMSNorm-with-(1+weight) blocks.

## Linear-attention layer (Gated DeltaNet)

GGUF tensors per layer:

| HF name | GGUF name | Shape (0.8B) | Quant | Role |
|---|---|---|---|---|
| `input_layernorm` | `attn_norm` | [1024] | F32 | Pre-block RMSNorm |
| `linear_attn.in_proj_qkv` | `attn_qkv` | [1024 × 6144] | Q5_K | Fused QKV projection (key_dim×2 + value_dim) |
| `linear_attn.in_proj_z` | `attn_gate` | [1024 × 2048] | Q4_K | Output gate signal |
| `linear_attn.in_proj_a` | `ssm_alpha` | [1024 × 16] | Q8_0 | Decay-rate input projection (per v-head) |
| `linear_attn.in_proj_b` | `ssm_beta` | [1024 × 16] | Q8_0 | Beta gate input projection (per v-head) |
| `linear_attn.A_log` | `ssm_a` | [16] | F32 | Log of A diagonal (per v-head) |
| `linear_attn.dt_bias` | `ssm_dt.bias` | [16] | F32 | Bias for time-step gate |
| `linear_attn.conv1d` | `ssm_conv1d` | [4 × 6144] | F32 | Depthwise causal conv on QKV |
| `linear_attn.norm` | `ssm_norm` | [128] | F32 | Pre-output RMSNorm (per state-dim) |
| `linear_attn.out_proj` | `ssm_out` | [2048 × 1024] | Q5_K | Output projection back to hidden |
| `post_attention_layernorm` | `post_attention_norm` | [1024] | F32 | Pre-FFN RMSNorm |
| `mlp.gate_proj` | `ffn_gate` | [1024 × 3584] | Q4_K | SwiGLU gate |
| `mlp.up_proj` | `ffn_up` | [1024 × 3584] | Q4_K | SwiGLU up |
| `mlp.down_proj` | `ffn_down` | [3584 × 1024] | Q6_K | FFN out |

For Qwen3.5-0.8B: `num_v_heads = num_k_heads = 16`, `key_head_dim = value_head_dim = 128`, `inner_size = 2048`, `key_dim = 2048`, `value_dim = 2048`, `conv_kernel = 4`, `state_size = 128`.

### Forward pass (linear-attention layer)

```
residual = x
h = rmsnorm_1plus(x, attn_norm.weight)              # 1+weight variant

mixed_qkv = h @ attn_qkv.weight                     # [seq, 6144]
z = h @ attn_gate.weight                            # [seq, 2048]
b = h @ ssm_beta.weight                             # [seq, 16]
a = h @ ssm_alpha.weight                            # [seq, 16]

mixed_qkv = silu(causal_conv1d(mixed_qkv, ssm_conv1d.weight))
                                                    # depthwise conv on 6144 channels, kernel=4

q, k, v = split(mixed_qkv, [2048, 2048, 2048])     # all → [seq, num_heads * head_dim]

beta = sigmoid(b)                                   # [seq, 16]
g = -exp(ssm_a) * softplus(a + ssm_dt.bias)        # [seq, 16] — decay rate per head, per token

# L2-normalize Q and K (inside delta rule)
# Per-token recurrent gated delta rule, per v-head h:
#   S_h = (1 - beta_h * <k_h, k_h>) * exp(g_h) * S_h_prev + beta_h * (v_h ⊗ k_h)
#   y_h = q_h @ S_h
# With use_qk_l2norm_in_kernel=True, <k_h, k_h> = 1, simplifying to:
#   S_h = (1 - beta_h) * exp(g_h) * S_h_prev + beta_h * (v_h ⊗ k_h)

y = gated_delta_recurrent(q, k, v, g, beta, recurrent_state)
                                                    # [seq, 2048]
y = rmsnorm_gated(y, ssm_norm.weight, z)            # gate via silu(z)
out = y @ ssm_out.weight                            # [seq, 1024]

x = residual + out

residual = x
h = rmsnorm_1plus(x, post_attention_norm.weight)
gate = h @ ffn_gate.weight                          # [seq, 3584]
up = h @ ffn_up.weight                              # [seq, 3584]
ffn = swiglu(gate, up) @ ffn_down.weight            # [seq, 1024]
x = residual + ffn
```

## Full-attention layer

GGUF tensors per layer:

| HF name | GGUF name | Shape (0.8B) | Role |
|---|---|---|---|
| `self_attn.q_proj` | `attn_q` | [1024 × 4096] | Q + gate (split 50/50: 2048 Q, 2048 gate) |
| `self_attn.k_proj` | `attn_k` | [1024 × 512] | K (head_count_kv × key_length = 2 × 256) |
| `self_attn.v_proj` | `attn_v` | [1024 × 512] | V (head_count_kv × value_length = 2 × 256) |
| `self_attn.o_proj` | `attn_output` | [2048 × 1024] | Output projection |
| `self_attn.q_norm` | `attn_q_norm` | [256] | Per-head RMSNorm-1plus on Q (head_dim=256) |
| `self_attn.k_norm` | `attn_k_norm` | [256] | Per-head RMSNorm-1plus on K |

Note: `attn_q.weight` outputs `head_count × head_dim × 2 = 8 × 256 × 2 = 4096`. The output is split: first half is Q, second half is gate signal. (Reference: `Qwen3NextAttention.forward` in modeling_qwen3_next.py line 296-299.)

### Forward pass (full-attention layer)

```
residual = x
h = rmsnorm_1plus(x, attn_norm.weight)

q_and_gate = h @ attn_q.weight                      # [seq, 4096]
q, gate = split(q_and_gate, [2048, 2048])           # half Q, half gate
q = q.reshape(seq, n_heads=8, head_dim=256)
q = rmsnorm_1plus(q, attn_q_norm.weight)            # over head_dim, broadcast

k = h @ attn_k.weight                               # [seq, 512]
k = k.reshape(seq, n_heads_kv=2, head_dim=256)
k = rmsnorm_1plus(k, attn_k_norm.weight)

v = h @ attn_v.weight                               # [seq, 512]
v = v.reshape(seq, n_heads_kv=2, head_dim=256)

q, k = partial_rope(q, k, positions, rotary_dim=64) # only first 64 dims rotated

# Standard scaled dot-product attention with GQA (broadcast K,V over Q heads)
attn = masked_attention(q, k, v, scale=1/sqrt(256))  # [seq, 8, 256]
attn = attn.reshape(seq, 2048)
attn = attn * sigmoid(gate)                          # gated output

out = attn @ attn_output.weight                      # [seq, 1024]
x = residual + out

# FFN (same as linear layer)
residual = x
h = rmsnorm_1plus(x, post_attention_norm.weight)
gate = h @ ffn_gate.weight
up = h @ ffn_up.weight
x = residual + (swiglu(gate, up) @ ffn_down.weight)
```

## RoPE (partial)

`partial_rotary_factor = 0.25`. With `head_dim = 256`: `rotary_dim = 64`.

Standard RoPE math, but only on first 64 dims of each head; remaining 192 dims pass through unchanged.

```
inv_freq[j] = 1 / base^(2j / rotary_dim)   for j in 0..rotary_dim/2 - 1
freqs = pos * inv_freq                      # [rotary_dim/2]
emb = cat(freqs, freqs)                     # [rotary_dim]
cos, sin = emb.cos(), emb.sin()

For dims 0..rotary_dim-1 of each head:
  q_rot_first = q[0..rotary_dim/2]
  q_rot_second = q[rotary_dim/2..rotary_dim]
  q_new[0..rotary_dim/2] = q_rot_first * cos[0..rotary_dim/2] - q_rot_second * sin[0..rotary_dim/2]
  q_new[rotary_dim/2..rotary_dim] = q_rot_second * cos[rotary_dim/2..rotary_dim] + q_rot_first * sin[rotary_dim/2..rotary_dim]
```

`base = rope.freq_base = 10000000` (much higher than standard 10000 — supports 262144 context).

This is the GLM-style apply (not interleaved).

## RMSNorm variants

**`rmsnorm_1plus`** (Qwen3NextRMSNorm): `output = norm(x) * (1 + weight)` — note the +1.

**`rmsnorm_gated`** (Qwen3NextRMSNormGated): `output = (norm(x) * weight) * silu(gate)` — standard weight, then gate by silu(z).

## Recurrent state (linear-attention only)

Per linear-attention layer:
- **conv_state**: `[hidden_size_qkv = 6144, conv_kernel = 4]` — rolling window for causal conv1d
- **recurrent_state**: `[num_v_heads = 16, key_head_dim = 128, value_head_dim = 128]` — gated delta rule state matrix per v-head

Both are persistent across decode steps. Reset on new prompt.

## Tokenizer

`tokenizer.ggml.pre = "qwen35"` — same regex as qwen2 in practice. Vocab size 248320. EOS = 248046, PAD = 248055. BOS not used by default (chat template controls turn boundaries).

## Notes on prefill vs decode

For **decode (T=1)**: per-token recurrent gated delta rule. Sequential.
For **prefill (T>1)**: ideally chunk-wise gated delta rule (block-parallel). Phase-1 implementation uses per-token recurrent for both — slower prefill but simpler / correct. Optimization later.
