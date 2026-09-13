export const attentionMaskedKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> q: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> k_cache: array<${useF16?'f16':'f32'}>;
@group(0) @binding(3) var<storage, read> v_cache: array<${useF16?'f16':'f32'}>;
@group(0) @binding(4) var<storage, read_write> out: array<${useF16?'f16':'f32'}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let head = gid.y;
let dim_idx = gid.x;
let n_heads_q = params.x;
let n_heads_kv = params.y;
let head_dim = params.z;
let kv_len = params.w;
if (head >= n_heads_q || dim_idx >= head_dim) { return; }
let kv_head = head * n_heads_kv / n_heads_q;
let q_base = head * head_dim;
let scale = inverseSqrt(f32(head_dim));
var max_score: f32 = -3.4e38;
for (var t: u32 = 0u; t < kv_len; t = t + 1u) {
let k_base = t * n_heads_kv * head_dim + kv_head * head_dim;
var dot: f32 = 0.0;
for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
dot = dot + f32(q[q_base + d]) * f32(k_cache[k_base + d]);
}
let score = dot * scale;
max_score = max(max_score, score);
}
var sum_exp: f32 = 0.0;
var weighted: f32 = 0.0;
for (var t: u32 = 0u; t < kv_len; t = t + 1u) {
let k_base = t * n_heads_kv * head_dim + kv_head * head_dim;
let v_base = k_base;
var dot: f32 = 0.0;
for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
dot = dot + f32(q[q_base + d]) * f32(k_cache[k_base + d]);
}
let p = exp(dot * scale - max_score);
sum_exp = sum_exp + p;
weighted = weighted + p * f32(v_cache[v_base + dim_idx]);
}
let result = weighted / sum_exp;
out[q_base + dim_idx] = ${useF16?'f16(result)':'result'};
}`;
export const attentionMaskedKernelKey=opts=>'attention-masked|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
