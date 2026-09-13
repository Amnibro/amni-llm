export const gatedDeltaRecurrentKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> q: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> k: array<${useF16?'f16':'f32'}>;
@group(0) @binding(3) var<storage, read> v: array<${useF16?'f16':'f32'}>;
@group(0) @binding(4) var<storage, read> g: array<${useF16?'f16':'f32'}>;
@group(0) @binding(5) var<storage, read> beta: array<${useF16?'f16':'f32'}>;
@group(0) @binding(6) var<storage, read_write> recurrent_state: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<${useF16?'f16':'f32'}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let head = gid.x;
let n_v_heads = params.x;
let key_head_dim = params.y;
let value_head_dim = params.z;
let n_tokens = params.w;
if (head >= n_v_heads) { return; }
let state_base = head * key_head_dim * value_head_dim;
let qkv_head_base = head;
for (var t: u32 = 0u; t < n_tokens; t = t + 1u) {
let q_base = t * n_v_heads * key_head_dim + head * key_head_dim;
let v_base = t * n_v_heads * value_head_dim + head * value_head_dim;
var q_norm_sq: f32 = 0.0;
var k_norm_sq: f32 = 0.0;
for (var i: u32 = 0u; i < key_head_dim; i = i + 1u) {
let qi = f32(q[q_base + i]);
let ki = f32(k[q_base + i]);
q_norm_sq = q_norm_sq + qi * qi;
k_norm_sq = k_norm_sq + ki * ki;
}
let q_inv = inverseSqrt(q_norm_sq + 1e-6);
let k_inv = inverseSqrt(k_norm_sq + 1e-6);
let g_t = f32(g[t * n_v_heads + head]);
let beta_t = f32(beta[t * n_v_heads + head]);
let beta_sig = 1.0 / (1.0 + exp(-beta_t));
let decay = exp(g_t);
let prefactor = (1.0 - beta_sig) * decay;
for (var ki_idx: u32 = 0u; ki_idx < key_head_dim; ki_idx = ki_idx + 1u) {
let k_norm = f32(k[q_base + ki_idx]) * k_inv;
for (var vi_idx: u32 = 0u; vi_idx < value_head_dim; vi_idx = vi_idx + 1u) {
let s_idx = state_base + ki_idx * value_head_dim + vi_idx;
let v_val = f32(v[v_base + vi_idx]);
recurrent_state[s_idx] = prefactor * recurrent_state[s_idx] + beta_sig * k_norm * v_val;
}
}
for (var vi_idx: u32 = 0u; vi_idx < value_head_dim; vi_idx = vi_idx + 1u) {
var y: f32 = 0.0;
for (var ki_idx: u32 = 0u; ki_idx < key_head_dim; ki_idx = ki_idx + 1u) {
let q_norm = f32(q[q_base + ki_idx]) * q_inv;
let s_idx = state_base + ki_idx * value_head_dim + vi_idx;
y = y + q_norm * recurrent_state[s_idx];
}
output[v_base + vi_idx] = ${useF16?'f16(y)':'y'};
}
}
}`;
export const gatedDeltaRecurrentKernelKey=opts=>'gated-delta-recurrent|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
