export const gatedDecayKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> a: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> A_log: array<f32>;
@group(0) @binding(3) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> g: array<${useF16?'f16':'f32'}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let head = gid.x;
let token = gid.y;
let n_heads = params.x;
let n_tokens = params.y;
if (head >= n_heads || token >= n_tokens) { return; }
let a_v = f32(a[token * n_heads + head]) + dt_bias[head];
let softplus_a = log(1.0 + exp(a_v));
let neg_exp_A = -exp(A_log[head]);
let g_v = neg_exp_A * softplus_a;
g[token * n_heads + head] = ${useF16?'f16(g_v)':'g_v'};
}`;
export const gatedDecayKernelKey=opts=>'gated-decay|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
