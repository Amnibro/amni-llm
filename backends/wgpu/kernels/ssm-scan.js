export const ssmScanKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> x: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> dt: array<${useF16?'f16':'f32'}>;
@group(0) @binding(3) var<storage, read> a_log: array<f32>;
@group(0) @binding(4) var<storage, read> b_proj: array<${useF16?'f16':'f32'}>;
@group(0) @binding(5) var<storage, read> c_proj: array<${useF16?'f16':'f32'}>;
@group(0) @binding(6) var<storage, read> d_skip: array<${useF16?'f16':'f32'}>;
@group(0) @binding(7) var<storage, read_write> ssm_state: array<f32>;
@group(0) @binding(8) var<storage, read_write> output: array<${useF16?'f16':'f32'}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let ch = gid.x;
let token = gid.y;
let n_channels = params.x;
let state_dim = params.y;
let n_tokens = params.z;
if (ch >= n_channels || token >= n_tokens) { return; }
let dt_softplus_raw = f32(dt[token * n_channels + ch]);
let dt_v = log(1.0 + exp(dt_softplus_raw));
let x_v = f32(x[token * n_channels + ch]);
let d_v = f32(d_skip[ch]);
var y: f32 = 0.0;
for (var s: u32 = 0u; s < state_dim; s = s + 1u) {
let a = -exp(a_log[ch * state_dim + s]);
let da = exp(dt_v * a);
let b = f32(b_proj[token * state_dim + s]);
let c = f32(c_proj[token * state_dim + s]);
let state_idx = ch * state_dim + s;
let prev = ssm_state[state_idx];
let new_state = prev * da + x_v * b * dt_v;
ssm_state[state_idx] = new_state;
y = y + new_state * c;
}
let out_v = y + x_v * d_v;
output[token * n_channels + ch] = ${useF16?'f16(out_v)':'out_v'};
}`;
export const ssmScanKernelKey=opts=>'ssm-scan|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
