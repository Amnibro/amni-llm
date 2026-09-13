export const causalConv1dUpdateKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> input: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> conv_state: array<${useF16?'f16':'f32'}>;
@group(0) @binding(4) var<storage, read_write> output: array<${useF16?'f16':'f32'}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let ch = gid.x;
let n_channels = params.x;
let kernel_size = params.y;
let token_idx = params.z;
if (ch >= n_channels) { return; }
let state_base = ch * kernel_size;
var acc: f32 = 0.0;
for (var k: u32 = 0u; k + 1u < kernel_size; k = k + 1u) {
let v = f32(conv_state[state_base + k + 1u]);
conv_state[state_base + k] = ${useF16?'f16(v)':'v'};
acc = acc + v * weight[k * n_channels + ch];
}
let new_x = f32(input[token_idx * n_channels + ch]);
conv_state[state_base + kernel_size - 1u] = ${useF16?'f16(new_x)':'new_x'};
acc = acc + new_x * weight[(kernel_size - 1u) * n_channels + ch];
let silu = acc / (1.0 + exp(-acc));
output[token_idx * n_channels + ch] = ${useF16?'f16(silu)':'silu'};
}`;
export const causalConv1dUpdateKernelKey=opts=>'causal-conv1d-update|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
