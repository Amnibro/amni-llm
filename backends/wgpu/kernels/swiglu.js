export const swigluKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> gate: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> up: array<${useF16?'f16':'f32'}>;
@group(0) @binding(3) var<storage, read_write> out: array<${useF16?'f16':'f32'}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
if (i >= params.x) { return; }
let g = f32(gate[i]);
let u = f32(up[i]);
let silu_g = g / (1.0 + exp(-g));
let v = silu_g * u;
out[i] = ${useF16?'f16(v)':'v'};
}`;
export const swigluKernelKey=opts=>'swiglu|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
