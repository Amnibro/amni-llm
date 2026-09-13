export const rmsnorm1plusKernel=({useF16=true,workgroupSize=256}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> input: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<${useF16?'f16':'f32'}>;
var<workgroup> shared_sum: array<f32, ${workgroupSize}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
let row = gid.y;
let dim = params.y;
let eps = bitcast<f32>(params.z);
let row_offset = row * dim;
var partial: f32 = 0.0;
var i: u32 = lid.x;
loop {
if (i >= dim) { break; }
let v = f32(input[row_offset + i]);
partial = partial + v * v;
i = i + ${workgroupSize}u;
}
shared_sum[lid.x] = partial;
workgroupBarrier();
var stride: u32 = ${workgroupSize >> 1}u;
loop {
if (stride == 0u) { break; }
if (lid.x < stride) { shared_sum[lid.x] = shared_sum[lid.x] + shared_sum[lid.x + stride]; }
workgroupBarrier();
stride = stride >> 1u;
}
let mean_sq = shared_sum[0] / f32(dim);
let scale = inverseSqrt(mean_sq + eps);
i = lid.x;
loop {
if (i >= dim) { break; }
let v = f32(input[row_offset + i]) * scale * (1.0 + weight[i]);
output[row_offset + i] = ${useF16?'f16(v)':'v'};
i = i + ${workgroupSize}u;
}
}`;
export const rmsnorm1plusKernelKey=opts=>'rmsnorm-1plus|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||256);
