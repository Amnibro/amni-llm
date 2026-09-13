export const softmaxKernel=({useF16=true,workgroupSize=256}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> data: array<${useF16?'f16':'f32'}>;
var<workgroup> shared_max: array<f32, ${workgroupSize}>;
var<workgroup> shared_sum: array<f32, ${workgroupSize}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
let row = gid.y;
let len = params.y;
let mask_len = params.z;
let scale = bitcast<f32>(params.w);
let off = row * len;
var local_max: f32 = -3.4e38;
var i: u32 = lid.x;
loop {
if (i >= len) { break; }
let v = f32(data[off + i]) * scale;
if (i >= mask_len) { local_max = local_max; } else { local_max = max(local_max, v); }
i = i + ${workgroupSize}u;
}
shared_max[lid.x] = local_max;
workgroupBarrier();
var stride: u32 = ${workgroupSize >> 1}u;
loop {
if (stride == 0u) { break; }
if (lid.x < stride) { shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + stride]); }
workgroupBarrier();
stride = stride >> 1u;
}
let m = shared_max[0];
var local_sum: f32 = 0.0;
i = lid.x;
loop {
if (i >= len) { break; }
let masked = i >= mask_len;
let v = f32(data[off + i]) * scale;
let e = select(exp(v - m), 0.0, masked);
data[off + i] = ${useF16?'f16(e)':'e'};
local_sum = local_sum + e;
i = i + ${workgroupSize}u;
}
shared_sum[lid.x] = local_sum;
workgroupBarrier();
stride = ${workgroupSize >> 1}u;
loop {
if (stride == 0u) { break; }
if (lid.x < stride) { shared_sum[lid.x] = shared_sum[lid.x] + shared_sum[lid.x + stride]; }
workgroupBarrier();
stride = stride >> 1u;
}
let denom = shared_sum[0];
let inv = select(1.0 / denom, 0.0, denom == 0.0);
i = lid.x;
loop {
if (i >= len) { break; }
let v = f32(data[off + i]) * inv;
data[off + i] = ${useF16?'f16(v)':'v'};
i = i + ${workgroupSize}u;
}
}`;
export const softmaxKernelKey=opts=>'softmax|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||256);
