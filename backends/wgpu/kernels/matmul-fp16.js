export const matmulFp16Kernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> a: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> b: array<${useF16?'f16':'f32'}>;
@group(0) @binding(3) var<storage, read_write> c: array<${useF16?'f16':'f32'}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let m = gid.y;
let n = gid.x;
let M = params.x;
let N = params.y;
let K = params.z;
if (m >= M || n >= N) { return; }
var sum: f32 = 0.0;
for (var k: u32 = 0u; k < K; k = k + 1u) {
sum = sum + f32(a[m * K + k]) * f32(b[n * K + k]);
}
c[m * N + n] = ${useF16?'f16(sum)':'sum'};
}`;
export const matmulFp16KernelKey=opts=>'matmul-fp16|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
