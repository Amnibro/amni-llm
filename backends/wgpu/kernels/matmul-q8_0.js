export const matmulQ80Kernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> input: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> weights: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<${useF16?'f16':'f32'}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let n = gid.x;
let row = gid.y;
let M = params.w;
let N = params.x;
let K = params.y;
if (n >= N || row >= M) { return; }
let blocks_per_row = K / 32u;
let words_per_block = 9u;
let row_word_base = n * blocks_per_row * words_per_block;
let input_row_base = row * K;
var sum: f32 = 0.0;
for (var b: u32 = 0u; b < blocks_per_row; b = b + 1u) {
let bw = row_word_base + b * words_per_block;
let d = unpack2x16float(weights[bw]).x;
for (var l: u32 = 0u; l < 32u; l = l + 1u) {
let w_word = weights[bw + 1u + (l >> 2u)];
let raw = (w_word >> ((l & 3u) * 8u)) & 0xFFu;
let signed_q = select(f32(raw), f32(raw) - 256.0, raw >= 128u);
let dq = d * signed_q;
let k = b * 32u + l;
sum = sum + f32(input[input_row_base + k]) * dq;
}
}
output[row * N + n] = ${useF16?'f16(sum)':'sum'};
}`;
export const matmulQ80KernelKey=opts=>'matmul-q8_0|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
