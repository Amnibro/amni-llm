export const matmulQ50Kernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
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
let words_per_block = 6u;
let row_word_base = n * blocks_per_row * words_per_block;
let input_row_base = row * K;
var sum: f32 = 0.0;
for (var b: u32 = 0u; b < blocks_per_row; b = b + 1u) {
let bw = row_word_base + b * words_per_block;
let d = unpack2x16float(weights[bw]).x;
let qh = weights[bw + 1u];
for (var l: u32 = 0u; l < 32u; l = l + 1u) {
let qs_word_idx = 2u + (l >> 3u);
let nibble_pos = (l & 7u) * 4u;
let low4 = (weights[bw + qs_word_idx] >> nibble_pos) & 0xFu;
let high1 = (qh >> l) & 1u;
let q5 = low4 | (high1 << 4u);
let centered = f32(q5) - 16.0;
let dq = d * centered;
let k = b * 32u + l;
sum = sum + f32(input[input_row_base + k]) * dq;
}
}
output[row * N + n] = ${useF16?'f16(sum)':'sum'};
}`;
export const matmulQ50KernelKey=opts=>'matmul-q5_0|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
