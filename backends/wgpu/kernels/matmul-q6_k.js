export const matmulQ6KKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
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
let blocks_per_row = K / 256u;
let words_per_block = 53u;
let row_word_base = n * blocks_per_row * words_per_block;
let input_row_base = row * K;
var sum: f32 = 0.0;
for (var b: u32 = 0u; b < blocks_per_row; b = b + 1u) {
let bw = row_word_base + b * words_per_block;
let ql_base = bw;
let qh_base = bw + 32u;
let scales_base = bw + 48u;
let d_word = weights[bw + 52u];
let d = unpack2x16float(d_word).x;
for (var sb: u32 = 0u; sb < 16u; sb = sb + 1u) {
let scale_word = weights[scales_base + (sb >> 2u)];
let scale_byte = (scale_word >> ((sb & 3u) * 8u)) & 0xFFu;
let scale_signed = select(f32(scale_byte), f32(scale_byte) - 256.0, scale_byte >= 128u);
let dq_scale = d * scale_signed;
for (var l: u32 = 0u; l < 16u; l = l + 1u) {
let weight_idx_in_block = sb * 16u + l;
let ql_byte_idx = weight_idx_in_block >> 1u;
let ql_word_idx = ql_base + (ql_byte_idx >> 2u);
let ql_byte = (weights[ql_word_idx] >> ((ql_byte_idx & 3u) * 8u)) & 0xFFu;
let ql_nibble = weight_idx_in_block & 1u;
let low4 = select(ql_byte & 0xFu, (ql_byte >> 4u) & 0xFu, ql_nibble == 1u);
let qh_bit_idx = weight_idx_in_block * 2u;
let qh_byte_idx = qh_bit_idx >> 3u;
let qh_word_idx = qh_base + (qh_byte_idx >> 2u);
let qh_byte = (weights[qh_word_idx] >> ((qh_byte_idx & 3u) * 8u)) & 0xFFu;
let high2 = (qh_byte >> (qh_bit_idx & 7u)) & 0x3u;
let q6 = low4 | (high2 << 4u);
let centered = f32(q6) - 32.0;
let dq = dq_scale * centered;
let k = b * 256u + sb * 16u + l;
sum = sum + f32(input[input_row_base + k]) * dq;
}
}
}
output[row * N + n] = ${useF16?'f16(sum)':'sum'};
}`;
export const matmulQ6KKernelKey=opts=>'matmul-q6_k|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
