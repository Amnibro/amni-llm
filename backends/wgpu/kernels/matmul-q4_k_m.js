export const matmulQ4KMKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
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
let row_word_base = n * blocks_per_row * 36u;
let input_row_base = row * K;
var sum: f32 = 0.0;
for (var b: u32 = 0u; b < blocks_per_row; b = b + 1u) {
let bw = row_word_base + b * 36u;
let dm = unpack2x16float(weights[bw]);
let d = dm.x;
let dmin = dm.y;
let s0 = weights[bw + 1u];
let s1 = weights[bw + 2u];
let s2 = weights[bw + 3u];
for (var sb: u32 = 0u; sb < 8u; sb = sb + 1u) {
var sc_v: u32;
var mn_v: u32;
if (sb < 4u) {
sc_v = (s0 >> (sb * 8u)) & 0x3Fu;
mn_v = (s1 >> (sb * 8u)) & 0x3Fu;
} else {
let pos = sb - 4u;
let lower = (s2 >> (pos * 8u)) & 0xFFu;
let high_sc = ((s0 >> (pos * 8u)) >> 6u) & 0x3u;
let high_mn = ((s1 >> (pos * 8u)) >> 6u) & 0x3u;
sc_v = (lower & 0xFu) | (high_sc << 4u);
mn_v = (lower >> 4u) | (high_mn << 4u);
}
let dq_scale = d * f32(sc_v);
let dq_min = dmin * f32(mn_v);
let pair_word_base = bw + 4u + (sb >> 1u) * 8u;
let nibble_high = (sb & 1u) == 1u;
for (var l: u32 = 0u; l < 32u; l = l + 1u) {
let w_word = weights[pair_word_base + (l >> 2u)];
let w_byte = (w_word >> ((l & 3u) * 8u)) & 0xFFu;
let q = select(w_byte & 0xFu, (w_byte >> 4u) & 0xFu, nibble_high);
let dq = dq_scale * f32(q) - dq_min;
let k = b * 256u + sb * 32u + l;
sum = sum + f32(input[input_row_base + k]) * dq;
}
}
}
output[row * N + n] = ${useF16?'f16(sum)':'sum'};
}`;
export const matmulQ4KMKernelKey=opts=>'matmul-q4_k_m|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
