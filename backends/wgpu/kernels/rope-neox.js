export const ropeNeoxKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> data: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> positions: array<u32>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let token = gid.z;
let head = gid.y;
let pair = gid.x;
let n_tokens = params.x;
let n_heads = params.y;
let head_dim = params.z;
let freq_base = bitcast<f32>(params.w);
let half_dim = head_dim / 2u;
if (token >= n_tokens || head >= n_heads || pair >= half_dim) { return; }
let pos = f32(positions[token]);
let inv_freq = 1.0 / pow(freq_base, f32(pair * 2u) / f32(head_dim));
let theta = pos * inv_freq;
let c = cos(theta);
let s = sin(theta);
let base = (token * n_heads + head) * head_dim;
let i_lo = base + pair;
let i_hi = base + pair + half_dim;
let x0 = f32(data[i_lo]);
let x1 = f32(data[i_hi]);
let r0 = x0 * c - x1 * s;
let r1 = x0 * s + x1 * c;
data[i_lo] = ${useF16?'f16(r0)':'r0'};
data[i_hi] = ${useF16?'f16(r1)':'r1'};
}`;
export const ropeNeoxKernelKey=opts=>'rope-neox|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
