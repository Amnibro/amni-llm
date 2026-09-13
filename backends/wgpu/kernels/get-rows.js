export const getRowsKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> table: array<${useF16?'f16':'f32'}>;
@group(0) @binding(3) var<storage, read_write> out: array<${useF16?'f16':'f32'}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let row = gid.y;
let col = gid.x;
let n_rows = params.x;
let dim = params.y;
if (row >= n_rows || col >= dim) { return; }
let id = ids[row];
out[row * dim + col] = table[id * dim + col];
}`;
export const getRowsKernelKey=opts=>'get-rows|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
