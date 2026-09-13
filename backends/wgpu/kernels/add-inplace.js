export const addInplaceKernel=({useF16=true,workgroupSize=64}={})=>`${useF16?'enable f16;':''}
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> a: array<${useF16?'f16':'f32'}>;
@group(0) @binding(2) var<storage, read> b: array<${useF16?'f16':'f32'}>;
@compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
if (i >= params.x) { return; }
a[i] = a[i] + b[i];
}`;
export const addInplaceKernelKey=opts=>'add-inplace|f16='+(opts&&opts.useF16!==false?'1':'0')+'|wg='+((opts&&opts.workgroupSize)||64);
