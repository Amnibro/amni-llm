import {ALL_KERNELS,KERNEL_NAMES} from '../backends/wgpu/kernels/index.js';
let pass=0,fail=0;
const log=(...a)=>console.log(...a);
const check=(cond,msg)=>{cond?(log('  PASS  '+msg),pass++):(log('  FAIL  '+msg),fail++);};
log('Validating '+KERNEL_NAMES.length+' kernel templates × 2 variants (f16 + f32) = '+(KERNEL_NAMES.length*2)+' WGSL outputs\n');
for(const name of KERNEL_NAMES){
for(const useF16 of[true,false]){
const k=ALL_KERNELS[name];
const opts={useF16};
const key=k.key(opts);
let wgsl='';
try{wgsl=k.build(opts);}catch(e){fail++;log('  FAIL  '+key+' — build threw: '+e.message);continue;}
check(typeof wgsl==='string'&&wgsl.length>50,key+' emits non-empty WGSL ('+wgsl.length+' chars)');
check(useF16?wgsl.includes('enable f16'):!wgsl.includes('enable f16'),key+' f16 directive matches variant');
check(wgsl.includes('@compute'),key+' has @compute entry point');
check(wgsl.includes('@workgroup_size'),key+' has @workgroup_size annotation');
check(wgsl.includes('@group(0) @binding(0)'),key+' has uniform binding 0');
check(wgsl.includes('fn main'),key+' has main function');
}
}
log('\n=== '+pass+' pass, '+fail+' fail ===');
log('\nSample (matmul-q4_k_m, f16 variant):');
log('-------- 8< --------');
log(ALL_KERNELS['matmul-q4_k_m'].build({useF16:true}));
log('-------- 8< --------');
process.exit(fail>0?1:0);
