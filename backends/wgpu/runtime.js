const SOFTWARE_RE=/swiftshader|llvmpipe|lavapipe|software|microsoft basic render/i;
const PIPELINE_ERROR_RE=/GPUPipelineError|CreateComputePipelines|VK_ERROR|Vulkan/i;
const nextPow2=n=>{let p=256;while(p<n)p*=2;return p;};
export function isPipelineError(e){return PIPELINE_ERROR_RE.test(String(e&&e.message||e));}
export async function probeWebGPU(){
if(typeof navigator==='undefined'||!('gpu'in navigator))return{available:false,reason:'navigator.gpu missing — WebGPU not supported in this browser'};
const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'}).catch(e=>({_err:e}));
if(!adapter||adapter._err)return{available:false,reason:'no WebGPU adapter (driver missing or disabled)'};
const info=adapter.info||(adapter.requestAdapterInfo?await adapter.requestAdapterInfo().catch(()=>({})):{});
const blob=((info.vendor||'')+' '+(info.architecture||'')+' '+(info.device||'')+' '+(info.description||'')).toLowerCase();
const isSoftware=SOFTWARE_RE.test(blob);
if(isSoftware)return{available:false,reason:'software adapter rejected: '+(info.description||info.vendor||'unknown')+' (slower than CPU WASM, would waste compile time)',adapter,info,isSoftware:true};
const hasF16=adapter.features.has('shader-f16');
return{available:true,adapter,info,hasF16,limits:adapter.limits};
}
export class WGPURuntime{
constructor(){this.device=null;this.adapter=null;this.info=null;this.hasF16=false;this.bufferPool=new Map();this.pipelineCache=new Map();this.lostHandlers=[];this._destroyed=false;}
async init(opts={}){
const probe=await probeWebGPU();
if(!probe.available)throw Object.assign(new Error('WGPU init: '+probe.reason),{kind:'webgpu-unavailable',probe});
this.adapter=probe.adapter;this.info=probe.info;this.hasF16=probe.hasF16;
const requiredFeatures=this.hasF16?['shader-f16']:[];
const al=this.adapter.limits;
const requiredLimits={maxStorageBufferBindingSize:al.maxStorageBufferBindingSize,maxBufferSize:al.maxBufferSize,maxStorageBuffersPerShaderStage:al.maxStorageBuffersPerShaderStage,maxComputeWorkgroupStorageSize:al.maxComputeWorkgroupStorageSize,maxComputeInvocationsPerWorkgroup:al.maxComputeInvocationsPerWorkgroup,maxComputeWorkgroupSizeX:al.maxComputeWorkgroupSizeX,maxComputeWorkgroupSizeY:al.maxComputeWorkgroupSizeY,maxComputeWorkgroupSizeZ:al.maxComputeWorkgroupSizeZ,...(opts.limits||{})};
console.log('[WGPU] runtime.js v2 — requesting maxStorageBufferBindingSize:',requiredLimits.maxStorageBufferBindingSize,'maxBufferSize:',requiredLimits.maxBufferSize);
this.device=await this.adapter.requestDevice({requiredFeatures,requiredLimits}).catch(e=>{throw Object.assign(new Error('requestDevice failed: '+(e.message||e)),{kind:'device-init-failed',cause:e});});
console.log('[WGPU] device granted maxStorageBufferBindingSize:',this.device.limits.maxStorageBufferBindingSize);
this.device.lost.then(info=>{this._destroyed=true;this.lostHandlers.forEach(h=>{try{h(info);}catch{}});});
this.device.onuncapturederror=ev=>{const m=ev&&ev.error&&ev.error.message||'unknown';if(opts.onError)opts.onError(m);else console.warn('WGPU uncaptured error:',m);};
return{adapter:{vendor:this.info.vendor||'?',architecture:this.info.architecture||'?',device:this.info.device||'',description:this.info.description||''},hasF16:this.hasF16,limits:{maxStorageBuffer:this.device.limits.maxStorageBufferBindingSize,maxComputeWorkgroupStorageSize:this.device.limits.maxComputeWorkgroupStorageSize}};
}
allocBuffer(size,usage,label){const cls=nextPow2(size);const key=cls+'|'+usage;const pool=this.bufferPool.get(key)||[];const buf=pool.pop()||this.device.createBuffer({size:cls,usage,label:label||('buf-'+cls)});return{buf,size,cls,usage,_key:key};}
freeBuffer(h){if(!h||!h.buf)return;const pool=this.bufferPool.get(h._key)||[];pool.push(h.buf);this.bufferPool.set(h._key,pool);}
destroyAllBuffers(){for(const[,pool]of this.bufferPool)for(const buf of pool){try{buf.destroy();}catch{}}this.bufferPool.clear();}
async getPipeline(key,wgslSource,entryPoint='main'){
if(this.pipelineCache.has(key))return this.pipelineCache.get(key);
let module;
try{module=this.device.createShaderModule({code:wgslSource,label:key});}catch(e){throw Object.assign(new Error('shader compile (createShaderModule) failed for '+key+': '+(e.message||e)),{kind:'shader-compile-failed',key,cause:e});}
if(module.getCompilationInfo){const info=await module.getCompilationInfo().catch(()=>null);if(info&&info.messages){const errors=info.messages.filter(m=>m.type==='error');if(errors.length){throw Object.assign(new Error('WGSL compile errors in '+key+':\n'+errors.map(e=>'  ['+e.lineNum+':'+e.linePos+'] '+e.message).join('\n')),{kind:'shader-compile-failed',key,errors});}}}
const pipeline=await this.device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint},label:key}).catch(e=>{throw Object.assign(new Error('pipeline create failed for '+key+': '+(e.message||e)),{kind:'pipeline-error',key,cause:e});});
this.pipelineCache.set(key,pipeline);return pipeline;
}
makeBindGroup(pipeline,resources){return this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:resources.map((r,i)=>({binding:i,resource:r.resource||{buffer:r.buf,offset:r.offset||0,size:r.size||r.cls}}))});}
dispatch(encoder,pipeline,resources,wx,wy=1,wz=1){const bg=this.makeBindGroup(pipeline,resources);const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bg);pass.dispatchWorkgroups(wx,wy,wz);pass.end();}
async readback(buffer,offset,size){const staging=this.device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,label:'readback'});const enc=this.device.createCommandEncoder();enc.copyBufferToBuffer(buffer,offset,staging,0,size);this.device.queue.submit([enc.finish()]);await staging.mapAsync(GPUMapMode.READ);const copy=staging.getMappedRange().slice(0);staging.unmap();staging.destroy();return new Uint8Array(copy);}
upload(buf,offset,bytes){this.device.queue.writeBuffer(buf,offset,bytes);}
beginEncoder(label){return this.device.createCommandEncoder({label:label||'enc'});}
submit(encoder){this.device.queue.submit([encoder.finish()]);}
async sync(){await this.device.queue.onSubmittedWorkDone();}
onDeviceLost(h){this.lostHandlers.push(h);}
destroy(){if(this._destroyed)return;this._destroyed=true;this.destroyAllBuffers();this.pipelineCache.clear();if(this.device){try{this.device.destroy();}catch{}}this.device=null;}
}
export const BUFFER_USAGE={STORAGE:0x80,UNIFORM:0x40,COPY_SRC:0x4,COPY_DST:0x8};
export const STORAGE_RW=BUFFER_USAGE.STORAGE|BUFFER_USAGE.COPY_SRC|BUFFER_USAGE.COPY_DST;
export const UNIFORM_RW=BUFFER_USAGE.UNIFORM|BUFFER_USAGE.COPY_DST;
