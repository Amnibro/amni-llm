import {loadGGUF,summarize,readTensorBytes,tensorByteSize,GGUF_MAGIC} from '../backends/wgpu/gguf.js';
const URLS=[
['Qwen2.5-0.5B Q4_K_M (V3, official Qwen)','https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf'],
['Qwen3.5-0.8B Q4_K_M (V3, unsloth, registry default)','https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf']
];
let pass=0,fail=0;
const check=(cond,msg)=>{cond?(console.log('  PASS  '+msg),pass++):(console.log('  FAIL  '+msg),fail++);};
const log=(...a)=>console.log(...a);
async function runOne(label,url){
log('\n========================================================================');
log(label);
log('  '+url);
log('========================================================================');
const t0=Date.now();
let g;try{g=await loadGGUF(url);}catch(e){log('  LOAD ERROR: '+e.message);fail++;return;}
log('Parsed in '+(Date.now()-t0)+'ms');
log('\nSummary:');
log(JSON.stringify(summarize(g),null,2));
log('\nHeader:');
check(g.header.magic===GGUF_MAGIC,'magic = 0x46554747');
check([1,2,3].includes(g.header.version),'version in {1,2,3} (got '+g.header.version+')');
check(g.header.tensorCount>0,'tensorCount > 0 (got '+g.header.tensorCount+')');
check(g.header.kvCount>0,'kvCount > 0 (got '+g.header.kvCount+')');
log('\nIntegrity:');
check(g.tensors.size===g.header.tensorCount,'tensors.size matches header.tensorCount');
check(g.metadata.size===g.header.kvCount,'metadata.size matches header.kvCount');
check(g.metadata.has('general.architecture'),'general.architecture present (got "'+g.arch+'")');
check(typeof g.arch==='string'&&g.arch!=='?','arch extracted');
check(g.dataStart>0&&g.dataStart%g.alignment===0,'dataStart alignment-aligned ('+g.dataStart+' % '+g.alignment+' === 0)');
check(typeof g.quantHint==='string','quantHint detected (got "'+g.quantHint+'")');
log('\nTensor table:');
let totalBytes=0;const invalid=[];
for(const t of g.tensors.values()){try{const sz=tensorByteSize(t);totalBytes+=sz;if(sz<=0)invalid.push(t.name+' (zero size)');}catch(e){invalid.push(t.name+' ('+e.message+')');}}
check(invalid.length===0,'all '+g.tensors.size+' tensors have computable byte sizes'+(invalid.length?' - first failures: '+invalid.slice(0,3).join('; '):''));
log('  total weight bytes = '+(totalBytes/1048576).toFixed(1)+' MB');
log('\nRanged tensor read:');
const small=Array.from(g.tensors.values()).filter(t=>t.dtype===0||t.dtype===1).sort((a,b)=>tensorByteSize(a)-tensorByteSize(b))[0];
if(small){
const expected=tensorByteSize(small);
log('  reading "'+small.name+'" ['+small.shape.join('x')+'] '+small.dtypeName+' (expect '+expected+' bytes)');
const t1=Date.now();
let bytes;try{bytes=await readTensorBytes(g,small.name);}catch(e){log('  FETCH ERROR: '+e.message);fail++;return;}
log('  fetched '+bytes.length+' bytes in '+(Date.now()-t1)+'ms');
check(bytes.length===expected,'fetched bytes match expected size');
check(bytes instanceof Uint8Array,'returns Uint8Array');
log('  first 16 bytes (hex): '+Array.from(bytes.slice(0,16)).map(b=>b.toString(16).padStart(2,'0')).join(' '));
}else{check(false,'no F32/F16 tensor found to test ranged read');}
}
for(const[label,url]of URLS){await runOne(label,url);}
log('\n========================================================================');
log('FINAL: '+pass+' pass, '+fail+' fail');
log('========================================================================');
process.exit(fail>0?1:0);
