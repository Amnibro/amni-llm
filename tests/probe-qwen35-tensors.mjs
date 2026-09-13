import {loadGGUF} from '../backends/wgpu/gguf.js';
const URL='https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf';
const g=await loadGGUF(URL);
console.log('arch:',g.arch,'  tensors:',g.tensors.size,'  KV:',g.metadata.size);
console.log('\nAll metadata keys (excluding tokenizer.* arrays):');
for(const[k,v]of g.metadata){if(k.startsWith('tokenizer.ggml.tokens')||k.startsWith('tokenizer.ggml.merges')||k.startsWith('tokenizer.ggml.scores')||k.startsWith('tokenizer.ggml.token_type'))continue;const display=Array.isArray(v)?'[array len='+v.length+']':typeof v==='string'&&v.length>60?v.slice(0,60)+'...':v;console.log('  '+k+' = '+display);}
for(const blkIdx of[0,1,2,3,4,5,6,7]){
console.log('\nBlock '+blkIdx+' tensors:');
const blk=Array.from(g.tensors.values()).filter(t=>t.name.startsWith('blk.'+blkIdx+'.'));
for(const t of blk)console.log('  '+t.name+'  ['+t.shape.join('×')+']  '+t.dtypeName);
}
console.log('\nNon-block tensors:');
for(const t of g.tensors.values())if(!t.name.startsWith('blk.'))console.log('  '+t.name+'  ['+t.shape.join('×')+']  '+t.dtypeName);
console.log('\nLayer-type detection — does each layer have ssm_*?');
const md=g.metadata;const nLayers=md.get('qwen35.block_count')||md.get('qwen3.block_count')||0;
const counts={attn:0,ssm:0,both:0,neither:0};
for(let l=0;l<nLayers;l++){const hasSSM=Array.from(g.tensors.keys()).some(n=>n.startsWith(`blk.${l}.ssm`));const hasAttn=Array.from(g.tensors.keys()).some(n=>n.startsWith(`blk.${l}.attn`));if(hasSSM&&hasAttn)counts.both++;else if(hasSSM)counts.ssm++;else if(hasAttn)counts.attn++;else counts.neither++;}
console.log('  '+nLayers+' layers total: attn-only='+counts.attn+', ssm-only='+counts.ssm+', both='+counts.both+', neither='+counts.neither);
