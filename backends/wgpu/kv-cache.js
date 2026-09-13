import {STORAGE_RW} from './runtime.js';
export class KVCache{
constructor(runtime,opts){this.runtime=runtime;this.nLayers=opts.nLayers;this.nHeadsKV=opts.nHeadsKV;this.headDim=opts.headDim;this.maxCtx=opts.maxCtx||4096;this.useF16=opts.useF16!==false;this.dtypeBytes=this.useF16?2:4;this.layerBytes=this.maxCtx*this.nHeadsKV*this.headDim*this.dtypeBytes;this.kBuffers=[];this.vBuffers=[];this.position=0;}
init(){for(let l=0;l<this.nLayers;l++){this.kBuffers.push(this.runtime.allocBuffer(this.layerBytes,STORAGE_RW,'kv-K-l'+l));this.vBuffers.push(this.runtime.allocBuffer(this.layerBytes,STORAGE_RW,'kv-V-l'+l));}}
reset(){this.position=0;}
advance(n=1){this.position+=n;}
tokenBytes(){return this.nHeadsKV*this.headDim*this.dtypeBytes;}
kBuffer(layer){return this.kBuffers[layer];}
vBuffer(layer){return this.vBuffers[layer];}
destroy(){for(const b of this.kBuffers)this.runtime.freeBuffer(b);for(const b of this.vBuffers)this.runtime.freeBuffer(b);this.kBuffers=[];this.vBuffers=[];}
}
