import {STORAGE_RW} from './runtime.js';
export class GatedDeltaState{
constructor(runtime,opts){this.runtime=runtime;this.layerIndices=opts.layerIndices||[];this.qkvDim=opts.qkvDim;this.kernelSize=opts.kernelSize;this.numVHeads=opts.numVHeads;this.keyHeadDim=opts.keyHeadDim;this.valueHeadDim=opts.valueHeadDim;this.useF16=opts.useF16!==false;this.dtypeBytes=this.useF16?2:4;this.convStateBytes=this.qkvDim*this.kernelSize*this.dtypeBytes;this.recurrentStateBytes=this.numVHeads*this.keyHeadDim*this.valueHeadDim*4;this.convStates=new Map();this.recurrentStates=new Map();}
init(){for(const l of this.layerIndices){this.convStates.set(l,this.runtime.allocBuffer(this.convStateBytes,STORAGE_RW,'conv-l'+l));this.recurrentStates.set(l,this.runtime.allocBuffer(this.recurrentStateBytes,STORAGE_RW,'recur-l'+l));}}
reset(){const cz=new Uint8Array(this.convStateBytes);const rz=new Uint8Array(this.recurrentStateBytes);for(const buf of this.convStates.values())this.runtime.upload(buf.buf,0,cz);for(const buf of this.recurrentStates.values())this.runtime.upload(buf.buf,0,rz);}
conv(layer){return this.convStates.get(layer);}
recur(layer){return this.recurrentStates.get(layer);}
destroy(){for(const buf of this.convStates.values())this.runtime.freeBuffer(buf);for(const buf of this.recurrentStates.values())this.runtime.freeBuffer(buf);this.convStates.clear();this.recurrentStates.clear();}
}
export class SSMState extends GatedDeltaState{}
