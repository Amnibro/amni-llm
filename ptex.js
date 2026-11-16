export const DIM=512;
const h32=s=>{let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);}return h>>>0;};
export const words=t=>String(t||'').toLowerCase().replace(/[-+]?\d*\.?\d+(?:\/\d+)?/g,' # ').replace(/[^a-z#]+/g,' ').trim().split(/\s+/).filter(Boolean);
export const key=(q,units,sig)=>String(q||'').toLowerCase().replace(/\s+/g,' ').trim()+'|'+(units||'')+'|'+(sig||'');
export const blank=u=>JSON.stringify({name:'Untitled part',units:u,material:'al6061',params:[],features:[],errors:[],size:null});
export const sigOf=st=>{try{const d=typeof st==='string'?JSON.parse(st):st||{};return(d.features||[]).map(f=>f.type).join(',');}catch{return'';}};
export function vec(t){const w=words(t),v=new Float32Array(DIM);for(let i=0;i<w.length;i++){if(w[i]!=='#')v[h32(w[i])%DIM]+=1;i+1<w.length&&(v[h32(w[i]+' '+w[i+1])%DIM]+=0.7);}let n=0;for(let i=0;i<DIM;i++)n+=v[i]*v[i];n=Math.sqrt(n)||1;for(let i=0;i<DIM;i++)v[i]/=n;return v;}
export const addr=(v,g=17)=>{let a=0;for(let i=0;i<7;i++){let s=0;for(let j=i;j<DIM;j+=7)s+=v[j];a=a*g+Math.min(g-1,Math.max(0,Math.floor((s+1)*g/2)));}return a;};
export class Bank{
constructor(){this.L=[];this.V=[];this.K=new Map();this.packs=new Set();}
add(l,pack=''){const k=key(l.q,l.u,l.s);if(this.K.has(k))return false;this.K.set(k,this.L.length);this.L.push({...l,pack});this.V.push(vec(l.q));return true;}
merge(p){if(!p||p.v!==1||!Array.isArray(p.lessons)||this.packs.has(p.id))return 0;this.packs.add(p.id);let n=0;for(const l of p.lessons)typeof l.q==='string'&&typeof l.a==='string'&&this.add(l,p.id)&&n++;return n;}
exact(q,units,sig){const i=this.K.get(key(q,units,sig));return i==null?null:this.L[i];}
recall(q,units,sig,k=2,gate=0.5){const v=vec(q),ed=sig?1:0,o=[];for(let i=0;i<this.L.length;i++){const l=this.L[i];if(l.u!==units||(l.s?1:0)!==ed)continue;const w=this.V[i];let d=0;for(let j=0;j<DIM;j++)d+=v[j]*w[j];d+=ed&&l.s===sig?0.2:0;d>=gate&&o.push({d,l});}o.sort((a,b)=>b.d-a.d);const out=[],seen=new Set();for(const x of o){const f=x.l.a.replace(/[-+]?\d*\.?\d+/g,'#');if(seen.has(f))continue;seen.add(f);out.push({...x.l,score:+x.d.toFixed(3)});if(out.length>=k)break;}return out;}}
export async function sha256hex(t){const b=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(t));return[...new Uint8Array(b)].map(x=>x.toString(16).padStart(2,'0')).join('');}
export async function fetchPacks(base,bank,dom='cad'){const m=await(await fetch(base+'manifest.json',{cache:'no-cache'})).json(),out=[];for(const e of m.packs||[]){if(bank.packs.has(e.id)||(e.domain||'cad')!==dom||!/^[\w.-]+\.json$/.test(e.file||''))continue;const t=await(await fetch(base+e.file)).text();if(await sha256hex(t)!==e.sha256)continue;let p;try{p=JSON.parse(t);}catch{continue;}if(p.id!==e.id)continue;out.push({id:e.id,n:bank.merge(p)});}return out;}
