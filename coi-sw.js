/*! Amni-LLM service worker: cross-origin isolation + offline app shell.
    COI part adapted from https://github.com/gzuidhof/coi-serviceworker (BSD-3-Clause). MIT-compatible.
    - Adds COOP/COEP/CORP headers so SharedArrayBuffer / multi-thread WASM work on static hosts.
    - Keeps the app shell (HTML, JS, WASM runtime, font) in its own versioned cache so the page opens offline.
      Pages/JS: network first, cache fallback. Versioned vendor files and the font: cache first.
    - Never caches model weights: .gguf files, /models/ paths, cross-origin requests, Range requests and partial
      (206 / Content-Range) responses. Model bytes are persisted by the app itself (Cache Storage 'amni-llm-v1' for
      verified defaults, wllama's OPFS cache for installed URLs), so nothing is stored twice.
    Bump VER whenever the precache list changes; old 'amni-llm-shell-*' caches are deleted on activate. */
const VER='amni-llm-shell-v2';
const SCOPE=new URL('./',self.location).href;
const FONT='/assets/fonts/archivo-var.woff2';
const PRECACHE=['./','./amni-llm.js','./registry.js','./coi-register.js','./ptex.js','./vendor/wllama-3.13.0/index.js','./vendor/wllama-3.13.0/wllama.wasm',FONT].map(p=>new URL(p,SCOPE).href);
const coi=res=>{if(!res||res.status===0||res.type==='opaqueredirect')return res;const h=new Headers(res.headers);h.set('Cross-Origin-Embedder-Policy','credentialless');h.set('Cross-Origin-Opener-Policy','same-origin');h.set('Cross-Origin-Resource-Policy','cross-origin');return new Response(res.body,{status:res.status,statusText:res.statusText,headers:h});};
const isModel=u=>/\.gguf$/i.test(u.pathname)||/\/models\//.test(u.pathname);
const cacheable=(req,u)=>req.method==='GET'&&u.origin===self.location.origin&&!req.headers.has('range')&&!isModel(u)&&(u.href.startsWith(SCOPE)||u.pathname===FONT);
const storable=res=>!!res&&res.status===200&&res.type==='basic'&&!res.headers.has('content-range');
const keyOf=u=>{const k=new URL(u.pathname,u.origin).href;return k===SCOPE+'index.html'?SCOPE:k;};
self.addEventListener('install',e=>{self.skipWaiting();e.waitUntil(caches.open(VER).then(c=>Promise.allSettled(PRECACHE.map(u=>fetch(u,{cache:'no-cache'}).then(r=>storable(r)?c.put(u,r):0)))));});
self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k.startsWith('amni-llm-shell-')&&k!==VER).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener('message',e=>{if(e.data&&e.data.type==='deregister')self.registration.unregister().then(()=>self.clients.matchAll()).then(cs=>cs.forEach(c=>c.navigate(c.url)));});
self.addEventListener('fetch',e=>{
const r=e.request;
if(r.cache==='only-if-cached'&&r.mode!=='same-origin')return;
const u=new URL(r.url);
if(!cacheable(r,u)){e.respondWith(fetch(r).then(coi,()=>Response.error()));return;}
const k=keyOf(u),pinned=u.pathname.includes('/vendor/')||u.pathname===FONT;
e.respondWith((async()=>{
const c=await caches.open(VER);
if(pinned){const hit=await c.match(k);if(hit)return coi(hit);}
try{const res=await fetch(r);if(storable(res))e.waitUntil(c.put(k,res.clone()).catch(()=>{}));return coi(res);}
catch(err){const hit=await c.match(k)||(r.mode==='navigate'?await c.match(SCOPE):null);return hit?coi(hit):Response.error();}
})());
});
