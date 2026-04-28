/*! coi-serviceworker pattern - enables cross-origin isolation on static hosts so SharedArrayBuffer works.
    Adapted from https://github.com/gzuidhof/coi-serviceworker (BSD-3-Clause). MIT-compatible. */
self.addEventListener('install',()=>self.skipWaiting());
self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));
self.addEventListener('message',e=>{if(e.data&&e.data.type==='deregister')self.registration.unregister().then(()=>self.clients.matchAll()).then(cs=>cs.forEach(c=>c.navigate(c.url)));});
self.addEventListener('fetch',e=>{
const r=e.request;
if(r.cache==='only-if-cached'&&r.mode!=='same-origin')return;
e.respondWith(fetch(r).then(res=>{
if(res.status===0)return res;
const h=new Headers(res.headers);
h.set('Cross-Origin-Embedder-Policy','credentialless');
h.set('Cross-Origin-Opener-Policy','same-origin');
h.set('Cross-Origin-Resource-Policy','cross-origin');
return new Response(res.body,{status:res.status,statusText:res.statusText,headers:h});
}).catch(err=>{console.error('[coi-sw] fetch err',err);return new Response('coi-sw fetch error',{status:500});}));
});
