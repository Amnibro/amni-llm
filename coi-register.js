(function(){
if(typeof window==='undefined'||!('serviceWorker' in navigator))return;
/* Always register: the worker also keeps the app shell cached for offline use, so it is needed even when the
   host already sends COOP/COEP (window.crossOriginIsolated is true). It only reloads the page when isolation is missing. */
const iso=window.crossOriginIsolated,K='amni-llm-coi-reload';
const reload=()=>{try{if(sessionStorage.getItem(K))return;sessionStorage.setItem(K,'1');}catch{}console.log('[amni-llm] reloading to activate cross-origin isolation');window.location.reload();};
if(iso){try{sessionStorage.removeItem(K);}catch{}}
navigator.serviceWorker.register('./coi-sw.js',{scope:'./'}).then(reg=>{
console.log('[amni-llm] service worker registered',reg.scope);
if(iso)return;
if(reg.active&&!navigator.serviceWorker.controller)reload();
else navigator.serviceWorker.addEventListener('controllerchange',()=>{if(!window.crossOriginIsolated)reload();},{once:true});
}).catch(e=>console.warn('[amni-llm] service worker registration failed (no offline shell; loads still work, just slower / size-limited):',e));
})();
