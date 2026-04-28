(function(){
if(typeof window==='undefined')return;
if(window.crossOriginIsolated){console.log('[amni-llm] cross-origin isolated already');return;}
if(!('serviceWorker' in navigator))return;
navigator.serviceWorker.register('./coi-sw.js',{scope:'./'}).then(reg=>{
console.log('[amni-llm] coi-sw registered',reg.scope);
if(reg.active&&!navigator.serviceWorker.controller){console.log('[amni-llm] reloading to activate cross-origin isolation');window.location.reload();}
}).catch(e=>console.warn('[amni-llm] coi-sw registration failed (loads still work, just slower / size-limited):',e));
})();
