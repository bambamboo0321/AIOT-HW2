import {writeFile} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const WEB=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
export async function collect(spaceId){
 const task=await taskSpace(spaceId),page=task.page('p1');
 await page.cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 const before=await page.evaluate(()=>document.documentElement.dataset.theme);
 await page.press('button[aria-label="切換深淺色模式"]','Enter');
 const mobile=await page.evaluate(before=>({width:innerWidth,scroll_width:document.documentElement.scrollWidth,no_overflow:document.documentElement.scrollWidth<=innerWidth,theme_before:before,theme_after:document.documentElement.dataset.theme}),before);
 await page.cdp('Emulation.clearDeviceMetricsOverride');
 await page.cdp('Network.enable');
 const assets=[];
 for(const name of ['pyodide.mjs','pyodide.asm.wasm']){
  const response=await page.fetch('https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'+name,{method:'HEAD',timeout:15000});
  assets.push({name,status:response.status,content_type:response.headers['content-type']});
 }
 const events=await page.events(),headers=[];
 for(const e of events){const r=e.params?.response;if(e.method==='Network.responseReceived'&&r?.url?.startsWith('https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'))headers.push({url:r.url,status:r.status,headers:Object.fromEntries(Object.entries(r.headers??{}).filter(([k])=>['content-type','content-length','content-encoding','access-control-allow-origin','cache-control'].includes(k.toLowerCase())))});}
 const worker=await page.fetch('/browser-compute/worker.mjs',{method:'HEAD'}),document=await page.fetch('/custom-dataset',{method:'HEAD'});
 const result=await page.evaluate(()=>window.uiAcceptance);
 result.mobile=mobile;result.remaining_workers=(await task.cdp('Target.getTargets')).targetInfos.filter(t=>t.type==='worker'&&t.url.includes('127.0.0.1:3017/browser-compute/')).length;
 result.status=result.api_calls.length===0&&result.server_ui.train&&result.mode_switch.workers_after===0&&result.config_change.workers===0&&mobile.no_overflow?'PASS':'FAIL';
 result.network={head_checks:assets,cdn_headers:headers,worker_csp:worker.headers['content-security-policy'],document_csp:document.headers['content-security-policy']??null,caveat:'Compatibility checks with cache state unspecified; not a new first-use benchmark.'};
 await writeFile(resolve(WEB,'verification/m17-b5-ui.json'),JSON.stringify(result,null,2)+'\n');
 console.log(result);
}
