// Focused production release check; reuses completed B5 parity evidence instead of repeating benchmarks.
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const web=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
export async function start(spaceId){
 const task=await taskSpace(spaceId),page=task.page('p1');
 const fixtures=JSON.parse(await readFile(resolve(web,'tests/fixtures/browser-compute/cases.json'),'utf8'));
 await page.cdp('Network.enable');
 await page.evaluate(({fixture,rtol,atol})=>{
  const state=window.releaseSmoke={status:'running',stage:'init',rtol,atol,api_calls:[],cache:'uncontrolled; not a cold-start benchmark',user_agent:navigator.userAgent,secure_context:isSecureContext};
  state.server_default=document.querySelector('#compute-mode-server').getAttribute('aria-pressed')==='true';
  state.server_unavailable=document.querySelector('#server-compute-status').textContent.includes('無法執行');
  void(async()=>{
   let client;
   const must=(v,message)=>{if(!v)throw new Error(message);};
   const original=window.fetch.bind(window);
   try{
    must(state.server_default&&state.server_unavailable,'Server default/unavailable indication missing');
    const direct=await original('/api/custom-dataset/validate',{method:'POST'});
    state.server_api_status=direct.status;must(direct.status===503,'Unconfigured M16 must fail closed');
    window.fetch=(input,init)=>{if(String(input instanceof Request?input.url:input).includes('/api/custom-dataset/'))state.api_calls.push(String(input));return original(input,init);};
    const {BrowserComputeClient}=await import('/browser-compute/modules/lib/browser-compute/client.mjs');
    client=new BrowserComputeClient();
    state.initialization=await client.initialize();
    state.stage='validate';const input={csv:new Blob([fixture.csv]),config:fixture.config,review:fixture.review};
    const validation=await client.validate(input);must(validation.report.status==='valid','Validation failed');
    const reference=fixture.references.find(r=>r.selection.method==='rfe');
    const experiment={schema_version:1,method:'linear_regression',confirmed:true,dataset_sha256:validation.report.dataset.sha256,config_sha256:validation.report.config_sha256,selection:{method:'rfe',k:reference.selection.requested_k}};
    state.stage='train';const trained=await client.train({...input,experiment});
    must(trained.report.status==='completed','Training failed');
    state.holdout_predictions=trained.report.preview.length;
    state.max_prediction_difference=Math.max(...trained.report.preview.map((p,i)=>Math.abs(p.prediction-reference.preview[i].prediction)));
    must(trained.report.preview.every((p,i)=>Math.abs(p.prediction-reference.preview[i].prediction)<=atol+rtol*Math.abs(reference.preview[i].prediction)),'Native parity mismatch');
    const native=fixture.inference.rfe,model=trained.report.prediction_model;
    state.stage='predict';const single=await client.predict(model.id,native.values);const batch=await client.predictBatch(model.id,native.rows);
    must(Math.abs(single.report.prediction-native.single.prediction)<=atol+rtol*Math.abs(native.single.prediction),'Single inference mismatch');
    must(batch.report.status==='batch_predicted','Batch failed');
    must(batch.report.rows.every((p,i)=>Math.abs(p.prediction-native.batch.rows[i].prediction)<=atol+rtol*Math.abs(native.batch.rows[i].prediction)),'Batch mismatch');
    state.single=true;state.batch_rows=batch.report.rows.length;state.metrics=trained.report.holdout.linear_regression;
    await client.deleteModel(model.id);state.model_deleted=!client.hasModel(model.id);
    must(state.api_calls.length===0,'Browser data reached Server API');
    state.status='PASS';state.stage='complete';
   }catch(e){state.status='FAIL';state.error={message:e.message,code:e.code};}
   finally{client?.dispose();window.fetch=original;}
  })();
 },{fixture:fixtures.cases.find(c=>c.name==='missing'),rtol:fixtures.rtol,atol:fixtures.atol});
 console.log({started:true,spaceId});
}
export async function collect(spaceId){
 const task=await taskSpace(spaceId),page=task.page('p1');
 const result=await page.evaluate(()=>window.releaseSmoke);
 if(result.status==='running'){console.log({status:result.status,stage:result.stage});return;}
 result.remaining_workers=(await task.cdp('Target.getTargets')).targetInfos.filter(t=>t.type==='worker'&&t.url.includes('127.0.0.1:3018/browser-compute/')).length;
 result.headers={};for(const path of ['/custom-dataset','/browser-compute/worker.mjs','/browser-compute/python-manifest.json']){
  const response=await page.fetch('http://127.0.0.1:3018'+path,{method:'HEAD'});
  result.headers[path]={status:response.status,content_type:response.headers['content-type'],csp:response.headers['content-security-policy']??null,cache:response.headers['cache-control']??null};
 }
 result.cdn=[];for(const name of ['pyodide.mjs','pyodide.asm.wasm']){
  const response=await page.fetch('https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'+name,{method:'HEAD',timeout:15000});
  result.cdn.push({name,status:response.status,mime:response.headers['content-type'],cors:response.headers['access-control-allow-origin']});
 }
 const events=await page.events();
 result.network=events.filter(e=>e.method==='Network.responseReceived'&&e.params?.response?.url?.includes('cdn.jsdelivr.net/pyodide/v314.0.7/full/')).map(e=>({url:e.params.response.url,status:e.params.response.status,mime:e.params.response.mimeType,from_disk_cache:e.params.response.fromDiskCache??false,from_service_worker:e.params.response.fromServiceWorker??false}));
 await writeFile(resolve(web,'verification/m17-release-browser.json'),JSON.stringify(result,null,2)+'\n');
 console.log({status:result.status,error:result.error,api_calls:result.api_calls,holdout_predictions:result.holdout_predictions,single:result.single,batch_rows:result.batch_rows,remaining_workers:result.remaining_workers,headers:result.headers,cdn:result.cdn});
}
