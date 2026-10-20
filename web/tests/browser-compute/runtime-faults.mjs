import {readFile,writeFile} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const WEB=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
export async function run(spaceId){
 const task=await taskSpace(spaceId),page=await task.newPage();
 await page.goto('http://127.0.0.1:3017/custom-dataset');
 const fixture=JSON.parse(await readFile(WEB+'/tests/fixtures/browser-compute/cases.json','utf8')).cases[0];
 await page.evaluate(fixture=>{
  window.faultAcceptance={status:'running'};
  void(async()=>{
   const state=window.faultAcceptance;let client;
   try{
    const {BrowserComputeClient}=await import('/browser-compute/modules/lib/browser-compute/client.mjs');
    const bad=new BrowserComputeClient({workerFactory:()=>new Worker('/browser-compute/intentional-missing-worker.mjs',{type:'module'})});
    try{await bad.initialize();throw new Error('Missing worker unexpectedly initialized');}catch(e){if(e.code!=='WORKER_FAILED')throw e;state.failed_initialization={code:e.code,cleaned:!bad.busy&&!bad.initialized};}finally{bad.dispose();}
    client=new BrowserComputeClient({methodTimeoutMs:5});
    const input={csv:new Blob([fixture.csv]),config:fixture.config,review:fixture.review};
    const validation=await client.validate(input);
    try{await client.train({...input,experiment:{schema_version:1,method:'linear_regression',confirmed:true,dataset_sha256:validation.report.dataset.sha256,config_sha256:validation.report.config_sha256,selection:{method:'forward_sfs',k:5}}});throw new Error('Watchdog did not fire');}
    catch(e){if(e.code!=='TIMEOUT')throw e;state.method_timeout={code:e.code,injected_watchdog_ms:5,cleaned:!client.busy&&!client.initialized};}
    state.status='passed';
   }catch(e){state.status='failed';state.error={code:e.code,message:e.message};}
   finally{client?.dispose();}
  })();
 },fixture);
 console.log({page:page.label});return page.label;
}
export async function collect(spaceId,label){
 const task=await taskSpace(spaceId),page=task.page(label);
 await page.waitForFunction(()=>window.faultAcceptance.status!=='running',undefined,{timeout:50000});
 const result=await page.evaluate(()=>window.faultAcceptance);
 await writeFile(WEB+'/verification/m17-b5-faults.json',JSON.stringify(result,null,2)+'\n');
 console.log(result);
}
