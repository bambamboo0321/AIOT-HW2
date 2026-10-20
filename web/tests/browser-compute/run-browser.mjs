import {readFile,writeFile} from "node:fs/promises";
import {resolve,dirname} from "node:path";
import {fileURLToPath} from "node:url";
const WEB=resolve(dirname(fileURLToPath(import.meta.url)),"../..");
export async function start(spaceId,origin="http://127.0.0.1:3017"){
 const task=await taskSpace(spaceId),page=task.page("p1");
 await page.goto(origin+"/custom-dataset");
 const fixtures=JSON.parse(await readFile(resolve(WEB,"tests/fixtures/browser-compute/cases.json"),"utf8"));
 await page.evaluate(fixtures=>{
  window.browserAcceptance={status:"running",stage:"initialize",cases:[],progress:[],rtol:fixtures.rtol,atol:fixtures.atol};
  const state=window.browserAcceptance;
  void (async()=>{
   let client;
   const mismatch=[];
   const check=(value,reference,path="result")=>{
    if(typeof reference==="number") {if(typeof value!=="number"||!Number.isFinite(value)||Math.abs(value-reference)>fixtures.atol+fixtures.rtol*Math.abs(reference))mismatch.push({path,value,reference});return;}
    if(Array.isArray(reference)){if(!Array.isArray(value)||value.length!==reference.length){mismatch.push({path,error:"array shape"});return;}reference.forEach((r,i)=>check(value[i],r,path+"["+i+"]"));return;}
    if(reference&&typeof reference==="object"){for(const [k,r] of Object.entries(reference))check(value?.[k],r,path+"."+k);return;}
    if(value!==reference)mismatch.push({path,value,reference});
   };
   const require=(condition,message)=>{if(!condition)throw new Error(message);};
   try{
    const {BrowserComputeClient}=await import("/browser-compute/modules/lib/browser-compute/client.mjs");
    const create=onProgress=>new BrowserComputeClient({onProgress:p=>{state.progress.push({...p,at_ms:performance.now()});onProgress?.(p);}});
    client=create();window.acceptanceClient=client;
    const initStart=performance.now();state.initialization=await client.initialize();state.initialization_ms=performance.now()-initStart;
    state.worker_headers=Object.fromEntries((await fetch('/browser-compute/worker.mjs',{method:'HEAD'})).headers);
    state.document_csp=document.querySelector('meta[http-equiv="Content-Security-Policy"]')?.content??null;
    let predictions=0,maxDifference=0;
    for(const fixture of fixtures.cases){
     state.stage=fixture.name+":validate";
     const input={csv:new Blob([fixture.csv]),config:fixture.config,review:fixture.review};
     const validation=await client.validate(input);check(validation.report,fixture.validation,fixture.name+".validation");
     const result={name:fixture.name,methods:[],comparison:null};state.cases.push(result);
     for(const reference of fixture.references){
      const method=reference.selection.method;state.stage=fixture.name+":"+method;
      const experiment={schema_version:1,method:"linear_regression",confirmed:true,dataset_sha256:validation.report.dataset.sha256,config_sha256:validation.report.config_sha256,selection:{method,k:reference.selection.requested_k}};
      const trained=await client.train({...input,experiment});check(trained.report,reference,fixture.name+"."+method);
      require(trained.report.status==="completed","Training did not complete");
      const report=trained.report,model=report.prediction_model;require(!!model,"MEMFS model missing");
      predictions+=report.preview.length;
      report.preview.forEach((p,i)=>maxDifference=Math.max(maxDifference,Math.abs(p.prediction-reference.preview[i].prediction)));
      const lines=fixture.csv.trim().split('\n').map(l=>l.split(',')),headers=lines[0],raw=lines[report.preview[0].source_row];
      const full=Object.fromEntries(model.features.map(n=>[n,raw[headers.indexOf(n)]===""?null:Number(raw[headers.indexOf(n)])]));
      const single=await client.predict(model.id,full);check(single.report.prediction,report.preview[0].prediction,"inference.full");
      let partialPassed=false;
      const expected=fixture.inference?.[method];
      if(expected){check((await client.predict(model.id,expected.values)).report,expected.single,"inference.partial");check((await client.predictBatch(model.id,expected.rows)).report,expected.batch,"inference.batch");partialPassed=true;}
      else {const batch=await client.predictBatch(model.id,[model.features.map(n=>full[n])]);check(batch.report.rows[0].prediction,report.preview[0].prediction,"inference.batch");}
      await client.deleteModel(model.id);let deleted=false;try{await client.predict(model.id,full);}catch(e){deleted=e.code==="MODEL_EXPIRED";}require(deleted,"Deleted model remained usable");
      result.methods.push({method,k:report.selection.final_features.length,selected_features:report.selection.final_features,metrics:report.holdout.linear_regression,preview_predictions:report.preview.length,elapsed_ms:trained.metadata.elapsed_ms,single:true,batch:true,partial:partialPassed,delete:true});
     }
     state.stage=fixture.name+":comparison";
     const compared=await client.compare({...input,experiment:fixture.comparison_experiment});check(compared.report,fixture.comparison_reference,fixture.name+".comparison");
     result.comparison={status:compared.report.status,points:compared.report.points.length,automatic:compared.report.automatic.length,elapsed_ms:compared.metadata.elapsed_ms};
     let stale=false;try{await client.train({...input,config:{...fixture.config,test_size:.3},experiment:{confirmed:true}});}catch(e){stale=e.code==="STALE";}require(stale,"Stale config was accepted");
    }
    client.dispose();client=null;
    state.stage="real-cancel";
    const base=fixtures.cases[0],input={csv:new Blob([base.csv]),config:base.config,review:base.review};
    client=create(p=>{if(p.phase==="started"&&p.method==="forward_sfs")client.cancel();});
    const validation=await client.validate(input);let cancelled=false;
    try{await client.train({...input,experiment:{schema_version:1,method:"linear_regression",confirmed:true,dataset_sha256:validation.report.dataset.sha256,config_sha256:validation.report.config_sha256,selection:{method:"forward_sfs",k:5}}});}catch(e){cancelled=e.code==="ABORTED";}
    require(cancelled&&!client.initialized&&!client.busy,"Real Worker cancel failed");client.dispose();client=null;
    state.real_cancel=true;state.predictions=predictions;state.baseline_predictions=state.cases[0].methods.reduce((n,m)=>n+m.preview_predictions,0);state.max_prediction_difference=maxDifference;state.mismatches=mismatch;
    state.status=mismatch.length?"failed":"passed";state.stage="complete";
   }catch(e){state.status="failed";state.error={message:e.message,code:e.code};state.mismatches=mismatch;}
   finally{client?.dispose();state.finished_at=new Date().toISOString();}
  })();
 },fixtures);
 console.log({started:true,spaceId,origin});
}
export async function collect(spaceId){
 const task=await taskSpace(spaceId),page=task.page("p1");
 const result=await page.evaluate(()=>window.browserAcceptance);
 if(result.status==='running'){console.log({status:result.status,stage:result.stage,methods:result.cases.map(c=>[c.name,c.methods.length])});return result;}
 result.remaining_workers=(await task.cdp('Target.getTargets')).targetInfos.filter(t=>t.type==='worker'&&t.url.includes('127.0.0.1:3017/browser-compute/')).length;
 await writeFile(resolve(WEB,'verification/m17-b5-browser.json'),JSON.stringify(result,null,2)+'\n');
 console.log({status:result.status,stage:result.stage,error:result.error,mismatches:result.mismatches?.slice(0,5),cases:result.cases.map(c=>({name:c.name,methods:c.methods.length,comparison:c.comparison})),predictions:result.predictions,baseline_predictions:result.baseline_predictions,max_prediction_difference:result.max_prediction_difference,remaining_workers:result.remaining_workers});
 return result;
}
