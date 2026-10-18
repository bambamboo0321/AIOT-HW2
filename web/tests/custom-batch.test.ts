import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readdir, rm } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { ModelStore } from "../lib/custom-model-store.ts";
import { createTrainingHandler, createValidationHandler } from "../lib/custom-validation-service.ts";
import { createPredictionHandler } from "../lib/custom-prediction-service.ts";
import { emptyDraft, profileCsv, validatePlan, parseCsv } from "../lib/custom-dataset.ts";
import { prepareBatchCsv, batchCsv } from "../lib/custom-batch-input.ts";
import { BATCH_LIMITS, validateBatchRows, parseBatchResult, type BatchResult } from "../lib/custom-batch-types.ts";
import type { PredictionModel } from "../lib/custom-prediction-types.ts";
const model:PredictionModel={id:"f".repeat(64),expires_at:Date.now()+10000,target:"profit",features:["x","y"],missing_numeric:"median",dataset_sha256:"a".repeat(64),config_sha256:"b".repeat(64)};
function result():BatchResult{return {schema_version:1,status:"batch_predicted",training_started:false,target:model.target,features:model.features,dataset_sha256:model.dataset_sha256,config_sha256:model.config_sha256,row_count:2,rows:[{position:1,prediction:-1.234567890123,imputed_features:[],out_of_range_features:["x"]},{position:2,prediction:4,imputed_features:["y"],out_of_range_features:[]}],imputation_values:[{feature:"y",value:.125}],warnings:[]};}
test("batch CSV matches exact selected headers, reorders features and ignores extras including optional target",()=>{
 const p=prepareBatchCsv('\uFEFFid,y,x,profit\nA,2,1,not a number\nB,NA,3,not used',model);assert.deepEqual(p.rows,[[1,2],[3,null]]);assert.deepEqual(p.ignored_columns,["id","profit"]);
 for(const text of ['x\n1','X,y\n1,2','x,y\n,','x,y\n1,Infinity','x,y\n1,true','x,x\n1,2','x,y\n1,"bad'])assert.throws(()=>prepareBatchCsv(text,model));
 assert.throws(()=>prepareBatchCsv('x,y\n1,NA',{...model,missing_numeric:"none"}));
});
test("batch limits reject the whole file or matrix without dropping rows",()=>{
 assert.throws(()=>prepareBatchCsv('x,y\n'+Array.from({length:501},()=>"1,2").join('\n'),model));
 assert.throws(()=>prepareBatchCsv('x,y\n'+" ".repeat(BATCH_LIMITS.bytes),model));
 const large={...model,features:Array.from({length:21},(_,i)=>`f${i}`)};assert.throws(()=>validateBatchRows(Array.from({length:500},()=>Array(21).fill(1)),large));
 for(const rows of [[],[[null,null]],[[1]],[[1,2,3]],[[1,true]],[[1,"2"]],[[1,Infinity]],[[1,2],[1,null]]]){
  const strict={...model,missing_numeric:"none" as const};assert.throws(()=>validateBatchRows(rows,strict));
 }
 assert.deepEqual(validateBatchRows([[1,2],[3,null]],model),[[1,2],[3,null]]);
});
test("batch runtime parser binds model, input count/order and exact imputation flags",()=>{
 const rows=[[1,2],[3,null]];assert.equal(parseBatchResult(result(),model,rows).rows.length,2);
 for(const mutate of [(r:BatchResult)=>{r.row_count=1;},(r:BatchResult)=>{r.training_started=true as false;},(r:BatchResult)=>{r.rows[0].position=2;},(r:BatchResult)=>{r.rows[0].prediction=Infinity;},(r:BatchResult)=>{r.rows[1].imputed_features=[];},(r:BatchResult)=>{r.rows[1].out_of_range_features=["y"];},(r:BatchResult)=>{r.imputation_values[0].feature="x";},(r:BatchResult)=>{r.features=["y","x"];},(r:BatchResult)=>{r.config_sha256="0".repeat(64);}]){const r=structuredClone(result());mutate(r);assert.throws(()=>parseBatchResult(r,model,rows));}
});
test("batch CSV download includes every row, numeric inputs and actual fills without ignored columns or capability IDs",()=>{
 const prepared={...prepareBatchCsv('x,y,secret\n1,2,private\n3,,hidden',model),input_sha256:"c".repeat(64)};const before=JSON.stringify(prepared);const csv=batchCsv(model,result(),prepared);const p=parseCsv(csv.content);assert.equal(p.rows.length,2);assert.equal(Number(p.rows[0][p.headers.indexOf('prediction')]),result().rows[0].prediction);assert.equal(p.rows[1][p.headers.indexOf('input:y')],"");assert.deepEqual(JSON.parse(p.rows[1][p.headers.indexOf('training_imputation_values')]),[{feature:"y",value:.125}]);assert.ok(!csv.content.includes(model.id));assert.ok(!csv.content.includes("private"));assert.equal(JSON.stringify(prepared),before);assert.throws(()=>batchCsv(model,result(),{...prepared,input_sha256:"bad"}));
});
const repo=resolve(dirname(fileURLToPath(import.meta.url)),"../..");const python=process.env.AEP_PYTHON??resolve(repo,".venv/bin/python");const origin="http://127.0.0.1:3000";
const text="x,y,profit\n"+Array.from({length:100},(_,i)=>`${Math.sin(i)},${Math.cos(i)},${20+3*Math.sin(i)+2*Math.cos(i)}`).join('\n');const sha=createHash('sha256').update(text).digest('hex');
function req(body:unknown,method="POST",signal?:AbortSignal){return new Request(origin+'/api/custom-dataset/predict-batch',{method,headers:{Origin:origin,"Content-Type":"application/json"},body:JSON.stringify(body),signal});}
async function fixture(run:(options:{repoRoot:string;pythonPath:string;temporaryRoot:string;store:ModelStore},model:PredictionModel)=>Promise<void>,missing:"median"|"none"="median"){
 const dir=await mkdtemp(resolve(repo,'web/.test-tmp-batch-'));const store=new ModelStore({root:dir});try{
  const uploads=resolve(dir,'uploads');await mkdir(uploads);const options={repoRoot:repo,pythonPath:python,temporaryRoot:uploads,store};const config={...validatePlan(profileCsv(text),{...emptyDraft(),target:'profit',features:['x','y'],cvSplits:3}).config,missing_numeric:missing};
  function form(){const f=new FormData();f.append('csv',new File([text],'fixture.csv'));f.append('config',JSON.stringify(config));f.append('review',JSON.stringify({schema_version:1,artifact_type:'custom_dataset_review',dataset:{sha256:sha},preprocessing:config,review:{confirmed:true,training_started:false,scope:'configuration_only'}}));return f;}
  function trainingRequest(body:FormData){return new Request(origin+'/api/custom-dataset/train',{method:'POST',headers:{Origin:origin},body});}
  const v=await(await createValidationHandler(options)(trainingRequest(form()))).json();assert.equal(v.status,'valid');const f=form();f.append('experiment',JSON.stringify({schema_version:1,method:'linear_regression',confirmed:true,dataset_sha256:sha,config_sha256:v.config_sha256}));const t=await(await createTrainingHandler({...options,retainModels:true})(trainingRequest(f))).json();assert.equal(t.status,'completed');await run(options,t.prediction_model);
 }finally{await store.dispose();await rm(dir,{recursive:true,force:true});}
}
test("real batch API matches single inference with partial input, handles large row count, rejects invalid matrices and cleans temp",async()=>fixture(async(o,m)=>{
 const h=createPredictionHandler({...o,batch:true});const rows=[[.5,null],[1,2],[-1000,-1000]];const payload={model_id:m.id,features:m.features,rows};const response=await h(req(payload));assert.equal(response.status,200);assert.equal(response.headers.get('cache-control'),'private, no-store');const r=parseBatchResult(await response.json(),m,rows);assert.deepEqual(r.rows[0].imputed_features,['y']);
 const single=createPredictionHandler(o);for(let i=0;i<rows.length;i++){const p=await(await single(req({model_id:m.id,values:{x:rows[i][0],y:rows[i][1]}}))).json();assert.ok(Math.abs(p.prediction-r.rows[i].prediction)<1e-10);}
 const many=Array.from({length:150},()=>[1,2]);const full=await h(req({...payload,rows:many}));assert.equal(full.status,200);assert.equal((await full.json()).row_count,150);
 for(const bad of [{...payload,features:['y','x']},{...payload,rows:[[1,2],[null,null]]},{...payload,rows:Array.from({length:501},()=>[1,2])},{...payload,rows:[[1,2**53]]}]){const e=await h(req(bad));assert.equal(e.status,422);assert.ok(!('rows' in await e.json()));}
 assert.equal((await h(req({...payload,model_path:'/secret'}))).status,400);assert.equal((await h(req({...payload,model_id:'e'.repeat(64)}))).status,410);assert.deepEqual(await readdir(o.temporaryRoot),[]);
}));
test("batch inherits local origin, bounded JSON, timeout/abort and concurrency protection",async()=>fixture(async(o,m)=>{
 const payload={model_id:m.id,features:m.features,rows:[[1,2]]};const h=createPredictionHandler({...o,batch:true});const foreign=req(payload);foreign.headers.set('origin','https://evil.example');assert.equal((await h(foreign)).status,403);assert.equal((await h(req(payload,'DELETE'))).status,405);
 assert.equal((await createPredictionHandler({...o,batch:true,bodyLimit:1})(req(payload))).status,413);assert.equal((await createPredictionHandler({...o,batch:true,timeoutMs:1})(req(payload))).status,504);
 const a=new AbortController();const aborted=h(req(payload,'POST',a.signal));a.abort();assert.equal((await aborted).status,499);
 const first=h(req(payload));assert.equal((await h(req(payload))).status,429);assert.equal((await first).status,200);assert.deepEqual(await readdir(o.temporaryRoot),[]);
}));
test("batch no-imputation mode rejects incomplete rows and model deletion waits for active worker lease",async()=>fixture(async(o,m)=>{
 const h=createPredictionHandler({...o,batch:true});const payload={model_id:m.id,features:m.features,rows:[[1,2]]};assert.equal((await h(req({...payload,rows:[[1,null]]}))).status,422);
 const acquire=o.store.acquire.bind(o.store);let ready:()=>void=()=>{};const acquired=new Promise<void>(resolve=>{ready=resolve;});o.store.acquire=async id=>{const lease=await acquire(id);ready();return lease;};
 const running=h(req(payload));await acquired;assert.equal((await createPredictionHandler(o)(req({model_id:m.id},'DELETE'))).status,204);assert.equal((await running).status,200);assert.equal((await h(req(payload))).status,410);assert.deepEqual(await readdir(o.temporaryRoot),[]);
},'none'));
