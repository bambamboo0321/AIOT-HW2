import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { ModelStore } from "../lib/custom-model-store.ts";
import { createTrainingHandler, createValidationHandler } from "../lib/custom-validation-service.ts";
import { createPredictionHandler } from "../lib/custom-prediction-service.ts";
import { parseFeatureInput, parsePredictionResult, parsePredictionModel, type PredictionModel } from "../lib/custom-prediction-types.ts";
import { parseTrainingReport } from "../lib/custom-training-result.ts";
import { emptyDraft, profileCsv, validatePlan } from "../lib/custom-dataset.ts";
const repo=resolve(dirname(fileURLToPath(import.meta.url)),"../..");const python=process.env.AEP_PYTHON??resolve(repo,".venv/bin/python");
const text="x,y,target\n"+Array.from({length:100},(_,i)=>`${Math.sin(i)},${Math.cos(i)},${20+3*Math.sin(i)+2*Math.cos(i)}`).join("\n");
const config=validatePlan(profileCsv(text),{...emptyDraft(),target:"target",features:["x","y"],split:"random",cvSplits:3}).config!;
const sha=createHash("sha256").update(text).digest("hex");
function form(){const f=new FormData();f.append("csv",new File([text],"fixture.csv"));f.append("config",JSON.stringify(config));f.append("review",JSON.stringify({schema_version:1,artifact_type:"custom_dataset_review",dataset:{sha256:sha},preprocessing:config,review:{confirmed:true,training_started:false,scope:"configuration_only"}}));return f;}
const origin="http://127.0.0.1:3000";
function req(payload:unknown,method="POST",signal?:AbortSignal){return new Request(`${origin}/api/custom-dataset/predict`,{method,headers:{Origin:origin,"Content-Type":"application/json"},body:JSON.stringify(payload),signal});}
const metadata={target:"target",features:["x","y"],missing_numeric:"median" as const,dataset_sha256:sha,config_sha256:"a".repeat(64)};
async function fixture(run:(dir:string,options:{repoRoot:string;pythonPath:string;temporaryRoot:string;store:ModelStore})=>Promise<void>){const dir=await mkdtemp(resolve(repo,"web/.test-tmp-prediction-"));const store=new ModelStore({root:dir});try{const uploads=resolve(dir,"uploads");await mkdir(uploads);await run(dir,{repoRoot:repo,pythonPath:python,temporaryRoot:uploads,store});}finally{await store.dispose();await rm(dir,{recursive:true,force:true});}}
async function trained(options:Parameters<typeof createTrainingHandler>[0],selection={method:"none",k:null as number|null}){
 const v=await (await createValidationHandler(options)(new Request(origin+"/api/custom-dataset/validate",{method:"POST",headers:{Origin:origin},body:form()}))).json();assert.equal(v.status,"valid");
 const body=form();body.append("experiment",JSON.stringify({schema_version:1,method:"linear_regression",confirmed:true,dataset_sha256:sha,config_sha256:v.config_sha256,selection}));
 const response=await createTrainingHandler({...options,retainModels:true})(new Request(origin+"/api/custom-dataset/train",{method:"POST",headers:{Origin:origin},body}));assert.equal(response.status,200);const r=parseTrainingReport(await response.json());assert.equal(r.status,"completed");if(r.status!=="completed"||!r.prediction_model)throw new Error("missing model");return r as typeof r & {prediction_model:PredictionModel};
}
test("retained real model predicts original-unit single rows and partial inputs without retraining, then deletes",async()=>fixture(async(dir,options)=>{
 const report=await trained(options);const model=report.prediction_model;assert.deepEqual(model.features,["x","y"]);assert.deepEqual(await readdir(options.temporaryRoot),[]);
 const h=createPredictionHandler(options);const row=report.preview[0];const i=row.source_row-1;
 const response=await h(req({model_id:model.id,values:{x:Math.sin(i),y:Math.cos(i)}}));assert.equal(response.status,200);assert.equal(response.headers.get("cache-control"),"private, no-store");const prediction=parsePredictionResult(await response.json(),model);assert.ok(Math.abs(prediction.prediction-row.prediction)<1e-10);assert.equal(prediction.training_started,false);
 const partial=await h(req({model_id:model.id,values:{x:.5}}));assert.equal(partial.status,200);const p=parsePredictionResult(await partial.json(),model);assert.deepEqual(p.imputed.map(v=>v.feature),["y"]);
 assert.equal((await h(req({model_id:model.id},"DELETE"))).status,204);assert.equal((await h(req({model_id:model.id,values:{x:1}}))).status,410);
 assert.deepEqual(await readdir(options.temporaryRoot),[]);assert.ok(!(await readdir(dir)).some(n=>n.startsWith("aep-model-")));
}));
test("selected model only needs selected names; rejects target, paths, unsafe input and stale handles",async()=>fixture(async(_dir,options)=>{
 const {prediction_model:model}=await trained(options,{method:"f_test",k:1});const name=model.features[0];assert.equal(model.features.length,1);const h=createPredictionHandler(options);
 assert.equal((await h(req({model_id:model.id,values:{[name]:.5}}))).status,200);
 for(const values of [{},{[name]:null},{[name]:true},{[name]:"1"},{[name]:1e20},{target:1},{other:1}])assert.equal((await h(req({model_id:model.id,values}))).status,422);
 assert.equal((await h(req({model_id:model.id,values:{[name]:1},model_path:"/secret"}))).status,400);
 assert.equal((await h(req({model_id:"../model.joblib",values:{[name]:1}}))).status,400);
 assert.equal((await h(req({model_id:"f".repeat(64),values:{[name]:1}}))).status,410);
 assert.deepEqual(await readdir(options.temporaryRoot),[]);
}));
test("prediction requires local origin, bounded JSON, honors abort/timeout and releases busy slot",async()=>fixture(async(_dir,options)=>{
 const {prediction_model:model}=await trained(options);const payload={model_id:model.id,values:{x:1,y:2}};const h=createPredictionHandler(options);
 const foreign=req(payload);foreign.headers.set("origin","https://evil.example");assert.equal((await h(foreign)).status,403);
 const missing=req(payload);missing.headers.delete("origin");assert.equal((await h(missing)).status,403);
 const cross=req(payload,"DELETE");cross.headers.set("sec-fetch-site","cross-site");assert.equal((await h(cross)).status,403);
 const wrong=req(payload);wrong.headers.set("content-type","text/plain");assert.equal((await h(wrong)).status,415);
 assert.equal((await createPredictionHandler({...options,bodyLimit:10})(req(payload))).status,413);
 assert.equal((await createPredictionHandler({...options,timeoutMs:1})(req(payload))).status,504);
 const a=new AbortController();const promise=h(req(payload,"POST",a.signal));a.abort();assert.equal((await promise).status,499);
 const first=h(req(payload));assert.equal((await h(req(payload))).status,429);assert.equal((await first).status,200);assert.deepEqual(await readdir(options.temporaryRoot),[]);
}));
test("temporary store expiry/capacity/deletion release files only after active inference ends",async()=>fixture(async(dir)=>{
 let now=100;const store=new ModelStore({root:dir,ttlMs:1000,capacity:1,now:()=>now});try{
  const source=resolve(dir,"trusted.joblib");await writeFile(source,"trusted",{mode:0o600});const one=await store.register(source,metadata);const lease=await store.acquire(one.id);assert.ok(lease);
  const two=await store.register(source,metadata);assert.equal(await store.acquire(one.id),null);assert.equal((await readFile(lease.path)).toString(),"trusted");await lease.release();await assert.rejects(()=>readFile(lease.path));
  now=two.expires_at;assert.equal(await store.acquire(two.id),null);assert.ok(!(await readdir(dir)).some(n=>n.startsWith("aep-model-")));
 }finally{await store.dispose();}
}));
test("runtime expiry timer removes a model without another request",async()=>fixture(async(dir)=>{
 const store=new ModelStore({root:dir,ttlMs:20});try{const source=resolve(dir,"source");await writeFile(source,"trusted");const model=await store.register(source,metadata);await new Promise(r=>setTimeout(r,60));assert.equal(await store.acquire(model.id),null);assert.ok(!(await readdir(dir)).some(n=>n.startsWith("aep-model-")));}finally{await store.dispose();}
}));
test("prediction parser rejects model/result mismatches and feature input never treats invalid text as missing",()=>{
 for(const s of ["NaN","Infinity","1,000","0x10","hello","1e999","9007199254740992"])assert.throws(()=>parseFeatureInput(s));
 for(const [s,n] of [["",null],["  ",null],["-2.5",-2.5],["1e3",1000],[".5",.5]] as const)assert.equal(parseFeatureInput(s),n);
 const model=parsePredictionModel({...metadata,id:"f".repeat(64),expires_at:Date.now()+1000});
 const good={schema_version:1,status:"predicted",training_started:false,...metadata,prediction:4,imputed:[],out_of_range:[],warnings:[]};assert.equal(parsePredictionResult(good,model).prediction,4);
 for(const change of [{training_started:true},{target:"other"},{features:["y","x"]},{prediction:Infinity},{dataset_sha256:"b".repeat(64)},{imputed:[{feature:"other",value:2}]},{out_of_range:["other"]}])assert.throws(()=>parsePredictionResult({...good,...change},model));
});
