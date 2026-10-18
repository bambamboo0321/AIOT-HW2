import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readdir, rm } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { createTrainingHandler, createValidationHandler } from "../lib/custom-validation-service.ts";
import { parseTrainingReport } from "../lib/custom-training-result.ts";
import { emptyDraft, profileCsv, validatePlan } from "../lib/custom-dataset.ts";
const repo=resolve(dirname(fileURLToPath(import.meta.url)),"../..");
const python=process.env.AEP_PYTHON??resolve(repo,".venv/bin/python");
const text="date,x,target\n"+Array.from({length:80},(_,i)=>`2024-01-${String(Math.floor(i/24)+1).padStart(2,"0")}T${String(i%24).padStart(2,"0")}:00,${i},${3*i+2+(i%3)*.2}`).join("\n");
const config=validatePlan(profileCsv(text),{...emptyDraft(),target:"target",features:["x"],split:"chronological",splitColumn:"date",cvSplits:3}).config;
const sha=createHash("sha256").update(text).digest("hex");
function form(){const f=new FormData();f.append("csv",new File([text],"../../user.csv"));f.append("config",JSON.stringify(config));f.append("review",JSON.stringify({schema_version:1,artifact_type:"custom_dataset_review",dataset:{sha256:sha},preprocessing:config,review:{confirmed:true,training_started:false,scope:"configuration_only"}}));return f;}
function request(f:FormData,signal?:AbortSignal){return new Request("http://127.0.0.1:3000/api/custom-dataset/train",{method:"POST",body:f,headers:{Origin:"http://127.0.0.1:3000"},signal});}
async function fixture(run:(dir:string,options:{repoRoot:string;pythonPath:string;temporaryRoot:string})=>Promise<void>){const dir=await mkdtemp(resolve(repo,"web/.test-tmp-training-"));try{await mkdir(resolve(dir,"uploads"));await run(dir,{repoRoot:repo,pythonPath:python,temporaryRoot:resolve(dir,"uploads")});}finally{await rm(dir,{recursive:true,force:true});}}
async function confirmed(options:Parameters<typeof createValidationHandler>[0]){const validation=await (await createValidationHandler(options)(request(form()))).json();assert.equal(validation.status,"valid");const f=form();f.append("experiment",JSON.stringify({schema_version:1,method:"linear_regression",confirmed:true,dataset_sha256:sha,config_sha256:validation.config_sha256}));return f;}
test("real Python training returns CV, full Holdout, bounded prediction preview and cleans uploads",async()=>fixture(async(dir,options)=>{
  const r=await createTrainingHandler(options)(request(await confirmed(options)));assert.equal(r.status,200);assert.equal(r.headers.get("cache-control"),"private, no-store");
  const report=parseTrainingReport(await r.json());assert.equal(report.status,"completed");if(report.status!=="completed")return;
  assert.equal(report.holdout.rows,16);assert.equal(report.cv.folds.length,3);assert.equal(report.preview.length,16);assert.equal(report.validation.dataset.sha256,sha);
  assert.ok(report.holdout.linear_regression.rmse<report.holdout.training_mean.rmse);
  const actual=report.preview.map(r=>r.actual),pred=report.preview.map(r=>r.prediction);const mse=actual.reduce((s,v,i)=>s+(v-pred[i])**2,0)/16;
  assert.ok(Math.abs(Math.sqrt(mse)-report.holdout.linear_regression.rmse)<1e-10);assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("training requires separate confirmation and rejects stale/extra experiment parameters",async()=>fixture(async(dir,options)=>{
  const h=createTrainingHandler(options);assert.equal((await h(request(form()))).status,400);
  const original=await confirmed(options);
  for(const mutation of [(v:Record<string,unknown>)=>{v.confirmed=false;},(v:Record<string,unknown>)=>{v.config_sha256="0".repeat(64);},(v:Record<string,unknown>)=>{v.method="rfecv";},(v:Record<string,unknown>)=>{v.output_path="/secret";}]){
    const f=form(),v=JSON.parse(String(original.get("experiment")));mutation(v);f.append("experiment",JSON.stringify(v));const r=await h(request(f));assert.equal(r.status,422);const report=parseTrainingReport(await r.json());assert.equal(report.status,"rejected");assert.equal(report.training_started,false);
  }
  assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("training inherits local-origin, bounded body and unknown-field protection",async()=>fixture(async(dir,options)=>{
  const f=await confirmed(options);const h=createTrainingHandler(options);
  const foreign=request(f);foreign.headers.set("Origin","https://evil.example");assert.equal((await h(foreign)).status,403);
  assert.equal((await createTrainingHandler({...options,bodyLimit:1})(request(f))).status,413);
  f.append("module","arbitrary");assert.equal((await h(request(f))).status,400);assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("training timeout, abort and concurrent duplicate do not retain data or lock the slot",async()=>fixture(async(dir,options)=>{
  const f=await confirmed(options);assert.equal((await createTrainingHandler({...options,timeoutMs:1})(request(f))).status,504);
  const h=createTrainingHandler(options);const a=new AbortController();const p=h(request(f,a.signal));a.abort();assert.equal((await p).status,499);
  const first=h(request(f));assert.equal((await h(request(f))).status,429);assert.equal((await first).status,200);assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("training report rejects inconsistent metrics, folds, summaries and preview instead of fabricating scores",async()=>fixture(async(_dir,options)=>{
  const response=await createTrainingHandler(options)(request(await confirmed(options)));const report=await response.json();
  for(const mutate of [(v:typeof report)=>{v.training_started=false;},(v:typeof report)=>{v.holdout.rows++;},(v:typeof report)=>{v.holdout.linear_regression.r2=2;},(v:typeof report)=>{v.cv.folds[0].training_rows++;},(v:typeof report)=>{v.cv.linear_regression.rmse.mean++;},(v:typeof report)=>{v.preview.pop();},(v:typeof report)=>{v.preview[1].source_row=v.preview[0].source_row;},(v:typeof report)=>{v.method="another";}]){const v=structuredClone(report);mutate(v);assert.throws(()=>parseTrainingReport(v));}
  assert.ok(!("model_coefficients" in parseTrainingReport({...report,model_coefficients:[1,2,3]})));
}));
