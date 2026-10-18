import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp,mkdir,readdir,rm } from "node:fs/promises";
import { resolve,dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { createTrainingHandler,createValidationHandler } from "../lib/custom-validation-service.ts";
import { parseTrainingReport } from "../lib/custom-training-result.ts";
import { emptyDraft,profileCsv,validatePlan } from "../lib/custom-dataset.ts";
const repo=resolve(dirname(fileURLToPath(import.meta.url)),"../..");
const text="date,x,z,w,target\n"+Array.from({length:80},(_,i)=>{const x=Math.sin(i*.7),z=Math.cos(i*1.37),w=Math.cos(i*.39);return `2024-01-${String(Math.floor(i/24)+1).padStart(2,"0")}T${String(i%24).padStart(2,"0")}:00,${x},${z},${w},${4*x+.7*z}`;}).join("\n");
const config=validatePlan(profileCsv(text),{...emptyDraft(),target:"target",features:["x","z","w"],split:"chronological",splitColumn:"date",cvSplits:3}).config;
const sha=createHash("sha256").update(text).digest("hex");
function form(){const f=new FormData();f.append("csv",new File([text],"selection.csv"));f.append("config",JSON.stringify(config));f.append("review",JSON.stringify({schema_version:1,artifact_type:"custom_dataset_review",dataset:{sha256:sha},preprocessing:config,review:{confirmed:true,training_started:false,scope:"configuration_only"}}));return f;}
const request=(f:FormData)=>new Request("http://127.0.0.1:3000/api/custom-dataset/train",{method:"POST",body:f,headers:{Origin:"http://127.0.0.1:3000"}});
async function fixture(run:(h:ReturnType<typeof createTrainingHandler>,make:(selection:unknown)=>FormData,uploads:string)=>Promise<void>){const dir=await mkdtemp(resolve(repo,"web/.test-tmp-selection-"));try{const uploads=resolve(dir,"uploads");await mkdir(uploads);const opts={repoRoot:repo,pythonPath:process.env.AEP_PYTHON??resolve(repo,".venv/bin/python"),temporaryRoot:uploads};const validation=await (await createValidationHandler(opts)(request(form()))).json();assert.equal(validation.status,"valid");const make=(selection:unknown)=>{const f=form();f.append("experiment",JSON.stringify({schema_version:1,method:"linear_regression",confirmed:true,dataset_sha256:sha,config_sha256:validation.config_sha256,selection}));return f;};await run(createTrainingHandler(opts),make,uploads);}finally{await rm(dir,{recursive:true,force:true});}}
test("F-test and MI requests report actual final/fold names and three matching evaluation populations",async()=>fixture(async(h,make,uploads)=>{
  let reference:unknown;
  for(const method of ["f_test","mutual_information"]){const r=await h(request(make({method,k:2})));assert.equal(r.status,200);const v=parseTrainingReport(await r.json());assert.equal(v.status,"completed");if(v.status!=="completed")return;
    assert.equal(v.selection.method,method);assert.equal(v.selection.requested_k,2);assert.equal(v.selection.ranking_performed,true);assert.equal(v.selection.final_features.length,2);assert.equal(v.selection.fold_features.length,3);assert.ok(v.selection.fold_features.every(f=>f.features.length===2));
    assert.equal(v.holdout.rows,16);assert.equal(v.preview.length,16);assert.ok(v.preview.every(p=>Number.isFinite(p.baseline_prediction)));
    if(reference)assert.deepEqual(v.holdout.all_features,reference);else reference=v.holdout.all_features;
  }
  assert.deepEqual(await readdir(uploads),[]);
}));
test("HTTP selection rejects invalid k, unimplemented algorithms and extra selector execution options",async()=>fixture(async(h,make,uploads)=>{
  for(const selection of [{method:"f_test",k:4},{method:"rfecv",k:1},{method:"f_test",k:1,command:"bad"}]){const r=await h(request(make(selection)));assert.equal(r.status,422);const v=parseTrainingReport(await r.json());assert.equal(v.status,"rejected");assert.equal(v.training_started,false);}
  const outside=make({method:"f_test",k:1});outside.append("selection","outside");assert.equal((await h(request(outside))).status,400);assert.deepEqual(await readdir(uploads),[]);
}));
test("runtime schema rejects forged selected names, k, folds, ranking flag and comparator output",async()=>fixture(async(h,make)=>{
  const report=await (await h(request(make({method:"f_test",k:2})))).json();assert.equal(report.status,"completed");
  const mutations=[(v:typeof report)=>{v.selection.final_features[0]="unknown";},(v:typeof report)=>{v.selection.requested_k=1;},(v:typeof report)=>{v.selection.fold_features[0].features.pop();},(v:typeof report)=>{v.selection.fold_features[0].fold=3;},(v:typeof report)=>{v.selection.ranking_performed=false;},(v:typeof report)=>{v.selection.method="rfecv";},(v:typeof report)=>{v.holdout.all_features.r2=2;},(v:typeof report)=>{v.preview[0].baseline_prediction=Infinity;}];
  for(const mutate of mutations){const v=structuredClone(report);mutate(v);assert.throws(()=>parseTrainingReport(v));}
  const v=parseTrainingReport({...report,execution_path:"never forward"});assert.ok(!("execution_path" in v));
}));
