import {test} from "node:test";
import assert from "node:assert/strict";
import {mkdtemp,mkdir,readdir,rm} from "node:fs/promises";
import {resolve,dirname} from "node:path";
import {fileURLToPath} from "node:url";
import {createHash} from "node:crypto";
import {createComparisonHandler,createValidationHandler} from "../lib/custom-validation-service.ts";
import {parseComparisonReport} from "../lib/custom-comparison-result.ts";
import {comparisonCsv,comparisonJson} from "../lib/custom-comparison-export.ts";
import {emptyDraft,profileCsv,validatePlan} from "../lib/custom-dataset.ts";
const repo=resolve(dirname(fileURLToPath(import.meta.url)),"../.."),python=process.env.AEP_PYTHON??resolve(repo,".venv/bin/python");
const text="date,x,z,w,target\n"+Array.from({length:80},(_,i)=>{const x=Math.sin(i*.7),z=Math.cos(i*1.37),w=Math.cos(i*.39);return `2024-01-${String(Math.floor(i/24)+1).padStart(2,"0")}T${String(i%24).padStart(2,"0")}:00,${x},${z},${w},${4*x+.7*z}`;}).join("\n");
const config=validatePlan(profileCsv(text),{...emptyDraft(),target:"target",features:["x","z","w"],split:"chronological",splitColumn:"date",cvSplits:3}).config;
const sha=createHash("sha256").update(text).digest("hex");
function form(){const f=new FormData();f.append("csv",new File([text],"../../user.csv"));f.append("config",JSON.stringify(config));f.append("review",JSON.stringify({schema_version:1,artifact_type:"custom_dataset_review",dataset:{sha256:sha},preprocessing:config,review:{confirmed:true,training_started:false,scope:"configuration_only"}}));return f;}
const request=(f:FormData,signal?:AbortSignal)=>new Request("http://127.0.0.1:3000/api/custom-dataset/compare",{method:"POST",body:f,headers:{Origin:"http://127.0.0.1:3000"},signal});
async function fixture(run:(options:{repoRoot:string;pythonPath:string;temporaryRoot:string},make:(change?:Record<string,unknown>)=>FormData)=>Promise<void>){
  const root=await mkdtemp(resolve(repo,"web/.test-tmp-comparison-"));try{const uploads=resolve(root,"uploads");await mkdir(uploads);const options={repoRoot:repo,pythonPath:python,temporaryRoot:uploads};const validation=await(await createValidationHandler(options)(request(form()))).json();assert.equal(validation.status,"valid");const make=(change:Record<string,unknown>={})=>{const f=form();f.append("experiment",JSON.stringify({schema_version:1,confirmed:true,dataset_sha256:sha,config_sha256:validation.config_sha256,methods:["f_test","mutual_information","rfe"],k_values:[1,2,3],...change}));return f;};await run(options,make);assert.deepEqual(await readdir(uploads),[]);}finally{await rm(root,{recursive:true,force:true});}
}
test("real comparison produces complete fixed-k Training CV curves without Holdout or saved model",async()=>fixture(async(options,make)=>{
  const response=await createComparisonHandler(options)(request(make()));assert.equal(response.status,200);assert.equal(response.headers.get("cache-control"),"private, no-store");const report=parseComparisonReport(await response.json());assert.equal(report.status,"compared");if(report.status!=="compared")return;
  assert.equal(report.points.length,9);assert.ok(report.points.every(p=>p.folds.length===3&&p.folds.every(f=>f.features.length===p.k)));assert.ok(!("holdout" in report));assert.ok(!("prediction_model" in report));
  const full=report.points.filter(p=>p.k===3);assert.deepEqual(full[0].folds,full[1].folds);assert.deepEqual(full[0].folds,full[2].folds);
}));
test("comparison rejects confirmation, stale binding, auto methods and invalid grids before calculation",async()=>fixture(async(options,make)=>{
  const h=createComparisonHandler(options);assert.equal((await h(request(form()))).status,400);
  for(const change of [{confirmed:false},{config_sha256:"0".repeat(64)},{methods:["rfecv"]},{k_values:[2,1]},{k_values:[1.5]},{output_path:"/private"}]){const response=await h(request(make(change)));assert.equal(response.status,422);const report=parseComparisonReport(await response.json());assert.equal(report.status,"rejected");assert.equal(report.comparison_started,false);}
}));
test("comparison preserves local-origin, size and exact-form-field protections",async()=>fixture(async(options,make)=>{
  const h=createComparisonHandler(options),foreign=request(make());foreign.headers.set("Origin","https://evil.example");assert.equal((await h(foreign)).status,403);
  assert.equal((await createComparisonHandler({...options,bodyLimit:1})(request(make()))).status,413);
  const f=make();f.append("module","arbitrary");assert.equal((await h(request(f))).status,400);
}));
test("comparison timeout, abort and duplicate requests clean uploads and release occupancy",async()=>fixture(async(options,make)=>{
  assert.equal((await createComparisonHandler({...options,timeoutMs:1})(request(make()))).status,504);
  const h=createComparisonHandler(options),abort=new AbortController(),pending=h(request(make(),abort.signal));abort.abort();assert.equal((await pending).status,499);
  const first=h(request(make()));assert.equal((await h(request(make()))).status,429);assert.equal((await first).status,200);
}));
test("comparison parser rejects incomplete grid, forged folds, names, metrics and summaries",async()=>fixture(async(options,make)=>{
  const value=await(await createComparisonHandler(options)(request(make()))).json();assert.equal(value.status,"compared");
  for(const mutate of [(r:typeof value)=>{r.points.pop();},(r:typeof value)=>{r.k_values=[1,1,3];},(r:typeof value)=>{r.methods.reverse();},(r:typeof value)=>{r.scaling="none";},(r:typeof value)=>{r.points[0].k=1.5;},(r:typeof value)=>{r.points[0].folds[0].training_rows++;},(r:typeof value)=>{r.points[0].folds[0].features=["bad"];},(r:typeof value)=>{r.points[0].folds[0].scores.r2=2;},(r:typeof value)=>{r.points[0].metrics.rmse.mean++;},(r:typeof value)=>{r.points[0].metrics.r2.defined_folds=0;},(r:typeof value)=>{r.points[0].ranking_performed=false;}]){const copy=structuredClone(value);mutate(copy);assert.throws(()=>parseComparisonReport(copy));}
  const parsed=parseComparisonReport({...value,holdout:{secret:true},prediction_model:{id:"secret"}});assert.ok(!("holdout" in parsed));assert.ok(!("prediction_model" in parsed));
}));
test("comparison exports full fold scores and selected names, null R2 stays blank and unknown fields are stripped",async()=>fixture(async(options,make)=>{
  const value=await(await createComparisonHandler(options)(request(make()))).json();const report=parseComparisonReport(value);if(report.status!=="compared")assert.fail("missing report");
  const csv=comparisonCsv(report),json=comparisonJson(report,{...config});assert.equal(csv.content.split("\r\n").filter(Boolean).length,1+9*3*4);const saved=JSON.parse(json.content);assert.equal(saved.points.length,9);assert.deepEqual(saved.preprocessing,config);assert.ok(!("holdout" in saved));assert.ok(!json.content.includes("prediction_model"));
  const n=structuredClone(report);n.points[0].folds.forEach(f=>{f.scores.r2=null;});n.points[0].metrics.r2={mean:null,std:null,defined_folds:0};assert.ok(comparisonCsv(n).content.includes('"r2",,,0,3'));
  const malicious=structuredClone(report);malicious.target="=1+1";assert.ok(comparisonCsv(malicious).content.includes('"\'=1+1"'));
  assert.throws(()=>comparisonJson(report,{...config,numerical_scaling:"none"}));
}));
