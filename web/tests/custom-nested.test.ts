import {test,before,after} from "node:test";
import assert from "node:assert/strict";
import {mkdtemp,mkdir,readdir,rm} from "node:fs/promises";
import {resolve,dirname} from "node:path";
import {fileURLToPath} from "node:url";
import {createHash} from "node:crypto";
import {createTrainingHandler,createComparisonHandler,createValidationHandler} from "../lib/custom-validation-service.ts";
import {createPredictionHandler} from "../lib/custom-prediction-service.ts";
import {ModelStore} from "../lib/custom-model-store.ts";
import {parseTrainingReport,type TrainingReport,type SelectionMethod} from "../lib/custom-training-result.ts";
import {parseComparisonReport,type Compared} from "../lib/custom-comparison-result.ts";
import {trainingSelectionError} from "../lib/custom-training-choice.ts";
import {ALPHA_GRID} from "../lib/custom-selection-policy.ts";
import {comparisonCsv,comparisonJson} from "../lib/custom-comparison-export.ts";
import {evaluationCsv,evaluationJson} from "../lib/custom-result-export.ts";
import {emptyDraft,profileCsv,validatePlan,parseCsv} from "../lib/custom-dataset.ts";
const repo=resolve(dirname(fileURLToPath(import.meta.url)),"../.."),origin="http://127.0.0.1:3000";
const text="x,z,w,target\n"+Array.from({length:96},(_,i)=>{const x=Math.sin(i*.7),z=Math.cos(i*1.37),w=Math.cos(i*.39);return `${x},${z},${w},${10+4*x+.7*z}`;}).join("\n");
const config=validatePlan(profileCsv(text),{...emptyDraft(),target:"target",features:["x","z","w"],split:"random",cvSplits:3}).config;
const sha=createHash("sha256").update(text).digest("hex");
type Completed=Extract<TrainingReport,{status:"completed"}>;
let root:string,store:ModelStore,options:{repoRoot:string;pythonPath:string;temporaryRoot:string;store:ModelStore},validation:Completed['validation'],curve:Compared;
const reports=new Map<SelectionMethod,Completed>();
function form(){const f=new FormData();f.append("csv",new File([text],"nested.csv"));f.append("config",JSON.stringify(config));f.append("review",JSON.stringify({schema_version:1,artifact_type:"custom_dataset_review",dataset:{sha256:sha},preprocessing:config,review:{confirmed:true,training_started:false,scope:"configuration_only"}}));return f;}
function req(body:FormData){return new Request(origin+"/api/custom-dataset/train",{method:"POST",headers:{Origin:origin},body});}
function experiment(method:SelectionMethod,k:number|null){const f=form();f.append("experiment",JSON.stringify({schema_version:1,method:"linear_regression",confirmed:true,dataset_sha256:sha,config_sha256:validation.config_sha256,selection:{method,k}}));return f;}
function json(payload:unknown){return new Request(origin+"/api/custom-dataset/predict",{method:"POST",headers:{Origin:origin,"Content-Type":"application/json"},body:JSON.stringify(payload)});}
before(async()=>{root=await mkdtemp(resolve(repo,"web/.test-tmp-nested-"));store=new ModelStore({root});await mkdir(resolve(root,"uploads"));options={repoRoot:repo,pythonPath:process.env.AEP_PYTHON??resolve(repo,".venv/bin/python"),temporaryRoot:resolve(root,"uploads"),store};validation=await(await createValidationHandler(options)(req(form()))).json();assert.equal(validation.status,"valid");
 for(const method of ["forward_sfs","rfecv","lasso_cv"] as const){const response=await createTrainingHandler({...options,retainModels:true})(req(experiment(method,method==="forward_sfs"?2:null)));assert.equal(response.status,200);const r=parseTrainingReport(await response.json());if(r.status!=="completed")throw new Error("nested training missing");reports.set(method,r);}
 const f=form();f.append("experiment",JSON.stringify({schema_version:1,confirmed:true,dataset_sha256:sha,config_sha256:validation.config_sha256,methods:["forward_sfs","rfecv","lasso_cv"],k_values:[1,2,3]}));const response=await createComparisonHandler(options)(req(f));assert.equal(response.status,200);const c=parseComparisonReport(await response.json());if(c.status!=="compared")throw new Error("comparison missing");curve=c;
});
after(async()=>{if(store)await store.dispose();if(root)await rm(root,{recursive:true,force:true});});
test("all three nested selectors train and return fold-local k/alpha with separate fixed and automatic comparison",async()=>{
 assert.equal(curve.points.length,3);assert.equal(curve.automatic.length,2);assert.ok(curve.points.every(p=>p.method==="forward_sfs"));
 for(const [method,r] of reports){assert.equal(r.selection.method,method);assert.equal(r.selection.requested_k,method==="forward_sfs"?2:null);assert.ok(r.selection.details);assert.equal(r.selection.details.inner_folds.length,3);if(method==="lasso_cv")assert.ok(ALPHA_GRID.includes(r.selection.details.alpha!));
 const comparison=method==="forward_sfs"?curve.points.find(p=>p.k===2)!:curve.automatic.find(p=>p.method===method)!;comparison.folds.forEach((f,i)=>{assert.deepEqual(f.features,r.selection.fold_features[i].features);for(const metric of ["rmse","mae","r2"] as const){const a=f.scores?.[metric],b=r.cv.folds[i].linear_regression[metric];assert.ok(a!=null&&b!=null&&Math.abs(a-b)<1e-9);}});}
 assert.deepEqual(await readdir(options.temporaryRoot),[]);
});
test("all retained nested models support single and batch selected-feature inference",async()=>{
 for(const r of reports.values()){const model=r.prediction_model;if(!model)assert.fail("missing model");const row=r.preview[0],i=row.source_row-1,values={x:Math.sin(i*.7),z:Math.cos(i*1.37),w:Math.cos(i*.39)};const selected=Object.fromEntries(model.features.map(n=>[n,values[n as keyof typeof values]]));const single=await createPredictionHandler(options)(json({model_id:model.id,values:selected}));assert.equal(single.status,200);assert.ok(Math.abs((await single.json()).prediction-row.prediction)<1e-9);const batch=await createPredictionHandler({...options,batch:true})(json({model_id:model.id,features:model.features,rows:[model.features.map(n=>selected[n])]}));assert.equal(batch.status,200);assert.ok(Math.abs((await batch.json()).rows[0].prediction-row.prediction)<1e-9);}
});
test("nested report parser rejects forced auto k, missing inner details, invalid alpha and forged zero-feature scores",()=>{
 const lasso=reports.get("lasso_cv")!;
 for(const mutate of [(r:Completed)=>{r.selection.requested_k=2;},(r:Completed)=>{delete r.selection.details;},(r:Completed)=>{r.selection.details!.alpha=123;},(r:Completed)=>{r.selection.fold_features[0].details!.inner_folds.pop();},(r:Completed)=>{r.selection.final_features=[];}]){const r=structuredClone(lasso);mutate(r);assert.throws(()=>parseTrainingReport(r));}
 const forged=structuredClone(curve);forged.automatic[0].folds[0].features=[];assert.throws(()=>parseComparisonReport(forged));
 const mismatch=structuredClone(curve);mismatch.automatic[1].metrics.rmse.defined_folds--;assert.throws(()=>parseComparisonReport(mismatch));
});
test("downloads preserve actual automatic k and alpha without fake curves or model handles",()=>{
 const parsed=JSON.parse(comparisonJson(curve,config).content);assert.deepEqual(parsed.automatic,curve.automatic);const csv=parseCsv(comparisonCsv(curve).content);const records=csv.rows.map(row=>Object.fromEntries(csv.headers.map((h,i)=>[h,row[i]])));assert.ok(records.some(r=>r.evaluation==="automatic_training_cv_fold"&&r.method==="lasso_cv"&&Number(r.k)>=1&&Number(r.selected_alpha)>0));assert.ok(records.filter(r=>r.evaluation==="automatic_training_cv_summary").every(r=>r.k===""));
 for(const r of reports.values()){const saved=evaluationJson(r,config);assert.deepEqual(JSON.parse(saved.content).selection,r.selection);assert.ok(!saved.content.includes(r.prediction_model!.id));const c=parseCsv(evaluationCsv(r).content);assert.ok(c.headers.includes("cv_fold_selected_k"));assert.ok(c.headers.includes("selector_alpha"));}
});
test("new selector request policy blocks forced k, unsupported scale, too-small inner folds and costly SFS before sending",()=>{
 assert.equal(trainingSelectionError("rfecv",null,config,validation),null);assert.equal(trainingSelectionError("lasso_cv",null,config,validation),null);assert.equal(trainingSelectionError("forward_sfs",2,config,validation),null);assert.ok(trainingSelectionError("rfecv",2,config,validation));assert.ok(trainingSelectionError("lasso_cv",null,{...config,numerical_scaling:"none"},validation));assert.ok(trainingSelectionError("forward_sfs",2,config,{...validation,dataset:{...validation.dataset,rows:2001}}));const small={...validation,split:{...validation.split,cv_folds:[{fold:1,training_rows:8,validation_rows:4}]}};assert.ok(trainingSelectionError("rfecv",null,{...config,split_strategy:"chronological",cv_gap:2},small));
});
test("new selector API still rejects stale and extra execution parameters and fixed k for automatic selection",async()=>{
 for(const [method,k] of [["rfecv",1],["lasso_cv",2],["forward_sfs",null]] as const){const response=await createTrainingHandler(options)(req(experiment(method,k)));assert.equal(response.status,422);assert.equal(parseTrainingReport(await response.json()).status,"rejected");}
 const f=experiment("rfecv",null),e=JSON.parse(String(f.get("experiment")));e.command="unsafe";f.set("experiment",JSON.stringify(e));assert.equal((await createTrainingHandler(options)(req(f))).status,422);assert.deepEqual(await readdir(options.temporaryRoot),[]);
});
test("zero-feature Lasso comparison keeps null scores and formal training does not retain a fallback model",async()=>{
 const zero="x,z,w,target\n"+Array.from({length:96},(_,i)=>`${Math.sin(i*.7)},${Math.cos(i*1.37)},${Math.cos(i*.39)},${100+1e-6*Math.sin(i*2.87)}`).join("\n"),digest=createHash("sha256").update(zero).digest("hex");
 function make(){const f=form();f.set("csv",new File([zero],"zero.csv"));const review=JSON.parse(String(f.get("review")));review.dataset.sha256=digest;f.set("review",JSON.stringify(review));return f;}
 const v=await(await createValidationHandler(options)(req(make()))).json();assert.equal(v.status,"valid");const f=make();f.append("experiment",JSON.stringify({schema_version:1,confirmed:true,dataset_sha256:digest,config_sha256:v.config_sha256,methods:["lasso_cv"],k_values:[]}));const response=await createComparisonHandler(options)(req(f));assert.equal(response.status,200);const parsed=parseComparisonReport(await response.json());if(parsed.status!=="compared")assert.fail("zero comparison missing");assert.ok(parsed.automatic[0].folds.every(f=>!f.features.length&&f.scores===null));assert.equal(parsed.automatic[0].metrics.rmse.mean,null);assert.equal(parsed.automatic[0].metrics.rmse.defined_folds,0);assert.ok(comparisonCsv(parsed).content.includes('"no_features"'));
 const train=make();train.append("experiment",JSON.stringify({schema_version:1,method:"linear_regression",confirmed:true,dataset_sha256:digest,config_sha256:v.config_sha256,selection:{method:"lasso_cv",k:null}}));const failed=await createTrainingHandler({...options,retainModels:true})(req(train));assert.equal(failed.status,422);const r=parseTrainingReport(await failed.json());assert.equal(r.status,"failed");assert.ok(!("prediction_model" in r));assert.deepEqual(await readdir(options.temporaryRoot),[]);
});
