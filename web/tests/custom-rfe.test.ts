import {test,before,after} from "node:test";
import assert from "node:assert/strict";
import {mkdtemp,mkdir,readdir,rm} from "node:fs/promises";
import {resolve,dirname} from "node:path";
import {fileURLToPath} from "node:url";
import {createHash} from "node:crypto";
import {createTrainingHandler,createComparisonHandler,createValidationHandler} from "../lib/custom-validation-service.ts";
import {createPredictionHandler} from "../lib/custom-prediction-service.ts";
import {ModelStore} from "../lib/custom-model-store.ts";
import {parseTrainingReport,type TrainingReport} from "../lib/custom-training-result.ts";
import {parseComparisonReport} from "../lib/custom-comparison-result.ts";
import {trainingSelectionError} from "../lib/custom-training-choice.ts";
import {evaluationJson,evaluationCsv} from "../lib/custom-result-export.ts";
import {emptyDraft,profileCsv,validatePlan,parseCsv} from "../lib/custom-dataset.ts";
const repo=resolve(dirname(fileURLToPath(import.meta.url)),"../.."),origin="http://127.0.0.1:3000";
const text="x,z,w,target\n"+Array.from({length:120},(_,i)=>{const x=Math.sin(i*.7),z=Math.cos(i*1.37),w=Math.cos(i*.39);return `${x},${z},${w},${10+4*x+.7*z}`;}).join("\n");
const config=validatePlan(profileCsv(text),{...emptyDraft(),target:"target",features:["x","z","w"],split:"random",cvSplits:3}).config;
const sha=createHash("sha256").update(text).digest("hex");
let root:string,store:ModelStore,options:{repoRoot:string;pythonPath:string;temporaryRoot:string;store:ModelStore},validation:Extract<TrainingReport,{status:"completed"}>['validation'],report:Extract<TrainingReport,{status:"completed"}>;
function request(body:FormData){return new Request(origin+"/api/custom-dataset/train",{method:"POST",headers:{Origin:origin},body});}
function form(plan=config){const f=new FormData();f.append("csv",new File([text],"rfe.csv"));f.append("config",JSON.stringify(plan));f.append("review",JSON.stringify({schema_version:1,artifact_type:"custom_dataset_review",dataset:{sha256:sha},preprocessing:plan,review:{confirmed:true,training_started:false,scope:"configuration_only"}}));return f;}
function confirmed(selection={method:"rfe",k:2}){const f=form();f.append("experiment",JSON.stringify({schema_version:1,method:"linear_regression",confirmed:true,dataset_sha256:sha,config_sha256:validation.config_sha256,selection}));return f;}
function jsonRequest(payload:unknown,method="POST"){return new Request(origin+"/api/custom-dataset/predict",{method,headers:{Origin:origin,"Content-Type":"application/json"},body:JSON.stringify(payload)});}
before(async()=>{root=await mkdtemp(resolve(repo,"web/.test-tmp-rfe-"));store=new ModelStore({root});await mkdir(resolve(root,"uploads"));options={repoRoot:repo,pythonPath:process.env.AEP_PYTHON??resolve(repo,".venv/bin/python"),temporaryRoot:resolve(root,"uploads"),store};validation=await(await createValidationHandler(options)(request(form()))).json();assert.equal(validation.status,"valid");const response=await createTrainingHandler({...options,retainModels:true})(request(confirmed()));assert.equal(response.status,200);const r=parseTrainingReport(await response.json());if(r.status!=="completed")throw new Error("RFE failed");report=r;});
after(async()=>{if(store)await store.dispose();if(root)await rm(root,{recursive:true,force:true});});
test("RFE formal training matches real comparison CV scores and fold features without using final features to backfill folds",async()=>{
  const body=form();body.append("experiment",JSON.stringify({schema_version:1,confirmed:true,dataset_sha256:sha,config_sha256:validation.config_sha256,methods:["rfe"],k_values:[2]}));const response=await createComparisonHandler(options)(request(body));assert.equal(response.status,200);const r=parseComparisonReport(await response.json());if(r.status!=="compared")assert.fail("comparison missing");assert.equal(report.selection.method,"rfe");assert.equal(report.selection.requested_k,2);assert.ok(report.selection.ranking_performed);assert.equal(report.selection.final_features.length,2);
  r.points[0].folds.forEach((f,i)=>{assert.deepEqual(f.scores,report.cv.folds[i].linear_regression);assert.deepEqual(f.features,report.selection.fold_features[i].features);});assert.deepEqual(await readdir(options.temporaryRoot),[]);
});
test("retained RFE model supports selected-only single and batch inference and evaluation exports",async()=>{
  const model=report.prediction_model;if(!model)assert.fail("missing model");const p=report.preview[0],i=p.source_row-1,raw={x:Math.sin(i*.7),z:Math.cos(i*1.37),w:Math.cos(i*.39)};const values=Object.fromEntries(model.features.map(n=>[n,raw[n as keyof typeof raw]]));const response=await createPredictionHandler(options)(jsonRequest({model_id:model.id,values}));assert.equal(response.status,200);const r=await response.json();assert.ok(Math.abs(r.prediction-p.prediction)<1e-10);
  const matrix=[model.features.map(n=>values[n]),model.features.map((n,i)=>i===0?values[n]:null)];const batch=await createPredictionHandler({...options,batch:true})(jsonRequest({model_id:model.id,features:model.features,rows:matrix}));assert.equal(batch.status,200);const b=await batch.json();assert.ok(Math.abs(b.rows[0].prediction-p.prediction)<1e-10);assert.deepEqual(b.rows[1].imputed_features,model.features.slice(1));assert.equal(b.training_started,false);
  const artifact=evaluationJson(report,config),saved=JSON.parse(artifact.content);assert.equal(saved.selection.method,"rfe");assert.deepEqual(saved.selection.final_features,model.features);assert.ok(!artifact.content.includes(model.id));const csv=parseCsv(evaluationCsv(report).content);const method=csv.headers.indexOf("experiment_selection_method");assert.ok(csv.rows.every(row=>row[method]==="rfe"));assert.deepEqual(await readdir(options.temporaryRoot),[]);
});
test("RFE requires explicit standard scaling and still rejects unknown algorithms and execution parameters",async()=>{
  const plan={...config,numerical_scaling:"none" as const};const v=await(await createValidationHandler(options)(request(form(plan)))).json();assert.equal(v.status,"valid");const f=form(plan);f.append("experiment",JSON.stringify({schema_version:1,method:"linear_regression",confirmed:true,dataset_sha256:sha,config_sha256:v.config_sha256,selection:{method:"rfe",k:2}}));const response=await createTrainingHandler(options)(request(f));assert.equal(response.status,422);const r=parseTrainingReport(await response.json());assert.equal(r.status,"rejected");assert.equal(r.training_started,false);
  for(const selection of [{method:"rfecv",k:2},{method:"rfe",k:1.5},{method:"rfe",k:2,output_path:"/private"}]){const f=confirmed();const e=JSON.parse(String(f.get("experiment")));e.selection=selection;f.set("experiment",JSON.stringify(e));assert.equal((await createTrainingHandler(options)(request(f))).status,422);}
});
test("training choice checks fixed integers and RFE scaling, row, feature and computation limits before applying settings",()=>{
  assert.equal(trainingSelectionError("rfe",2,config,validation),null);assert.equal(trainingSelectionError("f_test",2,config,validation),null);assert.equal(trainingSelectionError("none",null,config,validation),null);
  for(const k of [null,0,1.5,4,Infinity])assert.ok(trainingSelectionError("rfe",k,config,validation));
  assert.ok(trainingSelectionError("rfe",2,{...config,numerical_scaling:"none"},validation));assert.ok(trainingSelectionError("rfe",2,config,{...validation,dataset:{...validation.dataset,rows:5001}}));assert.ok(trainingSelectionError("rfe",2,{...config,feature_columns:Array.from({length:26},(_,i)=>`x${i}`)},validation));
  const wide={...config,feature_columns:Array.from({length:25},(_,i)=>`x${i}`)};const large={...validation,dataset:{...validation.dataset,rows:5000},split:{...validation.split,training_rows:4000,cv_folds:Array.from({length:10},(_,i)=>({fold:i+1,training_rows:3600,validation_rows:400}))}};assert.equal(trainingSelectionError("rfe",1,wide,large),null);assert.equal(trainingSelectionError("rfe",25,wide,large),null);const excessive={...large,split:{...large.split,cv_folds:large.split.cv_folds.map(f=>({...f,training_rows:20_000}))}};assert.ok(trainingSelectionError("rfe",1,wide,excessive));
});
