import { before, after, test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { emptyDraft, parseCsv, profileCsv, validatePlan } from "../lib/custom-dataset.ts";
import { createTrainingHandler, createValidationHandler } from "../lib/custom-validation-service.ts";
import { parseTrainingReport, type TrainingReport } from "../lib/custom-training-result.ts";
import { csvCell, evaluationJson, evaluationCsv, previewCsv, predictionJson } from "../lib/custom-result-export.ts";
import type { PredictionModel, PredictionResult } from "../lib/custom-prediction-types.ts";
const repo=resolve(dirname(fileURLToPath(import.meta.url)),"../..");const python=process.env.AEP_PYTHON??resolve(repo,".venv/bin/python");
const text="date,x,y,target\n"+Array.from({length:900},(_,i)=>`2024-01-${String(Math.floor(i/1440)+1).padStart(2,"0")}T${String(Math.floor(i/60)%24).padStart(2,"0")}:${String(i%60).padStart(2,"0")}:00,${Math.sin(i)},${Math.cos(i)},${i>=720?100:20+3*Math.sin(i)+2*Math.cos(i)}`).join("\n");
const config=validatePlan(profileCsv(text),{...emptyDraft(),target:"target",features:["x","y"],split:"chronological",splitColumn:"date",cvSplits:3}).config;
const sha=createHash("sha256").update(text).digest("hex");const origin="http://127.0.0.1:3000";let directory:string;
let selected:Extract<TrainingReport,{status:"completed"}>;let all:typeof selected;
before(async()=>{
 directory=await mkdtemp(resolve(repo,"web/.test-tmp-export-"));const uploads=resolve(directory,"uploads");await mkdir(uploads);const options={repoRoot:repo,pythonPath:python,temporaryRoot:uploads};
 function form(){const f=new FormData();f.append("csv",new File([text],"synthetic.csv"));f.append("config",JSON.stringify(config));f.append("review",JSON.stringify({schema_version:1,artifact_type:"custom_dataset_review",dataset:{sha256:sha},preprocessing:config,review:{confirmed:true,training_started:false,scope:"configuration_only"}}));return f;}
 function request(body:FormData){return new Request(origin+"/api/custom-dataset/train",{method:"POST",headers:{Origin:origin},body});}
 const validation=await(await createValidationHandler(options)(request(form()))).json();assert.equal(validation.status,"valid");
 async function train(method:string,k:number|null){const body=form();body.append("experiment",JSON.stringify({schema_version:1,method:"linear_regression",confirmed:true,dataset_sha256:sha,config_sha256:validation.config_sha256,selection:{method,k}}));const response=await createTrainingHandler(options)(request(body));assert.equal(response.status,200);const report=parseTrainingReport(await response.json());if(report.status!=="completed")throw new Error("training failed");return report;}
 selected=await train("f_test",1);all=await train("none",null);
});
after(async()=>{if(directory)await rm(directory,{recursive:true,force:true});});
function parsedRows(content:string){const p=parseCsv(content);return p.rows.map(row=>Object.fromEntries(p.headers.map((h,i)=>[h,row[i]])));}
test("evaluation JSON retains exact full metrics, fold selection and configuration, excluding raw rows and capability handles",()=>{
 const model={id:"f".repeat(64),expires_at:Date.now()+1000,target:selected.target,features:selected.selection.final_features,missing_numeric:"median" as const,dataset_sha256:sha,config_sha256:selected.validation.config_sha256};
 const r={...selected,prediction_model:model,raw_csv:"private cells",private_path:"/secret"};const before=JSON.stringify(r);
 const artifact=evaluationJson(r,{...config,command:"do not export"} as typeof config);const data=JSON.parse(artifact.content);
 assert.equal(data.artifact_type,"custom_dataset_evaluation");assert.deepEqual(data.holdout,selected.holdout);assert.deepEqual(data.cv,selected.cv);assert.deepEqual(data.selection,selected.selection);assert.deepEqual(data.preprocessing,config);
 assert.equal(data.preview.included,false);assert.equal(data.preview.available_rows,144);assert.equal(data.preview.holdout_rows,180);assert.ok(!artifact.content.includes(model.id));assert.ok(!artifact.content.includes("private cells"));assert.ok(!artifact.content.includes("/secret"));assert.ok(!artifact.content.includes("command"));assert.equal(JSON.stringify(r),before);assert.equal(data.holdout.linear_regression.r2,null);
 assert.throws(()=>evaluationJson(selected,{...config,target_column:"wrong"}));
});
test("metrics CSV keeps Holdout, CV summary and folds distinct with exact numeric precision and undefined R2",()=>{
 const artifact=evaluationCsv(selected);assert.ok(artifact.content.startsWith("\uFEFF"));assert.ok(artifact.content.endsWith("\r\n"));const rows=parsedRows(artifact.content);assert.equal(rows.length,3*(2+selected.cv.folds.length));
 const h=rows.find(r=>r.evaluation==="holdout"&&r.model==="linear_regression")!;assert.equal(h.evaluation_rows,"180");assert.equal(Number(h.rmse),selected.holdout.linear_regression.rmse);assert.equal(h.r2,"");assert.equal(h.r2_status,"undefined");assert.equal(h.dataset_sha256,sha);
 const cv=rows.find(r=>r.evaluation==="training_cv_summary"&&r.model==="linear_regression")!;assert.equal(cv.aggregation,"equal_fold_mean_std_ddof0");assert.equal(cv.evaluation_rows,"");assert.equal(Number(cv.rmse),selected.cv.linear_regression.rmse.mean);assert.equal(Number(cv.rmse_std),selected.cv.linear_regression.rmse.std);assert.equal(cv.r2_defined_folds,String(selected.cv.linear_regression.r2.defined_folds));
 const fold=rows.find(r=>r.evaluation==="training_cv_fold"&&r.model==="linear_regression"&&r.fold==="1")!;assert.equal(Number(fold.mae),selected.cv.folds[0].linear_regression.mae);assert.equal(fold.config_sha256,selected.validation.config_sha256);
 const raw=parseCsv(artifact.content);assert.ok(raw.rows.every(row=>row.length===raw.headers.length));assert.equal(parsedRows(evaluationCsv(all).content).length,2*(2+all.cv.folds.length));
});
test("preview CSV exports only saved preview with explicit counts, timestamps, original row references and exact values",()=>{
 const artifact=previewCsv(selected);const raw=parseCsv(artifact.content);const rows=parsedRows(artifact.content);assert.equal(rows.length,144);assert.ok(raw.rows.every(r=>r.length===raw.headers.length));
 assert.ok(rows.every(r=>r.scope==="holdout_preview"&&r.holdout_total_rows==="180"&&r.exported_preview_rows==="144"&&r.preview_limit==="144"));
 assert.equal(Number(rows[0].actual),selected.preview[0].actual);assert.equal(Number(rows[0].active_lr_prediction),selected.preview[0].prediction);assert.equal(rows[0].timestamp,selected.preview[0].timestamp);assert.equal(Number(rows[0].source_csv_row),selected.preview[0].source_row);
 assert.equal(rows[0].dataset_sha256,sha);assert.equal(rows[0].config_sha256,selected.validation.config_sha256);
});
test("CSV protects formula-like text while preserving negatives, comma/quote/newline escaping and Unicode",()=>{
 for(const text of ["=SUM(1,2)","+cmd","-formula","@function","  =1","\t=1","\r=1","\nline"]){assert.ok(csvCell(text).startsWith('"\''));}
 assert.equal(csvCell(-5.25),"-5.25");assert.equal(csvCell(null),"");assert.equal(csvCell('利潤,"目標"\n下一行'),'"利潤,""目標""\n下一行"');assert.throws(()=>csvCell(Infinity));
});
function predictionFixture(){const model:PredictionModel={id:"e".repeat(64),expires_at:Date.now()+1000,target:"target",features:["x","y"],missing_numeric:"median",dataset_sha256:sha,config_sha256:selected.validation.config_sha256};const result:PredictionResult={schema_version:1,status:"predicted",training_started:false,target:model.target,features:model.features,prediction:23.1234567890123,dataset_sha256:sha,config_sha256:model.config_sha256,imputed:[{feature:"y",value:0.125}],out_of_range:[],warnings:["Synthetic fixture"]};return {model,result};}
test("single-prediction export uses submitted snapshot, preserves numeric precision and excludes model handles",()=>{
 const {model,result}=predictionFixture();const values={x:1,y:null};const original=JSON.stringify({model,result,values});const artifact=predictionJson(model,result,values);const data=JSON.parse(artifact.content);
 assert.equal(data.scope,"single_row_inference");assert.equal(data.prediction,result.prediction);assert.deepEqual(data.inputs,[{feature:"x",submitted_value:1,training_imputation_value:null},{feature:"y",submitted_value:null,training_imputation_value:.125}]);assert.ok(!artifact.content.includes(model.id));assert.ok(!artifact.content.includes("expires_at"));assert.equal(JSON.stringify({model,result,values}),original);
});
test("single-prediction export rejects missing, unknown, invalid or mismatched input/model snapshots",()=>{
 const {model,result}=predictionFixture();const invalid:Record<string,number|null>[]=[{x:1},{x:1,y:null,z:2},{x:null,y:null},{x:Infinity,y:null},{x:1,y:2}];for(const values of invalid)assert.throws(()=>predictionJson(model,result,values));
 assert.throws(()=>predictionJson(model,{...result,config_sha256:"0".repeat(64)},{x:1,y:null}));
 assert.throws(()=>predictionJson({...model,missing_numeric:"none"},result,{x:1,y:null}));
});
