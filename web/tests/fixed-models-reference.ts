import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {gunzipSync} from "node:zlib";
import {createHash} from "node:crypto";
import {FIXED_DATASETS,FIXED_MODELS,predictFixedModel,verifyFixedModelJson,type FixedModel,type FixedDatasetId,type FixedModelId} from "../lib/fixed-models/core.ts";
import {fixedModelDescriptor} from "../lib/fixed-models/catalog.ts";
export const read=(p:string)=>readFileSync(new URL("../"+p,import.meta.url),"utf8");
export const manifest=JSON.parse(read("data/fixed-models/v1/manifest.json"));
export const policy=manifest.policy as {absoluteTolerance:number;relativeTolerance:number;relativeErrorReportingFloor:number};
type Inputs=Record<string,number|string|null>;
type Reference={inputs:Inputs[];savedPredictions:number[];nativePredictions:number[];extraCases:{name:string;inputs:Inputs;nativePrediction:number}[]};
export function loadReference(){
 const bytes=readFileSync(new URL("./fixtures/fixed-models/v1/holdout.json.gz",import.meta.url));
 assert.equal(createHash("sha256").update(bytes).digest("hex"),manifest.fixtureSha256);
 const fixture=JSON.parse(gunzipSync(bytes).toString()) as {schemaVersion:number;policy:unknown;datasets:Record<FixedDatasetId,Record<FixedModelId,Reference>>};
 assert.equal(fixture.schemaVersion,1);assert.deepEqual(fixture.policy,manifest.policy);return fixture;
}
export async function loadModel(datasetId:FixedDatasetId,modelId:FixedModelId){
 const d=fixedModelDescriptor(datasetId,modelId);const text=read("public"+d.path);
 assert.equal(Buffer.byteLength(text),d.bytes);return verifyFixedModelJson(text,d.binding,d.sha256);
}
export function stats(actual:number[],reference:number[]){
 assert.equal(actual.length,reference.length);let maxAbsoluteError=0,maxRelativeError=0,failures=0;
 for(let i=0;i<actual.length;i++){
  assert.ok(Number.isFinite(actual[i])&&Number.isFinite(reference[i]));const difference=Math.abs(actual[i]-reference[i]);
  maxAbsoluteError=Math.max(maxAbsoluteError,difference);maxRelativeError=Math.max(maxRelativeError,difference/Math.max(Math.abs(reference[i]),policy.relativeErrorReportingFloor));
  if(difference>policy.absoluteTolerance+policy.relativeTolerance*Math.abs(reference[i]))failures++;
 }
 return {rows:actual.length,maxAbsoluteError,maxRelativeError,failures};
}
export async function verifyAllModels(){
 assert.equal(policy.absoluteTolerance,1e-10);assert.equal(policy.relativeTolerance,1e-10);
 const fixture=loadReference();const models:Record<string,unknown>={};let total=0;
 for(const datasetId of FIXED_DATASETS)for(const modelId of FIXED_MODELS){
  const model=await loadModel(datasetId,modelId),ref=fixture.datasets[datasetId][modelId];
  assert.equal(ref.inputs.length,model.trainingSplit.holdoutRows);
  const original=JSON.stringify(ref.inputs),prediction=ref.inputs.map(input=>predictFixedModel(model,input).prediction);
  assert.equal(JSON.stringify(ref.inputs),original);
  const comparisons={nativeVsSaved:stats(ref.nativePredictions,ref.savedPredictions),typescriptVsNative:stats(prediction,ref.nativePredictions),typescriptVsSaved:stats(prediction,ref.savedPredictions),extraCases:stats(ref.extraCases.map(c=>predictFixedModel(model,c.inputs).prediction),ref.extraCases.map(c=>c.nativePrediction))};
  for(const [name,s] of Object.entries(comparisons))assert.equal(s.failures,0,`${datasetId}/${modelId} ${name}`);
  total+=prediction.length;models[datasetId+'--'+modelId]={status:"PASS",features:model.featureOrder,comparisons};
 }
 assert.equal(total,8236);return {status:"PASS",policy,totalHoldoutPredictions:total,models};
}
export function inputsFor(model:FixedModel){return loadReference().datasets[model.datasetId][model.modelId].extraCases[0].inputs;}
