import {test} from "node:test";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {FIXED_DATASETS,FIXED_MODELS,parseFixedModel,predictFixedModel,verifyFixedModelJson,type FixedDatasetId,type FixedModelId} from "../lib/fixed-models/core.ts";
import {fixedModelDescriptor,loadFixedModel} from "../lib/fixed-models/catalog.ts";
import {inputsFor,loadModel,loadReference,read,verifyAllModels} from "./fixed-models-reference.ts";

test("six frozen models: all 8236 Holdout predictions match saved research and native sklearn at predeclared tolerance",async()=>{
 const result=await verifyAllModels();assert.equal(result.status,"PASS");assert.equal(Object.keys(result.models).length,6);
});
test("Training boundaries, missing values, categories and legal extrapolation match native extra cases",async()=>{
 const data=loadReference();let cases=0,extrapolation=0,missing=0;
 for(const dataset of FIXED_DATASETS)for(const id of FIXED_MODELS){const model=await loadModel(dataset,id);
  for(const c of data.datasets[dataset][id].extraCases){const before=JSON.stringify(c.inputs),result=predictFixedModel(model,c.inputs);cases++;
   if(c.name.startsWith("extrapolation:")){assert.ok(result.outOfRange.length>0);extrapolation++;}
   if(c.name.startsWith("missing:")){assert.ok(result.imputed.length>0);missing++;for(const row of result.imputed)assert.equal(row.value,model.features.find(f=>f.name===row.feature)!.imputer!.value);}
   assert.equal(JSON.stringify(c.inputs),before);
  }
 }
 assert.ok(cases>200&&extrapolation>20&&missing>20);
});
test("Energy Baseline remains raw and required; other models retain their own frozen median/scaler",async()=>{
 const raw=await loadModel("appliances-energy","baseline"),input=inputsFor(raw);
 assert.ok(raw.features.every(f=>f.imputer===null&&f.scaler===null));
 assert.equal(predictFixedModel(raw,input).prediction,raw.featureOrder.reduce((s,n,i)=>s+Number(input[n])*raw.coefficients[i],0)+raw.intercept);
 assert.throws(()=>predictFixedModel(raw,{...input,lights:null}),/必填/);
 const selected=await loadModel("appliances-energy","mutual_information");assert.deepEqual(selected.featureOrder,["T3","T4","T5","RH_6","T8","T9"]);assert.ok(selected.features.every(f=>f.imputer&&f.scaler));assert.ok(!selected.featureOrder.includes("date"));
});
test("Automobile maps both category labels and valid codes; missing/unknown categories are rejected",async()=>{
 const model=await loadModel("automobile-price","baseline"),base=inputsFor(model);
 assert.equal(model.unit,"原始價格單位（幣別未核實）");
 for(const [name,mapping] of [["fuel-type",{gas:0,diesel:1}],["aspiration",{std:0,turbo:1}]] as const){
  for(const [label,code] of Object.entries(mapping))assert.equal(predictFixedModel(model,{...base,[name]:label}).prediction,predictFixedModel(model,{...base,[name]:code}).prediction);
  for(const value of [null,undefined,"unknown",true,2,NaN,Infinity,"__proto__"])assert.throws(()=>predictFixedModel(model,{...base,[name]:value}));
 }
});
test("Student validates integers and ordinal codes while allowing legal scores beyond Training ranges",async()=>{
 const model=await loadModel("student-performance","baseline"),base=inputsFor(model);
 assert.throws(()=>predictFixedModel(model,{...base,age:17.5}),/整數/);
 assert.throws(()=>predictFixedModel(model,{...base,studytime:5}),/合法值/);
 assert.throws(()=>predictFixedModel(model,{...base,G1:21}),/合法值/);
 assert.throws(()=>predictFixedModel(model,{...base,absences:-1}),/合法值/);
 const result=predictFixedModel(model,{...base,G1:20,G2:20});assert.ok(result.outOfRange.includes("G1")&&result.outOfRange.includes("G2"));
 assert.ok(result.warnings.some(w=>w.includes("G1、G2")));assert.equal(result.datasetId,"student-performance");
});
test("real synthetic edge combinations retain out-of-domain predictions rather than clipping",async()=>{
 const data=loadReference();let below=0,above=0;
 for(const id of FIXED_MODELS){const model=await loadModel("student-performance",id);
  for(const c of data.datasets["student-performance"][id].extraCases.filter(c=>c.name.startsWith("linear-extreme:"))){const r=predictFixedModel(model,c.inputs);if(r.prediction<0)below++;if(r.prediction>20)above++;assert.ok(Math.abs(r.prediction-c.nativePrediction)<1e-10);}
 }
 assert.ok(below>0&&above>0);
});
test("non-finite, unsafe, wrong-type or extra inputs fail without mutation",async()=>{
 for(const dataset of FIXED_DATASETS)for(const id of FIXED_MODELS){const m=await loadModel(dataset,id),base=inputsFor(m),n=m.featureOrder[0];
  for(const value of [NaN,Infinity,-Infinity,true,[],{},"NaN","12",Number.MAX_SAFE_INTEGER+1,undefined])assert.throws(()=>predictFixedModel(m,{...base,[n]:value}));
  for(const input of [null,[],"bad",{...base,[m.target]:1}])assert.throws(()=>predictFixedModel(m,input));
  assert.ok(Object.isFrozen(m)&&Object.isFrozen(m.coefficients)&&Object.isFrozen(m.features[0]));
 }
});
test("JSON byte integrity, dataset/model/version binding and corrupt metadata fail closed",async()=>{
 const desc=fixedModelDescriptor("student-performance","baseline"),text=read("public"+desc.path);
 await assert.rejects(verifyFixedModelJson(text+" ",desc.binding,desc.sha256),/SHA-256/);
 await assert.rejects(verifyFixedModelJson(text,{...desc.binding,datasetId:"automobile-price"},desc.sha256),/dataset/);
 const invalid="{";await assert.rejects(verifyFixedModelJson(invalid,desc.binding,createHash("sha256").update(invalid).digest("hex")));
 const mutations:((m:ReturnType<typeof JSON.parse>)=>void)[]=[m=>m.schemaVersion=2,m=>m.modelVersion="2.0.0",m=>m.modelId="mutual_information",m=>m.datasetId="automobile-price",m=>m.coefficients.pop(),m=>m.coefficients[0]=NaN,m=>m.intercept=Infinity,m=>m.featureOrder[0]="G3",m=>m.features[0].scaler.scale=0,m=>m.features[0].trainingMin=Infinity,m=>m.sourceHashes[0].sha256="bad",m=>m.trainingSplit.holdoutUsedForFitting=true,m=>m.predictionClipped=true,m=>m.preprocessingOrder.reverse()];
 for(const mutate of mutations){const raw=JSON.parse(text);mutate(raw);assert.throws(()=>parseFixedModel(raw,desc.binding));}
 assert.throws(()=>fixedModelDescriptor("../../file" as FixedDatasetId,"baseline"));assert.throws(()=>fixedModelDescriptor("student-performance","bad" as FixedModelId));
});
test("static loader requests only allowlisted JSON; failures and aborts never substitute another model",async()=>{
 const previous=globalThis.fetch;const desc=fixedModelDescriptor("automobile-price","mutual_information");let calls=0;
 try{
  globalThis.fetch=async(url,init)=>{calls++;assert.equal(url,desc.path);assert.equal(init?.cache,"no-cache");return new Response(read("public"+desc.path));};
  assert.equal((await loadFixedModel("automobile-price","mutual_information")).modelId,"mutual_information");
  globalThis.fetch=async()=>new Response("missing",{status:404});await assert.rejects(loadFixedModel("automobile-price","mutual_information"),/不可用/);
  const abort=new AbortController();abort.abort();globalThis.fetch=async(_url,init)=>{assert.equal(init?.signal,abort.signal);throw new DOMException("Aborted","AbortError");};await assert.rejects(loadFixedModel("automobile-price","mutual_information",abort.signal),{name:"AbortError"});
  assert.equal(calls,1);
 }finally{globalThis.fetch=previous;}
});
test("runtime core and catalog do not depend on Python, Worker, gateway, Node or experiments",()=>{
 const core=read("lib/fixed-models/core.ts"),catalog=read("lib/fixed-models/catalog.ts");
 assert.ok(!/from ["']node:|new Worker|\.fit\(|custom-compute|\/api\//.test(core+catalog));assert.ok(!core.includes("fetch("));assert.ok(!catalog.includes("experiments/"));
});
