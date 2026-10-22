import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { DATASET_REGISTRY } from "../lib/dataset-registry.ts";
import { parseStudyFeatureSelection, studyComparisonRows, STUDY_METHODS, type StudySelectionData } from "../lib/study-feature-selection-data.ts";
import type { StudyDatasetId, StudyOverview } from "../lib/study-overview-data.ts";
const ids:StudyDatasetId[]=["automobile-price","student-performance"];
const folder=new URL("../data/research-feature-selection/v1/",import.meta.url);
const overviewFolder=new URL("../data/research-overviews/v1/",import.meta.url);
const read=(name:string,base=folder)=>JSON.parse(readFileSync(new URL(name,base),"utf8"));
const manifest=read("manifest.json"), overviewManifest=read("manifest.json",overviewFolder);
function raw(id:StudyDatasetId):Omit<StudySelectionData,"study"> { return read(`${id}.json`); }
function parse(id:StudyDatasetId,value:unknown=raw(id),base:unknown=read(`${id}.json`,overviewFolder)) { return parseStudyFeatureSelection(value,id,overviewManifest.snapshots[id].sha256,base); }
function ready(id:StudyDatasetId) { const s=parse(id);assert.equal(s.status,"ready");if(s.status!=="ready")throw Error("snapshot unavailable");return s.data; }
function method(id:StudyDatasetId,index:number) { const m=ready(id).methods[index];assert.equal(m.status,"ready");if(m.status!=="ready")throw Error("method unavailable");return m; }

test("both versioned snapshots have exact hashes, six methods and 60 verified fold results",()=>{
  assert.equal(manifest.schemaVersion,1);assert.deepEqual(Object.keys(manifest.snapshots),ids);
  let folds=0;
  for(const id of ids){
    const bytes=readFileSync(new URL(`${id}.json`,folder));assert.equal(createHash("sha256").update(bytes).digest("hex"),manifest.snapshots[id].sha256);
    assert.ok(bytes.length<25000);
    const data=ready(id);assert.deepEqual(data.methods.map(m=>m.id),STUDY_METHODS);assert.equal(data.overviewSha256,overviewManifest.snapshots[id].sha256);
    assert.equal(data.study.dataset.rows,DATASET_REGISTRY[id].rows);assert.equal(data.study.protocol.selectedMethod,"mutual_information");
    for(const m of data.methods){assert.equal(m.status,"ready");if(m.status!=="ready")continue;folds+=m.folds.length;assert.equal(m.cv.status,"ready");assert.equal(m.features.length,m.finalK);assert.equal(m.folds.length,5);for(const f of m.folds){assert.equal(f.k,f.features.length);assert.ok(Number.isSafeInteger(f.k));}}
    for(const forbidden of ['actual','prediction','training_rows','holdout','curve','meanFinalK'])assert.ok(!bytes.toString().includes(`"${forbidden}"`));
  }
  assert.equal(folds,60);
});
test("Automobile has four fixed k=7 methods and real automatic final k 9/3",()=>{
  const data=ready("automobile-price");assert.equal(data.study.dataset.rows,201);assert.equal(data.study.dataset.candidates.length,14);
  for(const m of data.methods.slice(0,4)){assert.equal(m.mode,"fixed");assert.equal(m.requestedK,7);if(m.status==="ready"){assert.equal(m.finalK,7);assert.ok(m.folds.every(f=>f.k===7));}}
  const lasso=method("automobile-price",4),rfecv=method("automobile-price",5);
  assert.equal(lasso.finalK,9);assert.equal(rfecv.finalK,3);assert.deepEqual(lasso.folds.map(f=>f.k),[10,8,8,9,10]);assert.deepEqual(rfecv.folds.map(f=>f.k),[3,4,3,5,7]);
  assert.deepEqual(rfecv.features,["width","engine-size","horsepower"]);
  const mi=method("automobile-price",1);if(mi.cv.status==="ready"){assert.equal(mi.cv.mean.rmse,3013.100461350188);assert.equal(mi.cv.std.rmse,350.5561907182682);}
});
test("Student retains exact MI features and LassoCV/RFECV final 5/2 with distinct fold k",()=>{
  const data=ready("student-performance");assert.equal(data.study.dataset.rows,649);assert.equal(data.study.dataset.candidates.length,15);
  assert.deepEqual(method("student-performance",1).features,["Fedu","studytime","failures","Dalc","Walc","G1","G2"]);
  const lasso=method("student-performance",4),rfecv=method("student-performance",5);assert.equal(lasso.finalK,5);assert.equal(rfecv.finalK,2);
  assert.deepEqual(lasso.folds.map(f=>f.k),[4,15,8,14,3]);assert.deepEqual(rfecv.folds.map(f=>f.k),[2,5,2,2,2]);assert.deepEqual(rfecv.features,["G1","G2"]);
  const mi=method("student-performance",1);if(mi.cv.status==="ready"){assert.equal(mi.cv.mean.rmse,1.287132091493009);assert.equal(mi.cv.std.rmse,0.26865956174205147);}
});
test("CV means and population deviations independently match each of five actual fold scores",()=>{
 for(const id of ids)for(const m of ready(id).methods){if(m.status!=="ready"||m.cv.status!=="ready")throw Error("missing CV");for(const k of ["rmse","mae","r2"] as const){const values=m.folds.map(f=>{assert.ok(f.scores);return f.scores[k];});const mean=values.reduce((a,b)=>a+b,0)/5,std=Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/5);assert.ok(Math.abs(mean-m.cv.mean[k])<1e-10);assert.ok(Math.abs(std-m.cv.std[k])<1e-10);}}
});
test("comparison chart rows are method summaries, never fabricated k curves or Holdout scores",()=>{
 for(const id of ids){const d=ready(id);for(const k of ["rmse","mae","r2"] as const){const rows=studyComparisonRows(d,k);assert.equal(rows.length,6);for(let i=0;i<6;i++){const m=d.methods[i];assert.equal(rows[i].id,m.id);if(m.status==="ready"&&m.cv.status==="ready")assert.equal(rows[i].value,m.cv.mean[k]);assert.ok(!Object.hasOwn(rows[i],"k"));}}}
 const component=readFileSync(new URL("../components/study-feature-explorer.tsx",import.meta.url),"utf8");assert.match(component,/BarChart/);assert.ok(!component.includes("LineChart"));assert.ok(!component.includes('type="range"'));
});
test("unavailable method or CV keeps real features and null chart gaps, never zero or Energy data",()=>{
 const d=raw("student-performance");d.methods[0]={id:"f_test",mode:"fixed",requestedK:7,status:"unavailable",reason:"方法成果缺失"};
 const mi=d.methods[1];if(mi.status!=="ready")throw Error("MI missing");mi.cv={status:"unavailable",reason:"摘要缺失"};mi.folds[0].scores=null;
 const result=parse("student-performance",d);assert.equal(result.status,"ready");if(result.status!=="ready")return;const rows=studyComparisonRows(result.data,"rmse");assert.equal(rows[0].value,null);assert.equal(rows[1].value,null);assert.notEqual(rows[2].value,null);assert.deepEqual((result.data.methods[1] as typeof mi).features,mi.features);
});
test("wrong dataset, linked hash, feature count and inconsistent fold/CV values fail closed",()=>{
 const cases:unknown[]=[undefined,null,{},raw("automobile-price")];
 const mutations:((d:ReturnType<typeof raw>)=>void)[]=[d=>d.schemaVersion=2 as 1,d=>d.overviewSha256="0".repeat(64),d=>d.methods.pop(),d=>d.sources[0].sha256="bad",d=>{d.methods[0].requestedK=8 as 7;},d=>{const m=d.methods[4];if(m.status==="ready")m.finalK=5.2;},d=>{const m=d.methods[0];if(m.status==="ready")m.features[0]="G3";},d=>{const m=d.methods[0];if(m.status==="ready"&&m.cv.status==="ready")m.cv.mean.rmse+=1;},d=>{const m=d.methods[0];if(m.status==="ready")m.folds.pop();},d=>{const m=d.methods[0];if(m.status==="ready")m.folds[0].scores=null;},d=>{const m=d.methods[0];if(m.status==="ready")m.folds[0].fold=2;}];
 for(const mutate of mutations){const d=raw("student-performance");mutate(d);cases.push(d);}
 for(const [i,value] of cases.entries())assert.equal(parseStudyFeatureSelection(value,"student-performance",overviewManifest.snapshots["student-performance"].sha256,read("student-performance.json",overviewFolder)).status,"unavailable",`invalid case ${i}`);
 const corrupt=read("student-performance.json",overviewFolder) as StudyOverview;corrupt.dataset.unit="Wh";assert.equal(parse("student-performance",raw("student-performance"),corrupt).status,"unavailable");
});
test("Overview and Feature Selection remain connected with fixed research inference connected",()=>{
 for(const id of ids){const f=DATASET_REGISTRY[id].features;assert.equal(f.featureSelection,"connected");assert.equal(f.overview,"connected");assert.equal(f.researchPages,"connected");assert.equal(f.livePrediction,"connected");}
});
test("Energy feature snapshots, original curve explorer and Custom compute components remain byte-identical",()=>{
 const expected:Record<string,string>={"data/feature-selection.json": "bbfed52979af548ae4f2a06e1f16ee4688db233a09ef7d7ea431307fcbd9d45b", "data/overview.json": "281346552bdeb59f7b341cb97548e753a4bad50ff9693d35191696e7a37ac365", "components/feature-selection-explorer.tsx": "3a18f920601c4c5ca8eeff9f4b7cf61957dfbebbcc1d8e4d5b179e897e26da0e", "components/custom-dataset-wizard.tsx": "a81b54833ef8b5eef6110845a61090de97d46f47f020793e77108fe1910db5bd", "components/custom-compute-provider.tsx": "6e537b0a73b21811afcaa0e226b431f83db08f2fa6ce839aa9220bbf46e952dc", "public/browser-compute/worker.mjs": "198f3c938ded033dede0a7d5ee31024382806e33e3559d5c6f928b8afb7b29e9"};
 for(const [path,hash] of Object.entries(expected)){assert.equal(createHash("sha256").update(readFileSync(new URL('../'+path,import.meta.url))).digest("hex"),hash,path);}
});
