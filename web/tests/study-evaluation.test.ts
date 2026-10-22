import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { DATASET_REGISTRY } from "../lib/dataset-registry.ts";
import { parseStudyEvaluation,savedEvaluationMetrics,predictionPoints,intervalPoints,residualBins,quantile,type StudyEvaluationData } from "../lib/study-evaluation-data.ts";
import type { StudyDatasetId,StudyOverview } from "../lib/study-overview-data.ts";
const ids:StudyDatasetId[]=["automobile-price","student-performance"];
const folder=new URL("../data/research-evaluation/v1/",import.meta.url),overviewFolder=new URL("../data/research-overviews/v1/",import.meta.url);
const read=(path:string,base=folder)=>JSON.parse(readFileSync(new URL(path,base),"utf8"));
const manifest=read("manifest.json"),overviewManifest=read("manifest.json",overviewFolder);
type Source=Omit<StudyEvaluationData,"study"|"models">&{models:Omit<StudyEvaluationData["models"][number],"summary">[]};
function raw(id:StudyDatasetId):Source{return read(`${id}.json`);}
function parse(id:StudyDatasetId,value:unknown=raw(id),base:unknown=read(`${id}.json`,overviewFolder)){return parseStudyEvaluation(value,id,overviewManifest.snapshots[id].sha256,base);}
function ready(id:StudyDatasetId){const result=parse(id);assert.equal(result.status,"ready");if(result.status!=="ready")throw Error("evaluation snapshot unavailable");return result.data;}
const close=(a:number,b:number)=>assert.ok(Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a),Math.abs(b)),`${a} != ${b}`);

test("versioned evaluation bytes and linked Overview/source hashes are consistent",()=>{
 assert.equal(manifest.schemaVersion,1);assert.deepEqual(Object.keys(manifest.snapshots),ids);
 for(const id of ids){const bytes=readFileSync(new URL(`${id}.json`,folder));assert.equal(createHash("sha256").update(bytes).digest("hex"),manifest.snapshots[id].sha256);assert.ok(bytes.length<35000);const d=ready(id);assert.equal(d.overviewSha256,overviewManifest.snapshots[id].sha256);assert.equal(d.study.dataset.rows,DATASET_REGISTRY[id].rows);assert.equal(d.sources.filter(s=>s.path.endsWith('-holdout.csv')).length,2);assert.ok(!Object.hasOwn(d,"timestamp"));}
});
test("all 342 original predictions reproduce official MAE RMSE and R squared without synthesis",()=>{
 let count=0;
 for(const id of ids){const d=ready(id);for(const model of d.models){assert.equal(model.predictions.status,"ready");assert.equal(model.summary.status,"ready");if(model.predictions.status!=="ready"||model.summary.status!=="ready")throw Error("missing predictions");const p=model.predictions;assert.equal(p.actual.length,d.study.protocol.holdoutRows);count+=p.actual.length;const actual=savedEvaluationMetrics(p.actual,p.predicted);for(const k of ["mae","rmse","r2"] as const)close(actual[k],model.summary.holdout[k]);assert.equal(model.summary.k,model.id==="baseline"?d.study.dataset.candidates.length:7);}
 const [a,b]=d.models;if(a.predictions.status==="ready"&&b.predictions.status==="ready")assert.deepEqual(a.predictions.actual,b.predictions.actual);}
 assert.equal(count,342);
 const auto=ready("automobile-price"),student=ready("student-performance");
 assert.equal(auto.models[0].summary.status,"ready");if(auto.models[0].summary.status==="ready")assert.deepEqual(auto.models[0].summary.holdout,{rmse:5215.616480249021,mae:3657.6601741059035,r2:0.777659397271883});
 if(student.models[1].summary.status==="ready")assert.deepEqual(student.models[1].summary.holdout,{rmse:1.1557841973700598,mae:0.7360348555838029,r2:0.8630151745464493});
});
test("saved PI yields 34/41 and 129/130 coverage, exact misses and width statistics",()=>{
 for(const id of ids)for(const m of ready(id).models){if(m.predictions.status!=="ready"||m.interval.status!=="ready")throw Error("missing PI");const p=m.predictions,pi=m.interval,n=p.actual.length;let below=0,above=0;const widths=pi.lower.map((l,i)=>{const u=pi.upper[i];assert.ok(l<=p.predicted[i]&&p.predicted[i]<=u);close((l+u)/2,p.predicted[i]);if(p.actual[i]<l)below++;else if(p.actual[i]>u)above++;return u-l;});const covered=n-below-above,mean=widths.reduce((a,b)=>a+b,0)/n;assert.equal(covered,id==="automobile-price"?34:129);assert.equal(pi.stats.total,id==="automobile-price"?41:130);close(pi.stats.coverage,covered/n);close(pi.stats.averageWidth,mean);close(pi.stats.medianWidth,quantile(widths,.5));close(pi.stats.widthSD,Math.sqrt(widths.reduce((s,w)=>s+(w-mean)**2,0)/n));assert.equal(pi.stats.below,below);assert.equal(pi.stats.above,above);assert.equal(pi.holdoutResidualsUsed,false);assert.equal(pi.boundsClipped,false);}
});
test("negative bounds and bounds above G3=20 remain unchanged rather than clipped",()=>{
 const auto=ready("automobile-price"),student=ready("student-performance");
 for(const m of auto.models)if(m.interval.status==="ready")assert.ok(m.interval.lower.some(v=>v<0));
 for(const m of student.models)if(m.interval.status==="ready")assert.ok(m.interval.upper.some(v=>v>20));
});
test("scatter, residual, sorted PI and histogram all preserve every saved observation",()=>{
 for(const id of ids){const d=ready(id);const bins=residualBins(d.models);assert.equal(bins.length,20);for(const m of d.models){if(m.predictions.status!=="ready"||m.interval.status!=="ready")throw Error("missing data");const points=predictionPoints(m),pi=intervalPoints(m);assert.equal(points.length,d.study.protocol.holdoutRows);assert.equal(pi.length,points.length);assert.deepEqual(pi.map(p=>p.position).toSorted((a,b)=>a-b),points.map(p=>p.position));assert.equal(bins.reduce((s,b)=>s+b[m.id],0),points.length);for(let i=0;i<points.length;i++){assert.equal(points[i].actual,m.predictions.actual[i]);assert.equal(points[i].predicted,m.predictions.predicted[i]);assert.equal(points[i].residual,points[i].actual-points[i].predicted);}for(let i=0;i<pi.length;i++){const p=pi[i];assert.equal(p.rank,i+1);assert.equal(p.actual,m.predictions.actual[p.position-1]);assert.equal(p.predicted,m.predictions.predicted[p.position-1]);assert.deepEqual(p.bounds,[m.interval.lower[p.position-1],m.interval.upper[p.position-1]]);if(i>0)assert.ok(pi[i-1].predicted<=p.predicted);}}}
});
test("unavailable predictions and PI stay empty rather than being rebuilt from metrics",()=>{
 const d=raw("student-performance");d.models[0].predictions={status:"unavailable",reason:"逐筆預測缺失"};d.models[0].interval={status:"unavailable",reason:"無法核對 PI"};d.models[1].interval={status:"unavailable",reason:"PI 缺失"};const result=parse("student-performance",d);assert.equal(result.status,"ready");if(result.status!=="ready")return;assert.deepEqual(predictionPoints(result.data.models[0]),[]);assert.deepEqual(intervalPoints(result.data.models[0]),[]);assert.deepEqual(intervalPoints(result.data.models[1]),[]);assert.equal(predictionPoints(result.data.models[1]).length,130);const bins=residualBins(result.data.models);assert.equal(bins.reduce((s,b)=>s+b.mutual_information,0),130);
});
test("corrupt/missing arrays, dataset links, summary mismatch and invalid PI fail closed",()=>{
 const invalid:unknown[]=[undefined,null,{},raw("automobile-price")];
 const mutations:((d:Source)=>void)[]=[d=>d.overviewSha256="0".repeat(64),d=>d.sources.pop(),d=>d.models.pop(),d=>{const p=d.models[0].predictions;if(p.status==="ready")p.predicted[0]+=2;},d=>{const p=d.models[0].predictions;if(p.status==="ready")p.actual.pop();},d=>{const p=d.models[0].predictions;if(p.status==="ready")p.predicted[0]=NaN;},d=>{const p=d.models[0].interval;if(p.status==="ready")p.lower[0]+=1;},d=>{const p=d.models[0].interval;if(p.status==="ready")p.stats.covered=130;},d=>{const p=d.models[0].interval;if(p.status==="ready")p.holdoutResidualsUsed=true as false;},d=>{const p=d.models[0].interval;if(p.status==="ready")p.boundsClipped=true as false;},d=>{const p=d.models[0].interval;if(p.status==="ready")p.stats.averageWidth+=1;},d=>{d.models[0].predictions={status:"unavailable",reason:"缺失"};}];
 for(const mutate of mutations){const d=raw("student-performance");mutate(d);invalid.push(d);}
 for(const [i,value] of invalid.entries())assert.equal(parseStudyEvaluation(value,"student-performance",overviewManifest.snapshots["student-performance"].sha256,read("student-performance.json",overviewFolder)).status,"unavailable",`case ${i}`);
 const wrong=read("student-performance.json",overviewFolder) as StudyOverview;wrong.dataset.unit="Wh";assert.equal(parse("student-performance",raw("student-performance"),wrong).status,"unavailable");
});
test("evaluation and saved Prediction Explorer are connected with fixed research inference connected",()=>{
 for(const id of ids){assert.equal(DATASET_REGISTRY[id].features.modelEvaluation,"connected");assert.equal(DATASET_REGISTRY[id].features.researchPages,"connected");assert.equal(DATASET_REGISTRY[id].features.livePrediction,"connected");}
});
test("Energy evaluation and Custom inference remain byte-identical; Energy Prediction Explorer body is preserved",()=>{
 const hashes:Record<string,string>={"data/model-evaluation.json": "c40faef99342c0b8945e5b1a2af0870d410caebfb4eafe2aac08da2b87f15cdf", "components/evaluation-explorer.tsx": "5083791a0ddcd00853e6e2adf3fa4003a93da115d33b433587eceba5023a2c94", "lib/evaluation-data.ts": "871af711d3d37f622bbfaa550888af7ceb9a24ffe1523a7d0da3a3f099896660", "app/prediction-explorer/page.tsx": "b896cc37ae6f085af52744ac50c6eda32e793be3f6005ecf6a5476e3f34a9be0", "components/custom-dataset-wizard.tsx": "a81b54833ef8b5eef6110845a61090de97d46f47f020793e77108fe1910db5bd", "components/custom-prediction-form.tsx": "07b083858b8f5ddba19f9c1203fc9c145604c5655af09aa9aee33f1e78fa4efe", "public/browser-compute/worker.mjs": "198f3c938ded033dede0a7d5ee31024382806e33e3559d5c6f928b8afb7b29e9"};
 for(const [file,hash] of Object.entries(hashes)){
   let bytes=readFileSync(new URL('../'+file,import.meta.url));
   if(file==="app/prediction-explorer/page.tsx") bytes=Buffer.from(bytes.toString().replace('import { ResearchLivePredictor } from "@/components/research-live-predictor";\n','').replaceAll('<ResearchLivePredictor datasetId={dataset.id}/><h2 className="rlp-holdout-heading">Holdout Explorer · 已保存的測試資料</h2>','').replace('import { StudyPredictionExplorer } from "@/components/study-prediction-explorer";', 'import { ResearchNotConnected } from "@/components/research-not-connected";').replace('if (dataset.id !== "appliances-energy") return <StudyPredictionExplorer id={dataset.id}/>;', 'if (dataset.features.researchPages !== "connected") return <ResearchNotConnected dataset={dataset} pageTitle="Prediction Explorer" path="/prediction-explorer"/>;'));
   assert.equal(createHash("sha256").update(bytes).digest("hex"),hash,file);
 }

});
test("production loader has only allowlisted snapshot paths; client charts have no async fetch or fitting",()=>{
 const loader=readFileSync(new URL("../lib/load-study-evaluation.ts",import.meta.url),"utf8"),client=readFileSync(new URL("../components/study-evaluation-explorer.tsx",import.meta.url),"utf8");assert.match(loader,/data\/research-evaluation\/v1\/automobile-price.json/);assert.match(loader,/data\/research-evaluation\/v1\/student-performance.json/);assert.match(loader,/createHash\("sha256"\)/);assert.ok(!/readFile[^;]*experiments/s.test(loader));assert.ok(!/\bfetch\(|\.fit\(|new Worker\(/.test(client));assert.ok(!client.includes('Wh'));assert.match(client,/PI 圖 unavailable/);assert.match(client,/逐筆預測圖 unavailable/);
});
