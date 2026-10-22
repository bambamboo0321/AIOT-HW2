import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { parseStudyEvaluation, intervalPoints, predictionPoints, type StudyEvaluationData } from "../lib/study-evaluation-data.ts";
import { studyPredictionRows, visibleStudyPredictions, PREDICTION_SORTS } from "../lib/study-prediction-data.ts";
import type { StudyDatasetId } from "../lib/study-overview-data.ts";
const ids: StudyDatasetId[] = ["automobile-price", "student-performance"];
const read = (p:string) => readFileSync(new URL("../"+p,import.meta.url),"utf8");
const json = (p:string) => JSON.parse(read(p));
function data(id:StudyDatasetId):StudyEvaluationData {
 const overview = json(`data/research-overviews/v1/${id}.json`);
 const manifest = json("data/research-overviews/v1/manifest.json");
 const result = parseStudyEvaluation(json(`data/research-evaluation/v1/${id}.json`),id,manifest.snapshots[id].sha256,overview);
 if(result.status!=="ready")throw Error(result.issues.join(";"));return result.data;
}
test("all 342 Explorer observations and PI equal Model Evaluation, with original ordinals",()=>{
 let count=0;
 for(const id of ids)for(const model of data(id).models){
  const rows=studyPredictionRows(model),evaluated=intervalPoints(model);
  if(model.predictions.status!=="ready"||model.interval.status!=="ready")throw Error("missing data");
  assert.equal(rows.length,id==="automobile-price"?41:130);
  for(const [i,row] of rows.entries()){
   const same=evaluated.find(p=>p.position===row.position)!;
   assert.equal(row.position,i+1);assert.equal(row.actual,model.predictions.actual[i]);assert.equal(row.predicted,model.predictions.predicted[i]);
   assert.equal(row.residual,row.actual-row.predicted);assert.equal(row.absoluteError,Math.abs(row.residual));
   assert.equal(row.lower,model.interval.lower[i]);assert.equal(row.upper,model.interval.upper[i]);
   assert.deepEqual([row.actual,row.predicted,row.residual,row.lower,row.upper],[same.actual,same.predicted,same.residual,...same.bounds]);count++;
  }
 }
 assert.equal(count,342);
});
test("error sorts are stable, exact ordinal search works across both models, and source arrays are not mutated",()=>{
 for(const id of ids)for(const model of data(id).models){
  const before=JSON.stringify(model),rows=studyPredictionRows(model),original=JSON.stringify(rows);
  for(const sort of PREDICTION_SORTS){const sorted=visibleStudyPredictions(rows,"",sort);assert.equal(sorted.length,rows.length);
   for(let i=1;i<sorted.length;i++){if(sort==="sample")assert.ok(sorted[i-1].position<sorted[i].position);else assert.ok(sort==="error-desc"?sorted[i-1].absoluteError>=sorted[i].absoluteError:sorted[i-1].absoluteError<=sorted[i].absoluteError);}
   for(const p of rows)assert.deepEqual(visibleStudyPredictions(rows,` ${p.position} `,sort),[p]);
  }
  for(const search of ["0","-1","1.2","1e1","01","NaN","<script>",String(rows.length+1)])assert.deepEqual(visibleStudyPredictions(rows,search,"sample"),[]);
  assert.equal(JSON.stringify(model),before);assert.equal(JSON.stringify(rows),original);
  const tied=[{...rows[0],position:3,absoluteError:1},{...rows[0],position:1,absoluteError:1},{...rows[0],position:2,absoluteError:1}];
  for(const sort of PREDICTION_SORTS)assert.deepEqual(visibleStudyPredictions(tied,"",sort).map(p=>p.position),[1,2,3]);
 }
});
test("missing PI stays null, missing predictions stay empty; no metric-based point synthesis",()=>{
 const model=data("student-performance").models[0];model.interval={status:"unavailable",reason:"missing PI"};
 const rows=studyPredictionRows(model);assert.equal(rows.length,130);assert.ok(rows.every(p=>p.lower===null&&p.upper===null));assert.deepEqual(rows.map(p=>p.predicted),predictionPoints(model).map(p=>p.predicted));
 model.predictions={status:"unavailable",reason:"missing predictions"};assert.deepEqual(studyPredictionRows(model),[]);assert.deepEqual(visibleStudyPredictions([],"","sample"),[]);
});
test("Student scores and PI are never clipped, including out-of-range predictions",()=>{
 const model=data("student-performance").models[0];assert.ok(studyPredictionRows(model).some(p=>p.upper!==null&&p.upper>20));
 // Explicit presentation fixture: adapters already validate real snapshots; this tests display transformation only.
 model.predictions={status:"ready",actual:[0,20],predicted:[-2,22]};model.interval={status:"unavailable",reason:"fixture"};
 assert.deepEqual(studyPredictionRows(model).map(p=>[p.predicted,p.residual,p.absoluteError]),[[-2,2,2],[22,-2,2]]);
});
test("Explorer reuses only Phase 3D loader, has no data fetching or model execution, and keys state by dataset",()=>{
 const server=read("components/study-prediction-explorer.tsx"),client=read("components/study-prediction-browser.tsx");
 assert.match(server,/loadStudyEvaluation\(id\)/);assert.match(server,/<StudyPredictionBrowser key=\{id\}/);
 assert.ok(!/\bfetch\(|new Worker\(|\.fit\(|\.predict\(/.test(server+client));
 assert.ok(!/readFile|loadPredictions|time-series|USD|TWD|Wh/.test(server+client));
 assert.match(server,/逐筆預測 unavailable/);assert.match(client,/95% PI unavailable/);
 assert.match(client,/function selectModel[^}]*setQuery\(""\)[^}]*setSort\("sample"\)[^}]*setPosition\(1\)/);
 assert.match(client,/實際 − 預測/);assert.match(client,/0–20/);
});
test("Phase 3D snapshots/adapter, Energy API/client and pending SFS/Custom changes remain unchanged",()=>{
 const expected:Record<string,string> = {
  "src/custom_dataset_comparison.py": "8c06671cddcf31a8b3f7f90fc9356c260d1966e2f165b1c946e837f9c2e6aa53",
  "src/custom_dataset_selectors.py": "fd2efcd0bd47f384a718d2bb5b961001e50e53232bb6046f2508b8941cbab088",
  "src/custom_sfs_comparison.py": "9fd22b71df3eca1a102d4aa52692308dbb1e839980cbd42bfb81949f2f63863e",
  "web/app/api/predictions/route.ts": "9a20ef0e92144a0facb37059c2d63e3d43fe30c4415ed4e922685b339cddd548",
  "web/components/custom-compute-provider.tsx": "6e537b0a73b21811afcaa0e226b431f83db08f2fa6ce839aa9220bbf46e952dc",
  "web/components/custom-dataset-wizard.tsx": "a81b54833ef8b5eef6110845a61090de97d46f47f020793e77108fe1910db5bd",
  "web/components/custom-prediction-form.tsx": "07b083858b8f5ddba19f9c1203fc9c145604c5655af09aa9aee33f1e78fa4efe",
  "web/components/prediction-explorer.tsx": "77bfb8a562e9df5ab1ae7b0a77de608d4561f7077ea754da63483c5b6c731c78",
  "web/components/study-evaluation-explorer.tsx": "3f5d50c719a41a2f2f0d1ce212f0e943ce97002b81f2f854fb64ecd30c762f56",
  "web/data/prediction-explorer.json": "cf8a8ca5eff949e44f9c2c5c86ddef39ff17b376084d59bb38d739e44cccffb9",
  "web/data/research-evaluation/README.md": "2685eda20ec5777b3cc6574433df3e64fa183a3c4647cd9befcb73ad87aaca64",
  "web/data/research-evaluation/v1/automobile-price.json": "48b849b872b9a09097150923a339396fa29e5ab8817b176b8d6038744d120297",
  "web/data/research-evaluation/v1/manifest.json": "27e4ed22f2846bae684ad42b954ee347f356d0dc00111b3dca9fa084771b0a22",
  "web/data/research-evaluation/v1/student-performance.json": "5fc5be58b02804f909ba3c12bd3e2d4211d844d335c86c1278d7d0a8a1a6588c",
  "web/lib/load-predictions.ts": "4bc3d20c0b216454d1060f65e31669edf1dcf6ba10c290e812c4f423c9afb913",
  "web/lib/load-study-evaluation.ts": "e99d0171ebde814221a6f92d24eb6c2b365f3d2bf4143288cf88c5d78a987c09",
  "web/lib/prediction-window.ts": "65c73e29ef4b5fc9058080dbbcf0d420a340b15f1a71a3af2f10896bdba72119",
  "web/lib/study-evaluation-data.ts": "464ca2b7479931a23c91fb499cfe94c699b2d3ac488ca8e61216565808ba4d59",
  "web/public/browser-compute/worker.mjs": "198f3c938ded033dede0a7d5ee31024382806e33e3559d5c6f928b8afb7b29e9"
};
 for(const [p,hash] of Object.entries(expected))assert.equal(createHash("sha256").update(readFileSync(new URL("../../"+p,import.meta.url))).digest("hex"),hash,p);
});
