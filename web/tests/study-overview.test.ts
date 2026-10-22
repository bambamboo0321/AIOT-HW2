import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { DATASET_REGISTRY } from "../lib/dataset-registry.ts";
import { parseStudyOverview, type StudyDatasetId, type StudyOverview } from "../lib/study-overview-data.ts";
const directory = new URL("../data/research-overviews/v1/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json",directory),"utf8"));
const ids: StudyDatasetId[] = ["automobile-price","student-performance"];
function raw(id: StudyDatasetId): StudyOverview { return JSON.parse(readFileSync(new URL(`${id}.json`,directory),"utf8")); }

test("versioned overview snapshots match their byte hashes and validated metadata contracts", () => {
  assert.equal(manifest.schemaVersion,1);
  assert.deepEqual(Object.keys(manifest.snapshots),ids);
  for (const id of ids) {
    const bytes=readFileSync(new URL(`${id}.json`,directory));
    assert.equal(manifest.snapshots[id].file,`${id}.json`);
    assert.equal(createHash("sha256").update(bytes).digest("hex"),manifest.snapshots[id].sha256);
    assert.ok(bytes.length < 8000);
    const snapshot = parseStudyOverview(raw(id),id);
    assert.equal(snapshot.status,"ready");
    const data=raw(id), meta=DATASET_REGISTRY[id];
    assert.equal(data.dataset.rows,meta.rows); assert.equal(data.dataset.candidates.length,meta.candidateFeatures);
    assert.equal(data.protocol.trainRows+data.protocol.holdoutRows,data.dataset.rows);
    assert.equal(data.protocol.selectionLockedBeforeHoldout,true);
    assert.ok(!data.dataset.candidates.includes(meta.target));
    assert.equal(data.models[1].status,"ready");
    for (const forbidden of ['actual','prediction','source_rows','lower','upper','training_rows','model_path']) assert.ok(!bytes.toString().includes(`"${forbidden}"`));
  }
});
test("full precision Automobile metrics and actual PI coverage are retained from the verified experiment", () => {
  const d=raw("automobile-price"), [b,m]=d.models;
  assert.equal(b.status,"ready"); assert.equal(m.status,"ready");
  if (b.status!=="ready" || m.status!=="ready") return;
  assert.deepEqual(b.holdout,{rmse:5215.616480249021,mae:3657.6601741059035,r2:0.777659397271883});
  assert.deepEqual(m.holdout,{rmse:5394.760239130046,mae:3788.5915644677207,r2:0.7621233702826905});
  assert.ok(m.cv.rmse<b.cv.rmse && m.holdout.rmse>b.holdout.rmse);
  assert.deepEqual(m.features,["length","width","curb-weight","engine-size","horsepower","city-mpg","highway-mpg"]);
  for (const model of [b,m]) { assert.equal(model.interval.status,"ready"); if(model.interval.status==="ready") { assert.equal(model.interval.covered,34); assert.equal(model.interval.total,41); assert.equal(model.interval.coverage,34/41); } }
  assert.equal(d.dataset.removedTargets,4); assert.equal(d.dataset.rawRows,205);
  assert.match(d.dataset.unit,/幣別未核實/);
});
test("Student metrics retain original precision and G1/G2 timing rather than an early-term claim", () => {
  const d=raw("student-performance"), [b,m]=d.models;
  assert.equal(b.status,"ready"); assert.equal(m.status,"ready");
  if (b.status!=="ready" || m.status!=="ready") return;
  assert.deepEqual(b.holdout,{rmse:1.1664402826147413,mae:0.7348086550074645,r2:0.8604775881282163});
  assert.deepEqual(m.holdout,{rmse:1.1557841973700598,mae:0.7360348555838029,r2:0.8630151745464493});
  assert.ok(m.holdout.r2>b.holdout.r2 && m.holdout.mae>b.holdout.mae);
  for(const model of [b,m]) { assert.ok(model.features.includes("G1")&&model.features.includes("G2")); assert.equal(model.interval.status,"ready"); if(model.interval.status==="ready") assert.equal(model.interval.coverage,129/130); }
  assert.equal(d.dataset.removedTargets,0);assert.equal(d.dataset.unit,"分數（0–20）");
});
test("dataset/schema/protocol/metrics corruption fails closed without any Energy substitution", () => {
  const invalid: unknown[]=[undefined,null,{},raw("student-performance")];
  for(const mutate of [
    (d:StudyOverview)=>d.schemaVersion=2 as 1,
    (d:StudyOverview)=>d.protocol.selectionLockedBeforeHoldout=false as true,
    (d:StudyOverview)=>d.protocol.seed=0 as 42,
    (d:StudyOverview)=>d.dataset.target="Appliances",
    (d:StudyOverview)=>d.dataset.unit="Wh",
    (d:StudyOverview)=>d.dataset.candidates.push("price"),
    (d:StudyOverview)=>d.models.pop(),
    (d:StudyOverview)=>d.sources[0].sha256="invalid",
    (d:StudyOverview)=>{if(d.models[0].status==="ready")d.models[0].holdout.r2=NaN;},
    (d:StudyOverview)=>{if(d.models[0].status==="ready"&&d.models[0].interval.status==="ready")d.models[0].interval.covered=41;},
  ]){const data=raw("automobile-price");mutate(data);invalid.push(data);}
  for(const value of invalid){const result=parseStudyOverview(value,"automobile-price");assert.equal(result.status,"unavailable");if(result.status==="unavailable")assert.ok(result.issues.length>0);}
});
test("missing model and missing PI preserve explicit unavailable states instead of fabricated values", () => {
  const d=raw("automobile-price");
  d.models[1]={id:"mutual_information",label:"MI + LR",status:"unavailable",reason:"模型成果缺失"};
  assert.equal(parseStudyOverview(d,"automobile-price").status,"ready");
  if(d.models[0].status==="ready") d.models[0].interval={status:"unavailable",reason:"區間成果缺失"};
  assert.equal(parseStudyOverview(d,"automobile-price").status,"ready");
});
test("Overview and saved research pages are connected with fixed research inference connected", () => {
  for(const id of ids){assert.equal(DATASET_REGISTRY[id].features.overview,"connected");assert.equal(DATASET_REGISTRY[id].features.researchPages,"connected");assert.equal(DATASET_REGISTRY[id].features.livePrediction,"connected");}
});
test("Energy Overview JSON remains byte-identical to the Phase 3A baseline", () => {
  const bytes=readFileSync(new URL("../data/overview.json",import.meta.url));
  assert.equal(createHash("sha256").update(bytes).digest("hex"),"281346552bdeb59f7b341cb97548e753a4bad50ff9693d35191696e7a37ac365");
});
test("production loader is allowlisted and never opens raw CSV, model files or experiment paths", () => {
  const source=readFileSync(fileURLToPath(new URL("../lib/load-study-overview.ts",import.meta.url)),"utf8");
  assert.match(source,/data\/research-overviews\/v1\/automobile-price.json/);
  assert.match(source,/data\/research-overviews\/v1\/student-performance.json/);
  assert.match(source,/createHash\("sha256"\)/);
  assert.ok(!/readFile[^;]*(?:experiments|\.csv|\.joblib)/s.test(source));
});
