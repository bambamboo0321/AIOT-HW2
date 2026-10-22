import { test } from "node:test";
import assert from "node:assert/strict";
import { DATASET_IDS, DATASET_REGISTRY, DEFAULT_DATASET_ID, getDataset, resolveDatasetId } from "../lib/dataset-registry.ts";
import { RESEARCH_PATHS, datasetFromPageParams, datasetForPath, datasetIdFromQuery, datasetSwitchHref, researchHref, workspaceHref } from "../lib/dataset-routing.ts";

test("registry contains the three verified dataset metadata contracts without model results", () => {
  assert.deepEqual(DATASET_IDS, ["appliances-energy", "automobile-price", "student-performance"]);
  for (const id of DATASET_IDS) {
    const d = DATASET_REGISTRY[id];
    assert.equal(d.id, id);
    assert.equal(d.protocol.trainingRows + d.protocol.holdoutRows, d.rows);
    assert.ok(d.candidateFeatures > 0);
    assert.equal(d.protocol.outerFolds, 5);
    assert.equal(d.protocol.innerFolds, 3);
    assert.equal(d.features.livePrediction, "connected");
    for (const forbidden of ["rmse", "mae", "r2", "models", "predictions"]) assert.equal(Object.hasOwn(d, forbidden), false);
  }
  assert.deepEqual(DATASET_IDS.map(id => [DATASET_REGISTRY[id].target, DATASET_REGISTRY[id].rows, DATASET_REGISTRY[id].candidateFeatures]), [["Appliances",19735,25],["price",201,14],["G3",649,15]]);
});

test("all saved research pages are connected with independent fixed research inference", () => {
  for (const id of DATASET_IDS) assert.equal(getDataset(id).features.overview, "connected");
  for (const id of DATASET_IDS) assert.equal(getDataset(id).features.featureSelection, "connected");
  for (const id of DATASET_IDS) assert.equal(getDataset(id).features.modelEvaluation, "connected");
  assert.equal(getDataset("appliances-energy").features.researchPages, "connected");
  for (const id of ["automobile-price", "student-performance"]) assert.equal(getDataset(id).features.researchPages, "connected");
  assert.equal(getDataset("automobile-price").protocol.kPolicy, "fixed-7-and-auto");
  assert.match(getDataset("student-performance").note, /G1、G2/);
  assert.match(getDataset("student-performance").note, /非開學前/);
});

test("missing, unknown, prototype and file-path IDs fall back to energy", () => {
  for (const value of [undefined, null, "", "unknown", "constructor", "__proto__", "../../experiments", "Student-Performance", ["student-performance"], 42]) {
    assert.equal(resolveDatasetId(value), DEFAULT_DATASET_ID);
    assert.equal(getDataset(value).id, DEFAULT_DATASET_ID);
  }
});

test("Server Page params and client URL parsing agree, including duplicate parameters", () => {
  const cases: [string, {dataset?:string|string[]}][] = [["",{}],["dataset=student-performance",{dataset:"student-performance"}], ["dataset=automobile-price",{dataset:"automobile-price"}], ["dataset=bad",{dataset:"bad"}], ["dataset=student-performance&dataset=automobile-price",{dataset:["student-performance","automobile-price"]}]];
  for (const [query, server] of cases) assert.equal(datasetIdFromQuery(new URLSearchParams(query)), datasetFromPageParams(server).id);
});

test("research navigation retains the selected ID across every research page", () => {
  for (const id of DATASET_IDS) for (const path of RESEARCH_PATHS) {
    const url = new URL(workspaceHref(path, id), "https://example.test");
    assert.equal(url.pathname, path);
    assert.equal(url.searchParams.get("dataset"), id);
    assert.equal(datasetForPath(path, url.searchParams)?.id, id);
  }
  assert.equal(researchHref("/model-evaluation", "student-performance", "research-conclusion"), "/model-evaluation?dataset=student-performance#research-conclusion");
});

test("selector changes normalize duplicate IDs and preserve unrelated query and hash", () => {
  const href = datasetSwitchHref("/feature-selection", "dataset=bad&dataset=other&view=detail", "student-performance", "#method-comparison-title");
  const url = new URL(href!, "https://example.test");
  assert.deepEqual(url.searchParams.getAll("dataset"), ["student-performance"]);
  assert.equal(url.searchParams.get("view"), "detail");
  assert.equal(url.hash, "#method-comparison-title");
  assert.equal(datasetSwitchHref("/", "", "not-a-dataset"), "/?dataset=appliances-energy");
});

test("refresh, shared links and history entries resolve from the URL without storage", () => {
  const history = DATASET_IDS.map(id => researchHref("/model-evaluation", id));
  for (const href of [...history, ...history.toReversed()]) {
    const url = new URL(href, "https://example.test");
    assert.equal(datasetIdFromQuery(url.searchParams), url.searchParams.get("dataset"));
  }
});

test("Custom Dataset ignores all research query IDs and cannot be switched by selector", () => {
  for (const id of DATASET_IDS) {
    const query = new URLSearchParams({dataset:id});
    assert.equal(datasetForPath("/custom-dataset", query), null);
    assert.equal(workspaceHref("/custom-dataset", id), "/custom-dataset");
    assert.equal(datasetSwitchHref("/custom-dataset", query.toString(), "student-performance"), null);
  }
  assert.equal(datasetForPath("/not-a-research-route", new URLSearchParams()), null);
});
