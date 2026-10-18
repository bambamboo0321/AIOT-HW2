import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile, symlink } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { adaptFeatureSelection, attachOuterResults, adaptFeatureExploration, buildFeatureSnapshot, parseFeatureSnapshot, FEATURE_SOURCES } from "../lib/feature-selection-data.ts";
import { writeSnapshot } from "../lib/write-snapshot.ts";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(web, "..");
const json = async (path: string) => JSON.parse(await readFile(resolve(repo, path), "utf8"));
const raw = await json(FEATURE_SOURCES.selection);
const config = await json(FEATURE_SOURCES.config);
const plots = await json(FEATURE_SOURCES.plots);
const best = await json(FEATURE_SOURCES.best);
const phase3d = await json(FEATURE_SOURCES.exploration);
const formalRecords = await json(FEATURE_SOURCES.records);
async function fixture(run: (root: string) => Promise<void>) {
  const root = await mkdtemp(resolve(web, ".test-tmp-m2-"));
  try {
    for (const path of Object.values(FEATURE_SOURCES)) {
      await mkdir(dirname(resolve(root, path)), { recursive: true });
      await writeFile(resolve(root, path), await readFile(resolve(repo, path)));
    }
    await run(root);
  } finally { await rm(root, { recursive: true, force: true }); }
}

test("all six methods preserve exact formal CV scores and final feature mappings", () => {
  const data = adaptFeatureSelection(raw, config);
  assert.equal(data.methods.length, 6);
  assert.equal(data.winner, best.winner.method);
  for (const m of data.methods) {
    const cv = best.by_method.find((r: { method: string }) => r.method === m.id);
    const nested = best.nested_outer_assessment.find((r: { method: string }) => r.method === m.id);
    const refit = best.full_training_refits.find((r: { method: string }) => r.method === m.id);
    for (const metric of ["rmse", "mae", "r2"] as const) {
      assert.equal(m.finalCV[metric], cv[metric]); assert.equal(m.nestedCV[metric], nested[metric]);
    }
    assert.equal(m.finalK, refit.selected_k);
    assert.deepEqual(m.transformedFeatures, refit.selection.selected_feature_names);
    assert.deepEqual(m.outerK, nested.selected_k);
    assert.equal(m.features.length, m.finalK);
  }
});

test("fixed curves contain only actual tested k, with exact scores; SFS stays sparse", () => {
  const data = adaptFeatureSelection(raw, config);
  for (const method of data.methods.filter(m => !m.automatic)) {
    const points = plots.figures["rmse_vs_feature_count.png"].points.filter((p: { method: string }) => p.method === method.id);
    assert.deepEqual(method.curve.map(p => p.k), config.grids[method.id]);
    for (const [i, p] of method.curve.entries()) {
      for (const metric of ["rmse", "mae", "r2"] as const) assert.equal(p.scores![metric], points[i][metric]);
    }
  }
  const sfs = data.methods.find(m => m.id === "forward_sfs")!;
  assert.equal(sfs.curve.length, 9); assert.ok(!sfs.curve.some(p => p.k === 6));
});

test("automatic k has no curve and preserves fold k separately from full-Training k", () => {
  const data = adaptFeatureSelection(raw, config);
  for (const m of data.methods.filter(m => m.automatic)) {
    assert.deepEqual(m.curve, []);
    assert.equal(m.outerK.length, config.outer_splits);
    assert.equal(m.finalK, best.full_training_refits.find((r: { method: string }) => r.method === m.id).selected_k);
  }
  const lasso = data.methods.find(m => m.id === "lasso_cv")!;
  assert.notEqual(lasso.finalK, lasso.meanFinalK);
});

test("input scores propagate without hardcoded or rounded replacements", () => {
  const copy = structuredClone(raw); copy.data.methods[0].training_cv.rmse = 123.456789;
  assert.equal(adaptFeatureSelection(copy, config).methods[0].finalCV.rmse, 123.456789);
});

test("invalid schemas, holdout selection, corrupt k, units and wrong CV levels fail closed", () => {
  for (const change of [
    (d: typeof raw) => { d.schema_version = 2; },
    (d: typeof raw) => { d.units.rmse = "kWh"; },
    (d: typeof raw) => { d.data.holdout_used_for_selection = true; },
    (d: typeof raw) => { d.data.methods[0].nested_cv.level = "holdout"; },
    (d: typeof raw) => { d.data.methods[0].nested_cv.valid_folds = 4; },
    (d: typeof raw) => { d.data.methods[0].full_training_refit.selected_k = 25; },
    (d: typeof raw) => { d.data.methods[0].training_cv.rmse = NaN; },
    (d: typeof raw) => { d.data.methods[4].training_cv.requested_k = 7; },
    (d: typeof raw) => { d.data.methods.push(d.data.methods[0]); },
  ]) { const copy = structuredClone(raw); change(copy); assert.throws(() => adaptFeatureSelection(copy, config)); }
});

test("incomplete fixed-k folds retain a missing point rather than a partial or zero score", () => {
  const copy = structuredClone(raw);
  for (const metric of ["rmse", "mae", "r2"]) {
    const p = copy.data.curves[`${metric}_vs_feature_count.png`].points[0];
    p.valid_folds = 4; p.rmse = null; p.mae = null; p.r2 = null;
  }
  assert.equal(adaptFeatureSelection(copy, config).methods[0].curve[0].scores, null);
});

test("Phase 3D keeps fixed-combination scores apart from fold-local selector scores", () => {
  const data = adaptFeatureSelection(raw, config);
  const e = adaptFeatureExploration(phase3d, data);
  assert.equal(e.available, true); if (!e.available) return;
  const selector = e.selectorRows.find(r => r.id === "mutual_information")!;
  const fixed = e.fixedRows.find(r => r.label === "Mutual Information")!;
  assert.equal(selector.scores.rmse, phase3d.selector_cv[0].rmse);
  assert.equal(fixed.scores.rmse, phase3d.fixed_combinations.mutual_information.rmse_wh);
  assert.notEqual(fixed.scores.rmse, selector.scores.rmse);
  assert.ok(!e.fixedRows.some(r => r.label === "Forward SFS"));
  assert.equal(e.fixedRows.at(-1)!.scores.rmse, phase3d.random_search.best.rmse_wh);
  const copy = structuredClone(phase3d); copy.conditions.holdout_used = true;
  assert.throws(() => adaptFeatureExploration(copy, data));
});

test("snapshot is deterministic, traces raw formal sources and contains no model or row data", async () => {
  const a = await buildFeatureSnapshot(repo); const b = await buildFeatureSnapshot(repo);
  assert.equal(a.status, "ready"); assert.deepEqual(a, b); assert.deepEqual(parseFeatureSnapshot(a), a);
  for (const s of a.sources) assert.equal(s.sha256, createHash("sha256").update(await readFile(resolve(repo, s.path))).digest("hex"));
  assert.equal(a.sources.length, Object.keys(FEATURE_SOURCES).length);
  for (const forbidden of ["coefficients", "checkpoint_id", "train_range", "/Users/"]) assert.ok(!JSON.stringify(a).includes(forbidden));
});

test("missing mandatory sources or stale dashboard references never render fabricated data", async () => {
  await fixture(async root => {
    const file = resolve(root, FEATURE_SOURCES.best);
    const contents = await readFile(file);
    await rm(file); const absent = await buildFeatureSnapshot(root);
    assert.equal(absent.status, "unavailable"); assert.ok(!("data" in absent));
    await writeFile(file, contents.toString() + "\n");
    assert.equal((await buildFeatureSnapshot(root)).status, "unavailable");
  });
});

test("missing optional Phase 3D leaves formal results available with an explicit message", async () => {
  await fixture(async root => {
    await rm(resolve(root, FEATURE_SOURCES.exploration));
    const s = await buildFeatureSnapshot(root); assert.equal(s.status, "ready");
    if (s.status === "ready") assert.equal(s.data.exploration.available, false);
  });
});

test("out-of-repository symlinks and corrupted snapshot data are rejected", async () => {
  await fixture(async root => {
    await rm(resolve(root, FEATURE_SOURCES.selection));
    await symlink(resolve(repo, FEATURE_SOURCES.selection), resolve(root, FEATURE_SOURCES.selection));
    assert.equal((await buildFeatureSnapshot(root)).status, "unavailable");
  });
  const a = await buildFeatureSnapshot(repo); assert.equal(a.status, "ready"); if (a.status !== "ready") return;
  assert.throws(() => parseFeatureSnapshot({ ...a, schemaVersion: 2 }));
  const copy = structuredClone(a); copy.data.methods[4].curve = copy.data.methods[0].curve;
  assert.throws(() => parseFeatureSnapshot(copy));
});

test("sync updates the feature snapshot without modifying the Overview snapshot", async () => {
  await fixture(async root => {
    const s = await buildFeatureSnapshot(root);
    await mkdir(resolve(root, "data")); await writeFile(resolve(root, "data/overview.json"), "preserve M1");
    await writeSnapshot(root, s, "feature-selection");
    const first = await readFile(resolve(root, "data/feature-selection.json"));
    await writeSnapshot(root, s, "feature-selection");
    assert.deepEqual(first, await readFile(resolve(root, "data/feature-selection.json")));
    assert.equal(await readFile(resolve(root, "data/overview.json"), "utf8"), "preserve M1");
  });
});


test("every fold retains actual measured k and all three scores in chronological order", () => {
  const data = adaptFeatureSelection(raw, config); attachOuterResults(data, formalRecords);
  for (const m of data.methods) {
    const records = formalRecords.records.filter((r: {method: string; level: string}) => r.method === m.id && r.level === "outer_evaluation");
    assert.equal(m.outerResults.length, config.outer_splits);
    for (const [i, result] of m.outerResults.entries()) {
      assert.equal(result.fold, records[i].outer_fold); assert.equal(result.k, records[i].selected_k);
      for (const metric of ["rmse", "mae", "r2"] as const) assert.equal(result.scores[metric], records[i][metric]);
    }
    for (const metric of ["rmse", "mae", "r2"] as const) assert.ok(Math.abs(m.outerResults.reduce((sum, r) => sum + r.scores[metric], 0) / m.outerResults.length - m.nestedCV[metric]) < 1e-10);
  }
  const lasso = data.methods.find(m => m.id === "lasso_cv")!;
  assert.equal(lasso.outerResults.filter(r => r.k === 16).length, 2);
  assert.notEqual(lasso.outerResults[1].scores.rmse, lasso.outerResults[4].scores.rmse);
});

test("fold mapping is independent of source record order", () => {
  const a = adaptFeatureSelection(raw, config); const b = adaptFeatureSelection(raw, config);
  attachOuterResults(a, formalRecords);
  attachOuterResults(b, { ...formalRecords, records: [...formalRecords.records].reverse() });
  assert.deepEqual(a.methods.map(m => m.outerResults), b.methods.map(m => m.outerResults));
});

test("reject incomplete, repeated, nonfinite, inconsistent or Holdout fold records", () => {
  for (const change of [
    (rows: typeof formalRecords) => { rows.records = rows.records.filter((r: {method: string; level: string; outer_fold: number}) => !(r.method === "lasso_cv" && r.level === "outer_evaluation" && r.outer_fold === 1)); },
    (rows: typeof formalRecords) => { rows.records.find((r: {level: string}) => r.level === "outer_evaluation").outer_fold = 2; },
    (rows: typeof formalRecords) => { rows.records.find((r: {level: string}) => r.level === "outer_evaluation").selected_k = 25; },
    (rows: typeof formalRecords) => { rows.records.find((r: {level: string}) => r.level === "outer_evaluation").rmse = NaN; },
    (rows: typeof formalRecords) => { rows.records.find((r: {level: string}) => r.level === "outer_evaluation").status = "timeout"; },
    (rows: typeof formalRecords) => { rows.records.find((r: {level: string}) => r.level === "outer_evaluation").validation_range = [15000, 19000]; },
    (rows: typeof formalRecords) => { rows.records.find((r: {level: string}) => r.level === "outer_evaluation").rmse += 1; },
  ]) {
    const copy = structuredClone(formalRecords); change(copy);
    assert.throws(() => attachOuterResults(adaptFeatureSelection(raw, config), copy));
  }
});

test("fold result source is required and its recorded hash is checked", async () => {
  await fixture(async root => {
    const path = resolve(root, FEATURE_SOURCES.records); const contents = await readFile(path);
    await rm(path); assert.equal((await buildFeatureSnapshot(root)).status, "unavailable");
    await writeFile(path, contents.toString() + "\n"); assert.equal((await buildFeatureSnapshot(root)).status, "unavailable");
  });
});

test("snapshot rejects fold scores or k inconsistent with the verified summaries", async () => {
  const s = await buildFeatureSnapshot(repo); assert.equal(s.status, "ready"); if (s.status !== "ready") return;
  const changedScore = structuredClone(s); changedScore.data.methods[4].outerResults[0].scores.rmse += 1;
  assert.throws(() => parseFeatureSnapshot(changedScore));
  const changedK = structuredClone(s); changedK.data.methods[4].outerResults[0].k += 1;
  assert.throws(() => parseFeatureSnapshot(changedK));
});
