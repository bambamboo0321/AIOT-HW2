import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile, symlink } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { adaptEvaluation, parseSavedPredictions, buildEvaluationSnapshot, parseEvaluationSnapshot, EVALUATION_SOURCES } from "../lib/evaluation-data.ts";
import { writeSnapshot } from "../lib/write-snapshot.ts";
const web = resolve(dirname(fileURLToPath(import.meta.url)), ".."); const repo = resolve(web, "..");
const json = async (path: string) => JSON.parse(await readFile(resolve(repo, path), "utf8"));
const docs = await Promise.all([EVALUATION_SOURCES.dashboardModels, EVALUATION_SOURCES.dashboardIntervals, EVALUATION_SOURCES.metrics, EVALUATION_SOURCES.intervals, EVALUATION_SOURCES.plan].map(json));
const csv = await readFile(resolve(repo, EVALUATION_SOURCES.predictions), "utf8"); const rows = parseSavedPredictions(csv);
const adapt = (documents = docs, predictions = rows) => adaptEvaluation(...documents as [unknown, unknown, unknown, unknown, unknown], predictions);
async function fixture(run: (root: string) => Promise<void>) {
  const root = await mkdtemp(resolve(web, ".test-tmp-evaluation-"));
  try { for (const path of Object.values(EVALUATION_SOURCES)) { await mkdir(dirname(resolve(root, path)), { recursive: true }); await writeFile(resolve(root, path), await readFile(resolve(repo, path))); } await run(root); }
  finally { await rm(root, { recursive: true, force: true }); }
}

test("preserves exact formal training and holdout scores and both fixed feature sets", () => {
  const data = adapt(); assert.equal(data.holdoutRows, rows.length);
  for (const model of data.models) {
    const original = docs[2].models[model.id];
    for (const metric of ["rmse", "mae", "mse", "r2"] as const) { assert.equal(model.training[metric], original.training_metrics[metric]); assert.equal(model.holdout[metric], original.holdout_metrics[metric]); }
    assert.deepEqual(model.features, original.selected_features);
  }
  assert.equal(data.models[0].features.length, docs[2].models.baseline.feature_count);
  assert.equal(data.models[1].features.length, docs[2].models.selected.feature_count);
});

test("histogram contains every residual including extrema, on identical bins", () => {
  const data = adapt(); assert.equal(data.histogram.length, 20);
  for (const id of ["baseline", "selected"] as const) {
    const expected = data.histogram.map(() => 0);
    for (const row of rows) {
      const error = row.actual - row[id].prediction;
      const index = data.histogram.findIndex((b, i) => error >= b.lower && (i === data.histogram.length - 1 ? error <= b.upper : error < b.upper));
      assert.ok(index >= 0); expected[index]++;
    }
    assert.deepEqual(data.histogram.map(b => b[id]), expected);
    assert.equal(expected.reduce((a, b) => a + b, 0), rows.length);
  }
});

test("residual sign, linear quantiles and exact max agree with saved predictions", () => {
  const data = adapt();
  for (const model of data.models) {
    const errors = rows.map(r => r.actual - r[model.id].prediction).sort((a, b) => a - b);
    assert.equal(model.residual.median, errors[Math.floor(errors.length / 2)]);
    assert.equal(model.residual.maxAbsolute, Math.max(Math.abs(errors[0]), Math.abs(errors.at(-1)!)));
    assert.equal(model.residual.underpredictions, errors.filter(e => e > 0).length);
    assert.equal(model.residual.overpredictions, errors.filter(e => e < 0).length);
    assert.equal(model.residual.exact, errors.filter(e => e === 0).length);
    const position = (errors.length - 1) * 0.05;
    assert.equal(model.residual.p05, errors[Math.floor(position)] + (errors[Math.ceil(position)] - errors[Math.floor(position)]) * (position - Math.floor(position)));
  }
});

test("interval coverage counts and saved width summaries retain all Holdout observations", () => {
  const data = adapt();
  for (const model of data.models) {
    const original = docs[3].models[model.id]; const p = model.interval;
    assert.equal(p.covered, original.covered_count); assert.equal(p.above, original.above_upper_count); assert.equal(p.below, original.below_lower_count);
    assert.equal(p.covered + p.above + p.below, rows.length);
    assert.equal(p.coverage, original.empirical_coverage); assert.equal(p.averageWidth, original.average_width); assert.equal(p.widthSD, original.width_standard_deviation);
  }
  assert.ok(rows.some(r => r.baseline.lower < 0));
});

test("rejects missing, reordered or nonfinite CSV fields and inverted intervals", () => {
  const bad = [csv.replace("Row Index", "Index"), csv.replace(",370,", ",NaN,"), csv.replace(",370,", ",,"), csv.replace(",370,", ',"370",'), csv.replace("79.33052568517022", "999999999")];
  for (const text of bad) assert.throws(() => parseSavedPredictions(text));
  assert.deepEqual(parseSavedPredictions(csv.replace(/\n/g, "\r\n")), rows);
});

test("rejects partial, reordered or mismatched Holdout rows rather than partial scores", () => {
  assert.throws(() => adapt(docs, rows.slice(1)));
  assert.throws(() => adapt(docs, [...rows].reverse()));
  const changed = structuredClone(rows); changed[0].actual += 10; assert.throws(() => adapt(docs, changed));
  const shifted = structuredClone(rows); shifted[0].index--; assert.throws(() => adapt(docs, shifted));
});

test("fails closed on incompatible plans, stale metrics, mislabeled units or PI substitution", () => {
  for (const change of [
    (d: typeof docs) => { d[0].schema_version = 99; },
    (d: typeof docs) => { d[1].data.interval_type = "mean_confidence_interval"; },
    (d: typeof docs) => { d[0].units.rmse = "kWh"; },
    (d: typeof docs) => { d[4].plan.method_or_parameter_selection_using_holdout = true; },
    (d: typeof docs) => { d[3].models.baseline.covered_count++; },
    (d: typeof docs) => { d[2].models.selected.holdout_metrics.rmse = NaN; },
    (d: typeof docs) => { d[2].models.selected.holdout_metrics.mae += 1; },
    (d: typeof docs) => { d[3].models.selected.training_diagnostics.holdout_residuals_used = true; },
  ]) { const copy = structuredClone(docs); change(copy); assert.throws(() => adapt(copy)); }
});

test("snapshot is deterministic with seven traceable sources and only aggregated residual data", async () => {
  const a = await buildEvaluationSnapshot(repo); const b = await buildEvaluationSnapshot(repo);
  assert.equal(a.status, "ready"); assert.deepEqual(a, b); assert.deepEqual(parseEvaluationSnapshot(a), a); assert.equal(a.sources.length, Object.keys(EVALUATION_SOURCES).length);
  for (const s of a.sources) assert.equal(s.sha256, createHash("sha256").update(await readFile(resolve(repo, s.path))).digest("hex"));
  for (const forbidden of ["coefficients", "intercept", "training_diagnostics", "singular_values", "Timestamp", "Row Index", "/Users/"]) assert.ok(!JSON.stringify(a).includes(forbidden));
});

test("missing sources, out-of-repository symlinks and stale source digests are explicit failures", async () => {
  await fixture(async root => {
    const path = resolve(root, EVALUATION_SOURCES.predictions); const bytes = await readFile(path);
    await rm(path); const absent = await buildEvaluationSnapshot(root); assert.equal(absent.status, "unavailable"); assert.ok(!("data" in absent));
    await symlink(resolve(repo, EVALUATION_SOURCES.predictions), path); assert.equal((await buildEvaluationSnapshot(root)).status, "unavailable");
    await rm(path); await writeFile(path, bytes.toString() + "\n"); assert.equal((await buildEvaluationSnapshot(root)).status, "unavailable");
  });
});

test("runtime validation rejects broken totals, nonfinite scores and corrupt histogram boundaries", async () => {
  const original = await buildEvaluationSnapshot(repo); assert.equal(original.status, "ready"); if (original.status !== "ready") return;
  assert.throws(() => parseEvaluationSnapshot({ ...original, schemaVersion: 2 }));
  const a = structuredClone(original); a.data.histogram[0].baseline++; assert.throws(() => parseEvaluationSnapshot(a));
  const b = structuredClone(original); b.data.models[0].interval.covered++; assert.throws(() => parseEvaluationSnapshot(b));
  const c = structuredClone(original); c.data.models[0].holdout.rmse = Infinity; assert.throws(() => parseEvaluationSnapshot(c));
  const d = structuredClone(original); d.data.histogram[1].lower++; assert.throws(() => parseEvaluationSnapshot(d));
});

test("evaluation sync never alters existing Overview and Feature Selection snapshots", async () => {
  await fixture(async root => {
    await mkdir(resolve(root, "data"));
    for (const name of ["overview", "feature-selection"]) await writeFile(resolve(root, `data/${name}.json`), name);
    const s = await buildEvaluationSnapshot(root); await writeSnapshot(root, s, "model-evaluation");
    const first = await readFile(resolve(root, "data/model-evaluation.json")); await writeSnapshot(root, s, "model-evaluation"); assert.deepEqual(first, await readFile(resolve(root, "data/model-evaluation.json")));
    for (const name of ["overview", "feature-selection"]) assert.equal(await readFile(resolve(root, `data/${name}.json`), "utf8"), name);
  });
});
