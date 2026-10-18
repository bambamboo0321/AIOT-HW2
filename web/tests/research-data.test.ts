import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { adaptCore, buildSnapshot, parseSnapshot, SOURCE_PATHS } from "../lib/research-data.ts";
import { writeSnapshot } from "../lib/write-snapshot.ts";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(web, "..");
const json = async (path: string) => JSON.parse(await readFile(resolve(repo, path), "utf8"));
const originals = await Promise.all([SOURCE_PATHS.dataset, SOURCE_PATHS.models, SOURCE_PATHS.intervals].map(json));

async function fixture(run: (root: string) => Promise<void>) {
  const root = await mkdtemp(resolve(web, ".test-tmp-"));
  try {
    for (const path of Object.values(SOURCE_PATHS)) {
      await mkdir(dirname(resolve(root, path)), { recursive: true });
      await writeFile(resolve(root, path), await readFile(resolve(repo, path)));
    }
    await run(root);
  } finally { await rm(root, { recursive: true, force: true }); }
}

test("adapter reads dataset and both model metrics from actual JSON without rounding", () => {
  const data = adaptCore(...originals as [unknown, unknown, unknown]);
  assert.equal(data.dataset.records, originals[0].data.dataset.shape.rows);
  assert.equal(data.dataset.rawFeatures, originals[0].data.dataset.original_input_feature_count);
  assert.equal(data.dataset.modelingFeatures, originals[0].data.baseline_feature_count);
  for (const model of data.models) {
    assert.equal(model.rmse, originals[1].data.models[model.id].holdout_metrics.rmse);
    assert.equal(model.mae, originals[1].data.models[model.id].holdout_metrics.mae);
    assert.equal(model.r2, originals[1].data.models[model.id].holdout_metrics.r2);
    assert.equal(model.coverage, originals[2].data.models[model.id].empirical_coverage);
    assert.equal(model.averageWidth, originals[2].data.models[model.id].average_width);
  }
});

test("no hardcoded model results: input metric changes propagate", () => {
  const documents = structuredClone(originals);
  documents[1].data.models.selected.holdout_metrics.rmse = 123.456789;
  const result = adaptCore(...documents as [unknown, unknown, unknown]);
  assert.equal(result.models[1].rmse, 123.456789);
});

test("rejects incompatible schema, nonfinite metrics and confidence-interval substitution", () => {
  for (const change of [
    (docs: typeof originals) => { docs[0].schema_version = 99; },
    (docs: typeof originals) => { delete docs[1].data.models.selected.holdout_metrics.rmse; },
    (docs: typeof originals) => { docs[1].data.models.baseline.holdout_metrics.rmse = NaN; },
    (docs: typeof originals) => { docs[2].data.interval_type = "confidence_interval"; },
    (docs: typeof originals) => { docs[2].data.models.selected.empirical_coverage = 95; },
    (docs: typeof originals) => { docs[2].data.plan.holdout_rows += 1; },
  ]) {
    const docs = structuredClone(originals); change(docs);
    assert.throws(() => adaptCore(...docs as [unknown, unknown, unknown]));
  }
});

test("snapshot is deterministic, traceable and contains no individual rows or model objects", async () => {
  const before = await Promise.all(Object.values(SOURCE_PATHS).map(p => readFile(resolve(repo, p))));
  const first = await buildSnapshot(repo); const second = await buildSnapshot(repo);
  assert.deepEqual(first, second); assert.equal(first.status, "ready");
  assert.equal(first.sources.length, Object.keys(SOURCE_PATHS).length);
  for (const source of first.sources) {
    assert.ok(Object.values(SOURCE_PATHS).includes(source.path as typeof SOURCE_PATHS[keyof typeof SOURCE_PATHS]));
    assert.equal(source.sha256, createHash("sha256").update(await readFile(resolve(repo, source.path))).digest("hex"));
  }
  const after = await Promise.all(Object.values(SOURCE_PATHS).map(p => readFile(resolve(repo, p))));
  assert.deepEqual(before, after);
  assert.ok(!JSON.stringify(first).includes("coefficients"));
  assert.ok(!JSON.stringify(first).includes("training_diagnostics"));
  assert.ok(!JSON.stringify(first).includes("/Users/"));
  assert.deepEqual(parseSnapshot(first), first);
});

test("Phase 3D uses fixed-subset CV, not selector CV or holdout", async () => {
  const snapshot = await buildSnapshot(repo); const phase3d = await json(SOURCE_PATHS.exploration);
  assert.equal(snapshot.status, "ready"); if (snapshot.status !== "ready") return;
  const data = snapshot.data.exploration;
  assert.equal(data.available, true); if (!data.available) return;
  assert.equal(data.bestRmse, phase3d.random_search.best.rmse_wh);
  assert.equal(data.fixedMiRmse, phase3d.fixed_combinations.mutual_information.rmse_wh);
  assert.equal(data.sampleCount, phase3d.random_search.sample_count);
  assert.equal(data.folds, phase3d.conditions.cv_splits);
  assert.deepEqual(data.bestFeatures, phase3d.random_search.best.features);
  assert.notEqual(data.fixedMiRmse, phase3d.selector_cv.find((m: { method: string }) => m.method === "mutual_information").rmse);
});

test("missing mandatory data creates explicit unavailable state, never placeholder metrics", async () => {
  await fixture(async root => {
    await rm(resolve(root, SOURCE_PATHS.models));
    const snapshot = await buildSnapshot(root);
    assert.equal(snapshot.status, "unavailable");
    if (snapshot.status === "unavailable") assert.ok(snapshot.issues.some(issue => issue.includes(SOURCE_PATHS.models)));
    assert.ok(!("data" in snapshot));
  });
});

test("missing Phase 3D leaves Overview available with an explicit missing-data message", async () => {
  await fixture(async root => {
    await rm(resolve(root, SOURCE_PATHS.exploration));
    const snapshot = await buildSnapshot(root);
    assert.equal(snapshot.status, "ready");
    if (snapshot.status === "ready") assert.equal(snapshot.data.exploration.available, false);
  });
});

test("rejects malformed files and symlinks pointing outside the repository", async () => {
  await fixture(async root => {
    await writeFile(resolve(root, SOURCE_PATHS.dataset), "{bad json");
    assert.equal((await buildSnapshot(root)).status, "unavailable");
    await rm(resolve(root, SOURCE_PATHS.dataset));
    await symlink(resolve(repo, SOURCE_PATHS.dataset), resolve(root, SOURCE_PATHS.dataset));
    assert.equal((await buildSnapshot(root)).status, "unavailable");
  });
});

test("runtime loader rejects incomplete or corrupted snapshots", async () => {
  const snapshot = await buildSnapshot(repo);
  assert.equal(snapshot.status, "ready");
  if (snapshot.status !== "ready") return;
  const copy = structuredClone(snapshot) as unknown as Record<string, unknown>;
  assert.throws(() => parseSnapshot({ ...copy, schemaVersion: 2 }));
  assert.throws(() => parseSnapshot({ ...copy, data: {} }));
  assert.throws(() => parseSnapshot({ ...copy, data: { ...snapshot.data, nominalCoverage: 95 } }));
  assert.throws(() => parseSnapshot({ ...copy, data: { ...snapshot.data, models: [] } }));
});


test("sync writes deterministic UTF-8/LF snapshots only inside its data directory", async () => {
  await fixture(async root => {
    const snapshot = await buildSnapshot(root);
    await writeSnapshot(root, snapshot);
    const before = await readFile(resolve(root, "data/overview.json"));
    await writeSnapshot(root, snapshot);
    const after = await readFile(resolve(root, "data/overview.json"));
    assert.deepEqual(before, after);
    assert.ok(!before.includes(13));
    assert.deepEqual(parseSnapshot(JSON.parse(before.toString("utf8"))), snapshot);
  });
});

test("sync refuses directory and output-file symlinks without overwriting targets", async () => {
  await fixture(async root => {
    const snapshot = await buildSnapshot(root);
    await mkdir(resolve(root, "preserved"));
    const target = resolve(root, "preserved/marker");
    await writeFile(target, "preserve");
    await symlink(resolve(root, "preserved"), resolve(root, "data"));
    await assert.rejects(() => writeSnapshot(root, snapshot));
    await rm(resolve(root, "data"));
    await mkdir(resolve(root, "data"));
    await symlink(target, resolve(root, "data/overview.json"));
    await assert.rejects(() => writeSnapshot(root, snapshot));
    assert.equal(await readFile(target, "utf8"), "preserve");
  });
});
