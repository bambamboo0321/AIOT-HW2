import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildPredictionSnapshot, parsePredictionSnapshot } from "../lib/prediction-data.ts";
import { parseSavedPredictions, EVALUATION_SOURCES } from "../lib/evaluation-data.ts";
import { predictionWindow, parsePredictionWindow, parseWindowQuery, windowBounds, windowScores, timePosition } from "../lib/prediction-window.ts";
import { writeSnapshot } from "../lib/write-snapshot.ts";
const web = resolve(dirname(fileURLToPath(import.meta.url)), ".."); const repo = resolve(web, "..");
const snapshot = await buildPredictionSnapshot(repo); if (snapshot.status !== "ready") throw new Error(snapshot.issues.join("\n"));
const { meta, rows } = snapshot.data;
const savedRows = parseSavedPredictions(await readFile(resolve(repo, EVALUATION_SOURCES.predictions), "utf8"));
const plan = JSON.parse(await readFile(resolve(repo, EVALUATION_SOURCES.plan), "utf8"));

test("all saved predictions, bounds and source times are retained exactly without raw features", () => {
  assert.equal(rows.length, meta.totalRows);
  for (const [index, r] of rows.entries()) {
    assert.equal(r.position, index + 1); assert.equal(r.timestamp, savedRows[index].timestamp); assert.equal(r.actual, savedRows[index].actual);
    assert.deepEqual(r.baseline, savedRows[index].baseline); assert.deepEqual(r.selected, savedRows[index].selected);
    assert.deepEqual(Object.keys(r).sort(), ["position", "timestamp", "actual", "baseline", "selected"].sort());
  }
  assert.ok(rows.some(r => r.selected.lower < 0));
});

test("default window follows the predeclared Phase 3C interval figure", () => {
  assert.equal(meta.defaultStart, plan.plan.plot_window.holdout_start_position); assert.equal(meta.defaultSize, plan.plan.plot_window.consecutive_samples);
  const initial = predictionWindow(rows, meta, meta.defaultStart, meta.defaultSize);
  assert.deepEqual(initial.rows, rows.slice(meta.defaultStart, meta.defaultStart + meta.defaultSize));
  assert.equal(initial.rows.length, 144); assert.deepEqual(parsePredictionWindow(initial), initial);
});

test("window boundaries retain the final observation without padding or dropping peaks", () => {
  for (const size of [72, 144, 288]) {
    const last = predictionWindow(rows, meta, rows.length - 1, size);
    assert.equal(last.rows.length, size); assert.equal(last.rows.at(-1)!.position, rows.length);
    assert.deepEqual(last.rows, rows.slice(-size)); assert.equal(last.start, rows.length - size);
    const start = rows.reduce((max, r, i) => r.actual > rows[max].actual ? i : max, 0);
    const withPeak = predictionWindow(rows, meta, start, size);
    assert.equal(Math.max(...withPeak.rows.map(r => r.actual)), rows[start].actual);
  }
  assert.deepEqual(windowBounds(0, 72, 1), { start: 0, size: 72 });
});

test("window statistics match explicit sums and never replace full Holdout metrics", () => {
  const copy = structuredClone(meta); const w = predictionWindow(rows, meta, 144, 72);
  for (const id of ["baseline", "selected"] as const) {
    const errors = w.rows.map(r => r.actual - r[id].prediction);
    assert.equal(w.stats[id].rmse, Math.sqrt(errors.reduce((sum, e) => sum + e * e, 0) / 72));
    assert.equal(w.stats[id].mae, errors.reduce((sum, e) => sum + Math.abs(e), 0) / 72);
    assert.equal(w.stats[id].covered + w.stats[id].above + w.stats[id].below, 72);
  }
  assert.deepEqual(meta, copy);
});

test("constant or one-row Actual returns an undefined R2 instead of a made-up score", () => {
  const single = windowScores([rows[0]], "baseline"); assert.equal(single.r2, null);
  const constant = rows.slice(0, 72).map(r => ({ ...r, actual: 10 }));
  assert.equal(windowScores(constant, "selected").r2, null);
  assert.throws(() => windowScores([], "selected"));
});

test("coverage includes equality at either interval boundary", () => {
  const row = structuredClone(rows[0]); row.actual = row.baseline.upper;
  assert.equal(windowScores([row], "baseline").covered, 1);
  row.actual = row.baseline.lower; assert.equal(windowScores([row], "baseline").covered, 1);
});

test("date navigation uses source clock values with exact ten-minute alignment", () => {
  for (const index of [0, 1, 144, rows.length - 1]) assert.equal(timePosition(rows[index].timestamp.slice(0, 16), meta), index);
  assert.throws(() => timePosition("2016-04-30T08:21", meta));
  assert.throws(() => timePosition("2000-01-01T00:00", meta));
  assert.throws(() => timePosition("bad", meta));
});

test("read-only API parameters reject invalid, huge, duplicate or unknown inputs", () => {
  for (const query of ["start=-1", "start=1.5", "start=NaN", "start=99999", "size=3947", "size=0", "size=1000000000", "start=1&start=2", "size=72&size=144", "path=/etc/passwd", "start=01"]) assert.throws(() => parseWindowQuery(new URLSearchParams(query), rows.length));
  assert.deepEqual(parseWindowQuery(new URLSearchParams(), rows.length), { start: 0, size: 144 });
  assert.deepEqual(parseWindowQuery(new URLSearchParams(`start=${rows.length - 1}&size=288`), rows.length), { start: rows.length - 288, size: 288 });
});

test("window payload validation rejects swapped rows, altered scores and malformed bounds", () => {
  const w = predictionWindow(rows, meta, 0, 144);
  const a = structuredClone(w); a.rows.reverse(); assert.throws(() => parsePredictionWindow(a));
  const b = structuredClone(w); b.stats.baseline.rmse++; assert.throws(() => parsePredictionWindow(b));
  const c = structuredClone(w); c.rows[0].selected.lower = c.rows[0].selected.upper + 1; assert.throws(() => parsePredictionWindow(c));
  const d = structuredClone(w); d.rows[0].actual = Infinity; assert.throws(() => parsePredictionWindow(d));
  const e = structuredClone(w); e.size = 288; assert.throws(() => parsePredictionWindow(e));
});

test("snapshot is deterministic and rejects modified source metrics, row order and digests", async () => {
  assert.deepEqual(await buildPredictionSnapshot(repo), snapshot); assert.deepEqual(parsePredictionSnapshot(snapshot), snapshot);
  const a = structuredClone(snapshot); a.data.meta.sourceDigest = "0".repeat(64); assert.throws(() => parsePredictionSnapshot(a));
  const b = structuredClone(snapshot); b.data.rows[0].timestamp = b.data.rows[1].timestamp; assert.throws(() => parsePredictionSnapshot(b));
  const c = structuredClone(snapshot); c.data.meta.models[0].holdout.rmse++; assert.throws(() => parsePredictionSnapshot(c));
  for (const name of ["leverage", "coefficients", "singular_values", "/Users/"]) assert.ok(!JSON.stringify(snapshot).toLowerCase().includes(name.toLowerCase()));
});

test("missing sources yield explicit unavailable state without substitute rows", async () => {
  const root = await mkdtemp(resolve(web, ".test-tmp-prediction-"));
  try { const s = await buildPredictionSnapshot(root); assert.equal(s.status, "unavailable"); assert.ok(!("data" in s)); assert.deepEqual(parsePredictionSnapshot(s), s); }
  finally { await rm(root, { recursive: true, force: true }); }
});

test("initial client payload contains just one window, never the complete stored series", () => {
  const initial = predictionWindow(rows, meta, 0, 144);
  assert.ok(initial.rows.length < meta.totalRows); assert.equal(initial.rows.length, 144);
  assert.equal(Object.hasOwn(initial, "sources"), false);
});

test("prediction sync writes only its own derived snapshot and leaves previous pages intact", async () => {
  const root = await mkdtemp(resolve(web, ".test-tmp-prediction-"));
  try {
    await mkdir(resolve(root, "data"));
    for (const name of ["overview", "feature-selection", "model-evaluation"]) await writeFile(resolve(root, `data/${name}.json`), name);
    await writeSnapshot(root, snapshot, "prediction-explorer");
    const first = await readFile(resolve(root, "data/prediction-explorer.json")); await writeSnapshot(root, snapshot, "prediction-explorer");
    assert.deepEqual(first, await readFile(resolve(root, "data/prediction-explorer.json")));
    for (const name of ["overview", "feature-selection", "model-evaluation"]) assert.equal(await readFile(resolve(root, `data/${name}.json`), "utf8"), name);
  } finally { await rm(root, { recursive: true, force: true }); }
});
