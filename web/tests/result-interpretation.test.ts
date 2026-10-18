import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { explainR2, highestObservation } from "../lib/result-interpretation.ts";
import { predictionWindow } from "../lib/prediction-window.ts";
import type { PredictionSnapshot } from "../lib/prediction-types.ts";

const snapshot = JSON.parse(await readFile(new URL("../data/prediction-explorer.json", import.meta.url), "utf8")) as PredictionSnapshot;
if (snapshot.status !== "ready") throw new Error("需使用保存的真實預測驗證解讀");
const { meta, rows } = snapshot.data;

test("R2 interpretation separates undefined, negative, zero and positive scores without calling them accuracy", () => {
  assert.match(explainR2(null), /無法定義/);
  assert.match(explainR2(-0.001), /高於/);
  assert.match(explainR2(0), /等於/);
  assert.match(explainR2(0.001), /低於/);
  assert.match(explainR2(1), /低於/);
  for (const value of [-1, 0, 0.1, 1]) assert.match(explainR2(value), /事後比較基準/);
});

test("the screenshot window reports predictions at the actual maximum, rather than each model's maximum", () => {
  const w = predictionWindow(rows, meta, 3744, 72);
  const peak = highestObservation(w.rows);
  assert.equal(w.rows[0].timestamp, "2016-05-26T08:20:00");
  assert.equal(peak.actual, 850);
  assert.ok(w.rows.includes(peak));
  assert.equal(peak.actual, Math.max(...w.rows.map(r => r.actual)));
  for (const id of ["baseline", "selected"] as const) {
    assert.ok(w.stats[id].r2! < 0);
    assert.equal(w.stats[id].covered, 58);
    assert.equal(w.stats[id].above + w.stats[id].below, 14);
    assert.ok(peak.actual > peak[id].upper);
    assert.notEqual(peak[id].prediction, Math.max(...w.rows.map(r => r[id].prediction)));
  }
});

test("maximum observation follows resized and tail windows and preserves original row values", () => {
  const original = structuredClone(rows);
  for (const size of [72, 144, 288]) for (const start of [0, 144, rows.length - 1]) {
    const w = predictionWindow(rows, meta, start, size);
    const peak = highestObservation(w.rows);
    assert.ok(peak.position >= w.rows[0].position && peak.position <= w.rows.at(-1)!.position);
    assert.equal(peak.actual, Math.max(...w.rows.map(r => r.actual)));
  }
  assert.deepEqual(rows, original);
});

test("ties use the first actual maximum and an empty input cannot fabricate an observation", () => {
  const first = { ...rows[0], actual: 100 };
  const second = { ...rows[1], actual: 100 };
  assert.equal(highestObservation([first, second]), first);
  assert.equal(highestObservation([first]), first);
  assert.throws(() => highestObservation([]));
});
