import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readdir, readFile, rm, writeFile, chmod } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { once } from "node:events";
import { request as httpRequest } from "node:http";
import { setTimeout as delay } from "node:timers/promises";
import { createComputeService, COMPUTE_LIMITS, type ComputeOptions } from "../server.ts";
import { emptyDraft, profileCsv, validatePlan } from "../../web/lib/custom-dataset.ts";
import { parseValidationReport } from "../../web/lib/custom-validation-result.ts";
import { parseTrainingReport, type SelectionMethod } from "../../web/lib/custom-training-result.ts";
import { parseComparisonReport } from "../../web/lib/custom-comparison-result.ts";
import { createTrainingHandler } from "../../web/lib/custom-validation-service.ts";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const python = process.env.AEP_PYTHON ?? join(repo, ".venv/bin/python");
const token = "test_only_service_" + "0".repeat(40), sessionA = "a".repeat(64), sessionB = "b".repeat(64);
const values = Array.from({ length: 120 }, (_, i) => ({ x: Math.sin(i * .7), z: Math.cos(i * 1.37), w: Math.cos(i * .39) }));
const csv = "x,z,w,target\n" + values.map(v => `${v.x},${v.z},${v.w},${10 + 4*v.x + .7*v.z}`).join("\n");
const config = validatePlan(profileCsv(csv), { ...emptyDraft(), target: "target", features: ["x", "z", "w"], cvSplits: 3 }).config;
const sha = createHash("sha256").update(csv).digest("hex");
function form() { const f = new FormData(); f.append("csv", new File([csv], "fixture.csv")); f.append("config", JSON.stringify(config)); f.append("review", JSON.stringify({ schema_version: 1, artifact_type: "custom_dataset_review", dataset: { sha256: sha }, preprocessing: config, review: { confirmed: true, training_started: false, scope: "configuration_only" } })); return f; }
function local(body: FormData) { return new Request("http://localhost/api/custom-dataset/train", { method: "POST", headers: { Origin: "http://localhost" }, body }); }
async function fixture(run: (f: { root: string; base: string; options: ComputeOptions; post: (operation: string, body: BodyInit, session?: string, method?: string) => Promise<Response> }) => Promise<void>, extra: Partial<ComputeOptions> = {}) {
  const root = await mkdtemp(join(tmpdir(), "aep-backend-test-"));
  const options = { token, repoRoot: repo, pythonPath: python, temporaryRoot: root, ...extra };
  const service = await createComputeService(options);
  try {
    service.server.listen(0, "127.0.0.1"); await once(service.server, "listening");
    const address = service.server.address(); if (!address || typeof address === "string") throw new Error("missing address");
    const base = `http://127.0.0.1:${address.port}`;
    const post = (operation: string, body: BodyInit, session = sessionA, method = "POST") => fetch(base + "/api/custom-dataset/" + operation, { method, headers: { Authorization: "Bearer " + token, "X-AEP-Session": session, ...(typeof body === "string" ? { "Content-Type": "application/json" } : {}) }, body });
    await run({ root, base, options, post });
  } finally { await service.dispose(); assert.deepEqual(await readdir(root), []); await rm(root, { recursive: true, force: true }); }
}
async function experiment(post: (operation: string, body: BodyInit) => Promise<Response>, method: SelectionMethod = "rfecv") {
  const v = parseValidationReport(await (await post("validate", form())).json()); assert.equal(v.status, "valid"); if (v.status !== "valid") throw new Error("invalid fixture");
  const f = form(); f.append("experiment", JSON.stringify({ schema_version: 1, method: "linear_regression", confirmed: true, dataset_sha256: sha, config_sha256: v.config_sha256, selection: { method, k: ["none", "rfecv", "lasso_cv"].includes(method) ? null : 2 } })); return f;
}

test("gateway requires a strong configured service secret and explicit paths", async () => {
  await assert.rejects(createComputeService({ token: "short", repoRoot: repo, pythonPath: python }));
  await assert.rejects(createComputeService({ token, repoRoot: ".", pythonPath: python }));
  await assert.rejects(createComputeService({ token, repoRoot: repo, pythonPath: "python" }));
});
test("health is minimal; all compute routes require service auth, fixed paths, methods and a session", async () => fixture(async ({ base, root }) => {
  const h = await fetch(base + "/healthz"); assert.equal(h.status, 200); assert.deepEqual(await h.json(), { status: "ok" });
  for (const auth of [undefined, "Bearer wrong", token, "bearer " + token]) {
    const r = await fetch(base + "/api/custom-dataset/validate", { method: "POST", headers: auth ? { Authorization: auth } : {} }); assert.equal(r.status, 401); assert.equal(r.headers.get("access-control-allow-origin"), null);
  }
  const headers = { Authorization: "Bearer " + token, "X-AEP-Session": sessionA };
  for (const path of ["/api/custom-dataset/train?module=x", "/api/custom-dataset/%74rain", "/api/custom-dataset/unknown", "/file"]) assert.equal((await fetch(base + path, { method: "POST", headers })).status, 404);
  assert.equal((await fetch(base + "/api/custom-dataset/train", { headers })).status, 405);
  assert.equal((await fetch(base + "/api/custom-dataset/validate", { method: "POST", headers: { Authorization: "Bearer " + token } })).status, 400);
  const dirs = await readdir(root); assert.equal(dirs.length, 1); assert.deepEqual(await readdir(join(root, dirs[0])), []);
}));
test("real HTTP worker completes validation, six-method comparison and all training/inference flows without changing local results", async () => fixture(async ({ post, options }) => {
  const v = parseValidationReport(await (await post("validate", form())).json()); if (v.status !== "valid") assert.fail("invalid");
  const c = form(); c.append("experiment", JSON.stringify({ schema_version: 1, confirmed: true, dataset_sha256: sha, config_sha256: v.config_sha256, methods: ["f_test", "mutual_information", "forward_sfs", "rfe", "lasso_cv", "rfecv"], k_values: [1, 2] }));
  const comparisonResponse = await post("compare", c); assert.equal(comparisonResponse.status, 200); const comparison = parseComparisonReport(await comparisonResponse.json()); if (comparison.status !== "compared") assert.fail("comparison failed"); assert.equal(comparison.points.length, 8); assert.equal(comparison.automatic.length, 2);
  for (const method of ["none", "f_test", "mutual_information", "rfe", "forward_sfs", "rfecv", "lasso_cv"] as const) {
    const f = await experiment(post, method);
    const legacy = parseTrainingReport(await (await createTrainingHandler({ repoRoot: repo, pythonPath: python, temporaryRoot: options.temporaryRoot })(local(f))).json()); if (legacy.status !== "completed") assert.fail("local training failed");
    const response = await post("train", f); assert.equal(response.status, 200); const report = parseTrainingReport(await response.json()); if (report.status !== "completed" || !report.prediction_model) assert.fail("training failed");
    assert.deepEqual(report.selection, legacy.selection); assert.deepEqual(report.cv, legacy.cv); assert.deepEqual(report.holdout, legacy.holdout);
    const m = report.prediction_model, row = report.preview[0], inputs = values[row.source_row - 1];
    const selected = Object.fromEntries(m.features.map(n => [n, inputs[n as keyof typeof inputs]]));
    const single = await post("predict", JSON.stringify({ model_id: m.id, values: selected })); assert.equal(single.status, 200); assert.ok(Math.abs((await single.json()).prediction - row.prediction) < 1e-9);
    const batch = await post("predict-batch", JSON.stringify({ model_id: m.id, features: m.features, rows: [m.features.map(n => selected[n]), m.features.map((n, i) => i ? null : selected[n])] })); assert.equal(batch.status, 200); const b = await batch.json(); assert.equal(b.row_count, 2); assert.ok(Math.abs(b.rows[0].prediction - row.prediction) < 1e-9);
    assert.equal((await post("predict", JSON.stringify({ model_id: m.id }), sessionA, "DELETE")).status, 204);
    assert.equal((await post("predict", JSON.stringify({ model_id: m.id, values: selected }))).status, 410);
  }
}));
test("session B cannot infer from or delete session A's model even with its model handle", async () => fixture(async ({ post }) => {
  const r = await (await post("train", await experiment(post))).json(); const id = r.prediction_model.id;
  assert.equal((await post("train", await experiment(post), sessionB)).status, 200);
  const body = JSON.stringify({ model_id: id, values: { x: .2, z: .3, w: .4 } });
  assert.equal((await post("predict", body, sessionB)).status, 410);
  assert.equal((await post("predict", JSON.stringify({ model_id: id }), sessionB, "DELETE")).status, 204);
  // Only use the selected features, not the original candidate schema.
  const values = Object.fromEntries(r.prediction_model.features.map((n: string) => [n, .2]));
  assert.equal((await post("predict", JSON.stringify({ model_id: id, values }))).status, 200);
}));
test("stale confirmation, unsupported payloads and oversize requests fail before returning any scores", async () => fixture(async ({ post }) => {
  const stale = form(); stale.append("experiment", JSON.stringify({ schema_version: 1, method: "linear_regression", confirmed: true, dataset_sha256: sha, config_sha256: "0".repeat(64), selection: { method: "rfecv", k: null } }));
  const r = await post("train", stale); assert.equal(r.status, 422); const report = await r.json(); assert.equal(report.status, "rejected"); assert.equal(report.holdout, undefined);
  assert.equal((await post("validate", "{}")).status, 415);
  const f = form(); f.set("csv", new File(["x".repeat(COMPUTE_LIMITS.csvBytes + 1)], "large.csv")); assert.equal((await post("validate", f)).status, 413);
  assert.equal((await post("predict", " ".repeat(COMPUTE_LIMITS.bodyBytes + 1))).status, 413);
}));
test("expired sessions are pruned; active session capacity is bounded", async () => {
  let now = 0;
  await fixture(async ({ post }) => {
    const f = await experiment(post); const a = await (await post("train", f)).json(); assert.ok(a.prediction_model);
    assert.equal((await post("train", f, sessionB)).status, 429);
    now = 200000;
    assert.equal((await post("predict", JSON.stringify({ model_id: a.prediction_model.id, values: { x: .2 } }))).status, 410);
    assert.equal((await post("train", f, sessionB)).status, 200);
  }, { sessionLimit: 1, modelTtlMs: 100000, now: () => now });
});
test("failed training does not reserve a model session; chunked oversize bodies release the global slot", async () => fixture(async ({ base, post }) => {
  const f = await experiment(post); const bad = form(); bad.append("experiment", String(f.get("experiment")).replace('"confirmed":true', '"confirmed":false'));
  assert.equal((await post("train", bad)).status, 422);
  assert.equal((await post("train", f, sessionB)).status, 200);
  const status = await new Promise<number>((done, reject) => {
    const r = httpRequest(base + "/api/custom-dataset/validate", { method: "POST", headers: { Authorization: "Bearer " + token, "X-AEP-Session": sessionA, "Content-Type": "multipart/form-data; boundary=test", "Transfer-Encoding": "chunked" } }, response => { response.resume(); response.on("end", () => done(response.statusCode!)); });
    r.on("error", reject); r.setTimeout(5000, () => r.destroy(new Error("oversize request hung")));
    r.end(Buffer.alloc(COMPUTE_LIMITS.bodyBytes + 1, 32));
  });
  assert.equal(status, 413); assert.equal((await post("validate", "{}")).status, 415);
}, { sessionLimit: 1 }));

async function fakePython(root: string) {
  const filename = join(root, "slow-python"), marker = join(root, "worker-pid");
  await writeFile(filename, `#!${python}\nimport os,time\nopen(${JSON.stringify(marker)},'w').write(str(os.getpid()))\ntime.sleep(5)\nprint('{}')\n`); await chmod(filename, 0o700); return { filename, marker };
}
async function waitForFile(path: string) { for (let i = 0; i < 150; i++) { try { return Number(await readFile(path, "utf8")); } catch { await delay(20); } } throw new Error("worker did not start"); }
test("global concurrency, cancellation and cleanup apply across operations; disconnected workers are killed", async () => {
  const root = await mkdtemp(join(tmpdir(), "aep-slow-worker-"));
  try {
    const fake = await fakePython(root);
    await fixture(async ({ base, post }) => {
      const abort = new AbortController();
      const pending = fetch(base + "/api/custom-dataset/validate", { method: "POST", headers: { Authorization: "Bearer " + token, "X-AEP-Session": sessionA }, body: form(), signal: abort.signal }).then(() => false, () => true);
      const pid = await waitForFile(fake.marker);
      assert.equal((await post("compare", form())).status, 429);
      abort.abort(); assert.equal(await pending, true);
      for (let i = 0; i < 100; i++) { try { process.kill(pid, 0); await delay(20); } catch { break; } }
      assert.throws(() => process.kill(pid, 0));
      for (let i = 0; i < 50; i++) { const r = await post("validate", "{}"); if (r.status !== 429) { assert.equal(r.status, 415); return; } await delay(20); }
      assert.fail("busy slot did not release");
    }, { pythonPath: fake.filename });
  } finally { await rm(root, { recursive: true, force: true }); }
});
test("worker timeout is bounded and private uploads are cleared", async () => {
  const root = await mkdtemp(join(tmpdir(), "aep-timeout-worker-"));
  try { const fake = await fakePython(root); await fixture(async ({ post }) => { assert.equal((await post("validate", form())).status, 504); const pid = Number(await readFile(fake.marker, "utf8")); assert.throws(() => process.kill(pid, 0)); }, { pythonPath: fake.filename, workerTimeoutMs: 1500 }); }
  finally { await rm(root, { recursive: true, force: true }); }
});
