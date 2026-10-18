import { test } from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { createComputeRoute } from "../lib/custom-compute-proxy.ts";
import { createComputeService } from "../../backend/server.ts";
import { emptyDraft, profileCsv, validatePlan } from "../lib/custom-dataset.ts";
import { parseTrainingReport } from "../lib/custom-training-result.ts";
const env = { AEP_BACKEND_URL: "https://compute.example", AEP_BACKEND_TOKEN: "backend_" + "a".repeat(40), AEP_SESSION_SECRET: "cookie_" + "b".repeat(40), AEP_PUBLIC_ORIGIN: "https://project.example" };
const fallback = async () => Response.json({ local: true });
function request(body: BodyInit = "{}", extra: Record<string, string> = {}) { return new Request("https://project.example/api/custom-dataset/predict", { method: "POST", body, headers: { Origin: env.AEP_PUBLIC_ORIGIN, ...(typeof body === "string" ? { "Content-Type": "application/json" } : {}), ...extra } }); }
function form() { const f = new FormData(); f.append("csv", new File(["x,y\n1,2\n2,3"], "data.csv")); f.append("config", "{}"); f.append("review", "{}"); return f; }
const cookie = (r: Response) => r.headers.get("set-cookie")!.split(";")[0];
test("local mode preserves the exact original handler; partial or Vercel configuration fails closed", async () => {
  const r = request(); let seen: Request | undefined;
  assert.deepEqual(await (await createComputeRoute(async q => { seen = q; return fallback(); }, "validate", { env: {} })(r)).json(), { local: true }); assert.equal(seen, r);
  for (const settings of [{ VERCEL: "1" }, { AEP_BACKEND_URL: env.AEP_BACKEND_URL }, { ...env, AEP_BACKEND_URL: "http://compute.example" }, { ...env, AEP_BACKEND_URL: "https://compute.example/path" }, { ...env, AEP_SESSION_SECRET: env.AEP_BACKEND_TOKEN }]) assert.equal((await createComputeRoute(fallback, "validate", { env: settings })(request(form()))).status, 503);
});
test("remote mode rejects cross-site, missing-origin and spoofed-host requests before forwarding", async () => {
  let calls = 0; const h = createComputeRoute(fallback, "validate", { env, fetcher: async () => { calls++; return Response.json({}); } });
  const cases: Record<string,string>[] = [{ Origin: "https://evil.example" }, { Host: "evil.example", "X-Forwarded-Host": "project.example" }, { "Sec-Fetch-Site": "cross-site" }];
  for (const headers of cases) assert.equal((await h(request(form(), headers))).status, 403);
  const missing = request(form()); missing.headers.delete("origin"); assert.equal((await h(missing)).status, 403); assert.equal(calls, 0);
});
test("signed HttpOnly session is stable, isolated from client headers, rotated on tamper, and absent/expired sessions cannot predict", async () => {
  const calls: RequestInit[] = []; let now = 1_800_000_000_000;
  const fetcher: typeof fetch = async (_, init) => { calls.push(init!); return Response.json({ ok: true }, { headers: { "Set-Cookie": "backend-secret=bad", "X-Internal": "hidden" } }); };
  const h = createComputeRoute(fallback, "validate", { env, fetcher, now: () => now });
  const first = await h(request(form(), { Authorization: "Bearer browser", "X-AEP-Session": "c".repeat(64), Cookie: "unrelated=secret" })); assert.equal(first.status, 200); const c = cookie(first);
  assert.match(first.headers.get("set-cookie")!, /HttpOnly; SameSite=Strict; Max-Age=1800; Secure/); assert.equal(first.headers.get("x-internal"), null); assert.ok(!first.headers.get("set-cookie")!.includes("backend-secret"));
  const second = await h(request(form(), { Cookie: c })); assert.equal(cookie(second), c);
  const forwarded = new Headers(calls[0].headers), secondHeaders = new Headers(calls[1].headers);
  assert.equal(forwarded.get("authorization"), "Bearer " + env.AEP_BACKEND_TOKEN); assert.equal(forwarded.get("cookie"), null); assert.notEqual(forwarded.get("x-aep-session"), "c".repeat(64)); assert.equal(secondHeaders.get("x-aep-session"), forwarded.get("x-aep-session"));
  await h(request(form(), { Cookie: c + "tampered" })); assert.notEqual(new Headers(calls[2].headers).get("x-aep-session"), forwarded.get("x-aep-session"));
  const p = createComputeRoute(fallback, "predict", { env, fetcher, now: () => now });
  assert.equal((await p(request())).status, 410); assert.equal((await p(request("{}", { Cookie: c }))).status, 200);
  now += 1_801_000; assert.equal((await p(request("{}", { Cookie: c }))).status, 410);
  assert.ok(!(await first.text()).includes(env.AEP_BACKEND_TOKEN));
});
test("proxy bounds multipart and streamed inputs before forwarding", async () => {
  let calls = 0; const h = createComputeRoute(fallback, "validate", { env, fetcher: async () => { calls++; return Response.json({}); } });
  assert.equal((await h(request(form(), { "Content-Length": "999999999" }))).status, 413);
  const f = form(); f.set("csv", new File(["x".repeat(3*1024*1024+1)], "large.csv")); assert.equal((await h(request(f))).status, 413);
  assert.equal((await h(request("x".repeat(4*1024*1024+1)))).status, 413); assert.equal(calls, 0);
});
test("upstream authentication, invalid/oversized responses and timeouts return bounded public errors", async () => {
  for (const [upstream, status] of [[Response.json({ secret: env.AEP_BACKEND_TOKEN }, { status: 401 }), 503], [new Response("not JSON"), 502], [new Response("x".repeat(4*1024*1024+1)), 502]] as const) {
    const r = await createComputeRoute(fallback, "validate", { env, fetcher: async () => upstream })(request(form())); assert.equal(r.status, status); assert.ok(!(await r.text()).includes(env.AEP_BACKEND_TOKEN));
  }
  const fetcher: typeof fetch = async (_, init) => new Promise((_, reject) => { init!.signal!.addEventListener("abort", () => reject(new Error("private diagnostic")), { once: true }); });
  const r = await createComputeRoute(fallback, "validate", { env, fetcher, timeoutMs: 20 })(request(form())); assert.equal(r.status, 504); assert.ok(!(await r.text()).includes("private diagnostic"));
});
test("proxy cancellation reaches upstream; local HTTP is allowed only for loopback integration", async () => {
  let arrived = false; const controller = new AbortController();
  const fetcher: typeof fetch = async (_, init) => new Promise((_, reject) => { arrived = true; init!.signal!.addEventListener("abort", () => reject(new Error("cancelled")), { once: true }); controller.abort(); });
  const r = request(form()); const result = await createComputeRoute(fallback, "validate", { env, fetcher })(new Request(r, { signal: controller.signal })); assert.equal(result.status, 499); assert.equal(arrived, true);
  const bad = { ...env, AEP_BACKEND_URL: "http://127.0.0.1:8000" }; assert.equal((await createComputeRoute(fallback, "validate", { env: bad })(request(form()))).status, 503);
});
test("real proxy-to-backend validates, trains RFECV, infers single/batch and deletes with one signed session", async () => {
  const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../.."); const root = await mkdtemp(join(tmpdir(), "aep-proxy-test-"));
  const service = await createComputeService({ token: env.AEP_BACKEND_TOKEN, repoRoot: repo, pythonPath: process.env.AEP_PYTHON ?? join(repo, ".venv/bin/python"), temporaryRoot: root });
  try {
    service.server.listen(0, "127.0.0.1"); await once(service.server, "listening"); const addr = service.server.address(); if (!addr || typeof addr === "string") assert.fail("missing port");
    const configEnv = { ...env, AEP_PUBLIC_ORIGIN: "http://localhost:3000", AEP_BACKEND_URL: `http://127.0.0.1:${addr.port}` }; let savedCookie = "";
    const call = async (operation: "validate"|"train"|"predict"|"predict-batch", body: BodyInit, method = "POST") => { const r = await createComputeRoute(fallback, operation, { env: configEnv })(new Request("http://localhost:3000/api/custom-dataset/" + operation, { method, body, headers: { Origin: configEnv.AEP_PUBLIC_ORIGIN, Cookie: savedCookie, ...(typeof body === "string" ? { "Content-Type": "application/json" } : {}) } })); if (r.headers.has("set-cookie")) savedCookie = cookie(r); return r; };
    const text = "x,z,target\n" + Array.from({ length: 80 }, (_, i) => `${Math.sin(i)},${Math.cos(i)},${10+4*Math.sin(i)+.7*Math.cos(i)}`).join("\n");
    const config = validatePlan(profileCsv(text), { ...emptyDraft(), target: "target", features: ["x", "z"], cvSplits: 3 }).config, sha = createHash("sha256").update(text).digest("hex");
    const input = () => { const f = new FormData(); f.append("csv", new File([text], "fixture.csv")); f.append("config", JSON.stringify(config)); f.append("review", JSON.stringify({ schema_version: 1, artifact_type: "custom_dataset_review", dataset: { sha256: sha }, preprocessing: config, review: { confirmed: true, training_started: false, scope: "configuration_only" } })); return f; };
    const v = await call("validate", input()); assert.equal(v.status, 200); const validated = await v.json();
    const f = input(); f.append("experiment", JSON.stringify({ schema_version: 1, method: "linear_regression", confirmed: true, dataset_sha256: sha, config_sha256: validated.config_sha256, selection: { method: "rfecv", k: null } }));
    const t = await call("train", f); assert.equal(t.status, 200); const report = parseTrainingReport(await t.json()); if (report.status !== "completed" || !report.prediction_model) assert.fail("missing model");
    const model = report.prediction_model, values = Object.fromEntries(model.features.map(n => [n, .2]));
    assert.equal((await call("predict", JSON.stringify({ model_id: model.id, values }))).status, 200);
    assert.equal((await call("predict-batch", JSON.stringify({ model_id: model.id, features: model.features, rows: [model.features.map(() => .2)] }))).status, 200);
    assert.equal((await call("predict", JSON.stringify({ model_id: model.id }), "DELETE")).status, 204);
    assert.equal((await call("predict", JSON.stringify({ model_id: model.id, values }))).status, 410);
  } finally { await service.dispose(); await rm(root, { recursive: true, force: true }); }
});
