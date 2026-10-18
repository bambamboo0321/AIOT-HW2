import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readdir, rm } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { createValidationHandler } from "../lib/custom-validation-service.ts";
import { parseValidationReport } from "../lib/custom-validation-result.ts";
import { emptyDraft, profileCsv, validatePlan } from "../lib/custom-dataset.ts";
const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(web, "..");
const python = process.env.AEP_PYTHON ?? resolve(repo, ".venv/bin/python");
const text = "date,x,target\n" + Array.from({ length: 30 }, (_, i) => `2024-01-${String(i+1).padStart(2,"0")},${i*2},${i%7}`).join("\n");
const config = validatePlan(profileCsv(text), { ...emptyDraft(), target: "target", features: ["x"], split: "chronological", splitColumn: "date", gap: 1 }).config;
const sha = createHash("sha256").update(text).digest("hex");
function form() { const f = new FormData(); f.append("csv", new File([text], "../../client-name.csv")); f.append("config", JSON.stringify(config)); f.append("review", JSON.stringify({ schema_version: 1, artifact_type: "custom_dataset_review", dataset: { sha256: sha }, preprocessing: config, review: { confirmed: true, training_started: false, scope: "configuration_only" } })); return f; }
function request(body: BodyInit = form(), headers: Record<string,string> = {}, signal?: AbortSignal) { return new Request("http://127.0.0.1:3000/api/custom-dataset/validate", { method: "POST", body, headers: { Origin: "http://127.0.0.1:3000", ...headers }, signal }); }
async function fixture(run: (directory: string) => Promise<void>) { const directory = await mkdtemp(resolve(web, ".test-tmp-api-")); try { await mkdir(resolve(directory,"uploads")); await run(directory); } finally { await rm(directory,{recursive:true,force:true}); } }
const handler = (dir: string, extra = {}) => createValidationHandler({ repoRoot: repo, pythonPath: python, temporaryRoot: resolve(dir,"uploads"), ...extra });

test("real Python request validates matching CSV/config/review and cleans all temporary files", async () => fixture(async dir => {
  const r = await handler(dir)(request()); assert.equal(r.status,200); assert.equal(r.headers.get("cache-control"),"private, no-store");
  const report = parseValidationReport(await r.json()); assert.equal(report.status,"valid"); if(report.status!=="valid")return;
  assert.equal(report.dataset.sha256,sha); assert.equal(report.review_checked,true); assert.equal(report.training_started,false);
  assert.equal(report.split.training_rows,24); assert.equal(report.split.holdout_rows,6); assert.equal(report.split.cv_folds.length,5);
  assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("invalid data/config and stale digest produce domain errors with HTTP 422 and cleanup", async () => fixture(async dir => {
  const h = handler(dir);
  for(const field of ["config","review"]) {
    const f=form();if(field==="config")f.set("config","{");else{const value=JSON.parse(String(f.get("review")));value.dataset.sha256="0".repeat(64);f.set("review",JSON.stringify(value));}
    const response=await h(request(f));assert.equal(response.status,422);const value=parseValidationReport(await response.json());assert.equal(value.status,"invalid");assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
  }
}));
test("cross-origin, missing origin and non-loopback requests are rejected before Python", async () => fixture(async dir => {
  const h=handler(dir);
  assert.equal((await h(request(form(),{Origin:"https://evil.example"}))).status,403);
  assert.equal((await h(new Request("http://127.0.0.1:3000/api/custom-dataset/validate",{method:"POST",body:form()}))).status,403);
  assert.equal((await h(new Request("http://evil.example/api/custom-dataset/validate",{method:"POST",body:form(),headers:{Origin:"http://evil.example"}}))).status,403);
  assert.equal((await h(request(form(),{"Sec-Fetch-Site":"cross-site"}))).status,403);assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("duplicate/unknown fields, missing review, non-CSV files and non-multipart bodies are rejected", async () => fixture(async dir => {
  const h=handler(dir);
  for(const alter of [(f:FormData)=>f.append("csv",new File([text],"extra.csv")),(f:FormData)=>f.set("path","/etc/passwd"),(f:FormData)=>f.delete("review"),(f:FormData)=>f.set("csv",new File([text],"data.joblib"))]){const f=form();alter(f);assert.equal((await h(request(f))).status,400);}
  assert.equal((await h(request("{}",{"Content-Type":"application/json"}))).status,415);assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("request size is enforced for both declared and actual streaming bodies", async () => fixture(async dir => {
  const h=handler(dir,{bodyLimit:100});assert.equal((await h(request(form(),{"Content-Length":"101"}))).status,413);
  assert.equal((await h(request(form()))).status,413);assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("missing Python is a safe 503 without local paths in its response", async () => fixture(async dir => {
  const r=await handler(dir,{pythonPath:resolve(dir,"secret-python-path")})(request());assert.equal(r.status,503);assert.ok(!JSON.stringify(await r.json()).includes(dir));assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("timeouts terminate Python and clean uploads before returning", async () => fixture(async dir => {
  const r=await handler(dir,{timeoutMs:1})(request());assert.equal(r.status,504);assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("aborted requests cancel pending validation and release the single execution slot", async () => fixture(async dir => {
  const h=handler(dir);const controller=new AbortController();const pending=h(request(form(),{},controller.signal));const timer=setTimeout(()=>controller.abort(),30);
  const r=await pending;clearTimeout(timer);assert.equal(r.status,499);assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
  assert.equal((await h(request())).status,200);
}));
test("simultaneous validations get 429 and the slot is available after completion", async () => fixture(async dir => {
  const h=handler(dir);const first=h(request());assert.equal((await h(request())).status,429);assert.equal((await first).status,200);assert.equal((await h(request())).status,200);assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("malformed interpreter output is a safe error and still removes temporary data", async () => fixture(async dir => {
  const r=await handler(dir,{pythonPath:process.execPath})(request());assert.equal(r.status,500);assert.deepEqual(await readdir(resolve(dir,"uploads")),[]);
}));
test("client result parsing rejects training, wrong scope, malformed folds and inconsistent counts", async () => fixture(async dir => {
  const r=await handler(dir)(request());const report=await r.json();
  for(const modify of [(v:typeof report)=>{v.training_started=true;},(v:typeof report)=>{v.scope="training";},(v:typeof report)=>{v.split.holdout_rows++;},(v:typeof report)=>{v.split.cv_folds[0].fold=2;},(v:typeof report)=>{v.dataset.sha256="wrong";}]){const value=structuredClone(report);modify(value);assert.throws(()=>parseValidationReport(value));}
  const normalized=parseValidationReport({...report,raw_rows:["never expose"]});assert.ok(!("raw_rows" in normalized));
}));

test("Next bind address uses local Host while rejecting alias, remote and forwarded authorities", async () => fixture(async dir => {
  const h=handler(dir);
  const make=(headers:Record<string,string>)=>new Request("http://0.0.0.0:3000/api/custom-dataset/validate",{method:"POST",body:form(),headers:{Host:"127.0.0.1:3000",Origin:"http://127.0.0.1:3000","Sec-Fetch-Site":"same-origin",...headers}});
  assert.equal((await h(make({}))).status,200);
  const rejectedAuthorities:Record<string,string>[] = [{Origin:"http://localhost:3000"},{Host:"evil.example:3000"},{Host:"0.0.0.0:3000"},{Host:"evil.example:3000","X-Forwarded-Host":"127.0.0.1:3000"},{Host:"127.0.0.1:3000@evil.example"}];
  for(const headers of rejectedAuthorities) assert.equal((await h(make(headers))).status,403);
}));
