/** Private HTTP gateway for the existing, bounded Python workers.
 * No browser CORS: only the trusted frontend server knows the service token.
 */
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createHash, timingSafeEqual } from "node:crypto";
import { access, chmod, mkdtemp, rm } from "node:fs/promises";
import { constants } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createValidationHandler, createTrainingHandler, createComparisonHandler, headers, RequestError } from "../web/lib/custom-validation-service.ts";
import { createPredictionHandler } from "../web/lib/custom-prediction-service.ts";
import { ModelStore } from "../web/lib/custom-model-store.ts";

export const COMPUTE_LIMITS = { bodyBytes: 4 * 1024 * 1024, csvBytes: 3 * 1024 * 1024, responseBytes: 4 * 1024 * 1024, sessions: 8, modelsPerSession: 3, modelTtlMs: 30 * 60_000 } as const;
const prefix = "/api/custom-dataset/";
const methods = new Map([
  ["validate", ["POST"]], ["compare", ["POST"]], ["train", ["POST"]],
  ["predict", ["POST", "DELETE"]], ["predict-batch", ["POST"]],
]);
type Handler = (request: Request) => Promise<Response>;
type Session = { store: ModelStore; train: Handler; predict: Handler; batch: Handler; lastUsed: number };
export type ComputeOptions = {
  token: string; repoRoot: string; pythonPath: string; temporaryRoot?: string;
  modelTtlMs?: number; sessionLimit?: number; workerTimeoutMs?: number; now?: () => number;
};
const digest = (text: string) => createHash("sha256").update(text).digest();

function incomingBody(req: IncomingMessage, signal: AbortSignal): Promise<Buffer> {
  return new Promise((resolveBody, reject) => {
    const chunks: Buffer[] = []; let size = 0;
    const cleanup = () => { req.off("data", data); req.off("end", end); req.off("error", error); signal.removeEventListener("abort", abort); clearTimeout(timer); };
    const fail = (reason: RequestError) => { cleanup(); req.resume(); reject(reason); };
    const data = (chunk: Buffer) => { size += chunk.length; if (size > COMPUTE_LIMITS.bodyBytes) fail(new RequestError(413, "雲端傳送內容超過 4 MiB，請縮小檔案或設定。")); else chunks.push(chunk); };
    const end = () => { cleanup(); resolveBody(Buffer.concat(chunks)); };
    const error = () => fail(new RequestError(400, "無法讀取完整請求。"));
    const abort = () => fail(new RequestError(499, "操作已取消。"));
    const timer = setTimeout(() => fail(new RequestError(408, "上傳等待逾時，請重新送出。")), 20_000);
    req.on("data", data); req.once("end", end); req.once("error", error);
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
  });
}

export async function createComputeService(options: ComputeOptions) {
  if (!/^[A-Za-z0-9_-]{32,256}$/.test(options.token)) throw new Error("AEP_SERVICE_TOKEN must contain 32–256 base64url characters.");
  if (!isAbsolute(options.repoRoot) || !isAbsolute(options.pythonPath)) throw new Error("Repository and Python paths must be absolute.");
  const ttl = options.modelTtlMs ?? COMPUTE_LIMITS.modelTtlMs;
  const sessionLimit = options.sessionLimit ?? COMPUTE_LIMITS.sessions;
  if (!Number.isSafeInteger(ttl) || ttl < 1 || ttl > COMPUTE_LIMITS.modelTtlMs || !Number.isSafeInteger(sessionLimit) || sessionLimit < 1 || sessionLimit > COMPUTE_LIMITS.sessions) throw new Error("Invalid backend capacity.");
  await access(options.pythonPath, constants.X_OK);
  for (const name of ["custom_dataset_validation", "custom_dataset_training", "custom_dataset_comparison", "custom_dataset_prediction"]) await access(join(options.repoRoot, "src", name + ".py"), constants.R_OK);
  const directory = await mkdtemp(join(options.temporaryRoot ?? tmpdir(), "aep-compute-"));
  await chmod(directory, 0o700);
  const workerOptions = { repoRoot: options.repoRoot, pythonPath: options.pythonPath, temporaryRoot: directory, bodyLimit: COMPUTE_LIMITS.bodyBytes, timeoutMs: options.workerTimeoutMs };
  const validate = createValidationHandler(workerOptions), compare = createComparisonHandler(workerOptions);
  const sessions = new Map<string, Session>();
  const active = new Set<AbortController>(), pending = new Set<Promise<void>>();
  const now = options.now ?? Date.now, expectedAuth = digest("Bearer " + options.token);
  let occupied = false, closing = false;
  async function prune() {
    for (const [id, entry] of sessions) if (now() - entry.lastUsed >= ttl) { sessions.delete(id); await entry.store.dispose(); }
  }
  function newSession(id: string): Session {
    if (sessions.size >= sessionLimit) throw new RequestError(429, "服務的暫存工作已滿，請稍後重試。");
    const store = new ModelStore({ root: directory, ttlMs: ttl, capacity: COMPUTE_LIMITS.modelsPerSession, now });
    const settings = { ...workerOptions, store };
    const entry = { store, train: createTrainingHandler({ ...settings, retainModels: true }), predict: createPredictionHandler(settings), batch: createPredictionHandler({ ...settings, batch: true }), lastUsed: now() };
    sessions.set(id, entry); return entry;
  }
  const sweep = setInterval(() => { if (!occupied && !closing) { occupied = true; void prune().catch(() => {}).finally(() => { occupied = false; }); } }, 60_000); sweep.unref();

  async function dispatch(req: IncomingMessage, signal: AbortSignal): Promise<Response> {
    if (req.url === "/healthz" && req.method === "GET") return Response.json({ status: closing ? "stopping" : "ok" }, { status: closing ? 503 : 200, headers });
    if (!timingSafeEqual(digest(req.headers.authorization ?? ""), expectedAuth)) throw new RequestError(401, "服務驗證失敗。");
    if (closing) throw new RequestError(503, "服務正在停止，請稍後重試。");
    const operation = req.url?.startsWith(prefix) ? req.url.slice(prefix.length) : "";
    const allowed = methods.get(operation);
    if (!allowed) throw new RequestError(404, "找不到指定的運算入口。");
    if (!allowed.includes(req.method ?? "")) return Response.json({ error: "不支援此操作方式。" }, { status: 405, headers: { ...headers, Allow: allowed.join(", ") } });
    const sessionId = req.headers["x-aep-session"];
    if (typeof sessionId !== "string" || !/^[a-f0-9]{64}$/.test(sessionId)) throw new RequestError(400, "缺少有效的工作階段。");
    const length = req.headers["content-length"];
    if (length !== undefined && (!/^\d+$/.test(length) || Number(length) > COMPUTE_LIMITS.bodyBytes)) throw new RequestError(413, "雲端傳送內容超過 4 MiB，請縮小檔案或設定。");
    if (occupied) return Response.json({ error: "運算服務忙碌中，請稍後重試。" }, { status: 429, headers: { ...headers, "Retry-After": "2" } });
    occupied = true;
    let retained: { store: ModelStore; id: string } | undefined;
    let created: Session | undefined, keptModel = false;
    try {
      await prune();
      const body = await incomingBody(req, signal);
      // Preserve the existing local-only guard: requests become internal only
      // after service authentication. Browser-controlled forwarding headers are discarded.
      const request = new Request("http://localhost" + req.url, { method: req.method, headers: { Origin: "http://localhost", "Content-Type": req.headers["content-type"] ?? "" }, body: new Uint8Array(body), signal });
      if (["validate", "compare", "train"].includes(operation)) {
        if (!request.headers.get("content-type")?.toLowerCase().startsWith("multipart/form-data;")) throw new RequestError(415, "請使用 CSV 與設定表單。");
        let form: FormData;
        try { form = await request.clone().formData(); } catch { throw new RequestError(400, "無法讀取上傳表單。"); }
        const csv = form.get("csv");
        if (csv instanceof File && csv.size > COMPUTE_LIMITS.csvBytes) throw new RequestError(413, "雲端 CSV 上限為 3 MiB，請縮小檔案。");
      }
      let response: Response;
      if (operation === "validate") response = await validate(request);
      else if (operation === "compare") response = await compare(request);
      else {
        let entry = sessions.get(sessionId);
        if (!entry && operation === "train") { entry = newSession(sessionId); created = entry; }
        if (!entry) {
          if (req.method === "DELETE") return new Response(null, { status: 204, headers });
          throw new RequestError(410, "模型已到期、清除或服務已重啟；請重新訓練。");
        }
        entry.lastUsed = now();
        response = await (operation === "train" ? entry.train : operation === "predict-batch" ? entry.batch : entry.predict)(request);
        if (operation === "train" && response.ok) {
          const value = await response.clone().json();
          if (typeof value.prediction_model?.id === "string") retained = { store: entry.store, id: value.prediction_model.id };
        }
      }
      const output = await response.arrayBuffer();
      if (signal.aborted) throw new RequestError(499, "操作已取消。");
      if (output.byteLength > COMPUTE_LIMITS.responseBytes) throw new RequestError(413, "結果超過雲端回傳上限，請縮小比較範圍或特徵數。");
      keptModel = retained !== undefined; retained = undefined;
      return new Response(response.status === 204 ? null : output, { status: response.status, headers: response.headers });
    } finally {
      try {
        if (retained) await retained.store.delete(retained.id);
        if (created && !keptModel) { sessions.delete(sessionId); await created.store.dispose(); }
      } finally { occupied = false; }
    }
  }

  async function handle(req: IncomingMessage, res: ServerResponse) {
    const controller = new AbortController(); active.add(controller);
    const abort = () => controller.abort();
    const disconnected = () => { if (!res.writableEnded) abort(); };
    req.once("aborted", abort); res.once("close", disconnected);
    try {
      let response: Response;
      try { response = await dispatch(req, controller.signal); }
      catch (e) { const failure = e instanceof RequestError ? e : new RequestError(500, "運算服務暫時無法完成。"); response = Response.json({ error: failure.message }, { status: failure.status, headers }); }
      if (res.destroyed) return;
      const body = Buffer.from(await response.arrayBuffer());
      res.statusCode = response.status;
      for (const name of ["content-type", "cache-control", "x-content-type-options", "retry-after", "allow"]) { const value = response.headers.get(name); if (value) res.setHeader(name, value); }
      if (response.status !== 204) res.setHeader("Content-Length", body.length);
      res.end(body);
    } finally { req.off("aborted", abort); res.off("close", disconnected); active.delete(controller); }
  }
  const server = createServer({ maxHeaderSize: 16 * 1024 }, (req, res) => {
    const work = handle(req, res).catch(() => { if (!res.destroyed) res.destroy(); });
    pending.add(work); void work.finally(() => pending.delete(work));
  });
  server.requestTimeout = 30_000; server.headersTimeout = 10_000; server.keepAliveTimeout = 5_000;
  async function dispose() {
    if (closing) return;
    closing = true; clearInterval(sweep);
    const stopped = server.listening ? new Promise<void>((done) => server.close(() => done())) : Promise.resolve();
    for (const controller of active) controller.abort();
    server.closeAllConnections();
    await Promise.all([...pending]); await stopped;
    await Promise.all([...sessions.values()].map(s => s.store.dispose()));
    sessions.clear(); await rm(directory, { recursive: true, force: true });
  }
  return { server, dispose };
}

async function main() {
  const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const port = Number(process.env.PORT ?? "8000"), host = process.env.AEP_BIND_HOST ?? "127.0.0.1";
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("PORT must be 1–65535.");
  const service = await createComputeService({ token: process.env.AEP_SERVICE_TOKEN ?? "", repoRoot, pythonPath: process.env.AEP_PYTHON ?? join(repoRoot, ".venv/bin/python") });
  const stop = () => { void service.dispose().catch(() => { process.exitCode = 1; }); };
  process.once("SIGTERM", stop); process.once("SIGINT", stop);
  service.server.once("error", () => { console.error("Compute service could not listen."); stop(); process.exitCode = 1; });
  service.server.listen(port, host, () => console.log(`Compute service listening on ${host}:${port}`));
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) void main().catch(() => { console.error("Compute service startup failed; check token, paths and configuration."); process.exitCode = 1; });
