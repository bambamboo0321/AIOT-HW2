import { parseComparisonReport } from "./custom-comparison-result.ts";
import { modelStore, type ModelStore } from "./custom-model-store.ts";
import { spawn } from "node:child_process";
import { constants } from "node:fs";
import { access, mkdtemp, rm, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { isAbsolute, resolve, join } from "node:path";
import { parseTrainingReport } from "./custom-training-result.ts";
import { parseValidationReport } from "./custom-validation-result.ts";

export const VALIDATION_BODY_LIMIT = 25 * 1024 * 1024 + 128 * 1024 + 1024 * 1024 + 64 * 1024;
export const headers = { "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff" };
export class RequestError extends Error { readonly status: number; constructor(status: number, message: string) { super(message); this.status = status; } }
export type Options = { retainModels?: boolean; store?: ModelStore; repoRoot?: string; pythonPath?: string; temporaryRoot?: string; timeoutMs?: number; bodyLimit?: number; operation?: "validate" | "train" | "compare" };
export async function readBounded(request: Request, limit: number): Promise<Uint8Array> {
  if (!request.body) throw new RequestError(400, "請選擇 CSV 並確認設定。");
  const reader = request.body.getReader(); const chunks: Uint8Array[] = []; let size = 0;
  const cancel = () => { void reader.cancel().catch(() => {}); };
  request.signal.addEventListener("abort", cancel, { once: true });
  try {
    if (request.signal.aborted) throw new RequestError(499, "驗證已取消。");
    while (true) {
      const { done, value } = await reader.read();
      if (request.signal.aborted) throw new RequestError(499, "驗證已取消。");
      if (done) break;
      size += value.byteLength;
      if (size > limit) { await reader.cancel(); throw new RequestError(413, "傳送內容過大，請縮小 CSV 或設定。"); }
      chunks.push(value);
    }
  } finally { request.signal.removeEventListener("abort", cancel); reader.releaseLock(); }
  const body = new Uint8Array(size); let offset = 0; for (const chunk of chunks) { body.set(chunk, offset); offset += chunk.byteLength; }
  return body;
}
export async function runPython(python: string, root: string, directory: string, signal: AbortSignal, timeoutMs: number, operation: "validate" | "train" | "compare" | "predict" | "batch_predict", modelPath?: string, retainModel = false): Promise<unknown> {
  if (signal.aborted) throw new RequestError(499, "驗證已取消。");
  return new Promise((done, reject) => {
    const args = ["-B", "-m", operation === "compare" ? "src.custom_dataset_comparison" : operation === "train" ? "src.custom_dataset_training" : "src.custom_dataset_validation", "--csv", join(directory, "dataset.csv"), "--config", join(directory, "config.json"), "--review", join(directory, "review.json")];
    if (operation === "train" || operation === "compare") args.push("--experiment", join(directory, "experiment.json"));
    if(operation==="train" && retainModel) args.push("--model-output",join(directory,"model.joblib"));
    if(operation==="predict" || operation==="batch_predict") args.splice(0,args.length,"-B","-m","src.custom_dataset_prediction","--model",modelPath!,"--input",join(directory,"input.json"));
    if(operation==="batch_predict")args.push("--batch");
    const child = spawn(python, args, {
      cwd: root, shell: false, stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1", PYTHONUNBUFFERED: "1", OMP_NUM_THREADS: "1", OPENBLAS_NUM_THREADS: "1", MKL_NUM_THREADS: "1" },
    });
    let failure: RequestError | null = null; const chunks: Buffer[] = []; let bytes = 0; let stderrBytes = 0;
    const stop = (error: RequestError) => { if (!failure) { failure = error; child.kill("SIGKILL"); } };
    const abort = () => stop(new RequestError(499, "驗證已取消。"));
    signal.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => stop(new RequestError(504, "Python 運算超過時間限制，請縮小資料或調整設定。")), timeoutMs);
    const clean = () => { clearTimeout(timer); signal.removeEventListener("abort", abort); };
    child.stdout.on("data", (chunk: Buffer) => { bytes += chunk.length; if (bytes > (operation==="batch_predict" || operation==="compare" || operation==="train" ? 8 : 1) * 1024 * 1024) stop(new RequestError(500, "驗證結果過大，無法顯示。")); else chunks.push(chunk); });
    child.stderr.on("data", (chunk: Buffer) => { stderrBytes += chunk.length; if (stderrBytes > 64 * 1024) stop(new RequestError(500, "Python 環境回傳異常，請檢查本機依賴。")); });
    child.on("error", () => { clean(); reject(new RequestError(503, "無法啟動本機 Python，請檢查虛擬環境。")); });
    child.on("close", code => {
      clean();
      if (failure) { reject(failure); return; }
      try {
        if (code !== 0 && code !== 2) throw new Error("exit");
        const value = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        if ((code === 0) !== (["valid", "completed", "compared", "predicted", "batch_predicted"].includes(value.status))) throw new Error("status");
        done(value);
      } catch { reject(new RequestError(500, "Python 未回傳有效的驗證結果，請檢查本機依賴。")); }
    });
    if (signal.aborted) abort();
  });
}
export function assertLocalRequest(request:Request) {
  const url = new URL(request.url);
  // Next.js can build request.url from its bind address (0.0.0.0). The
  // incoming Host is the browser's authority; never use forwarded headers.
  const host = request.headers.get("host") ?? url.host;
  let sameOrigin = false;
  if (/^(localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?$/i.test(host) && ["http:", "https:"].includes(url.protocol)) {
    try { sameOrigin = request.headers.get("origin") === new URL(`${url.protocol}//${host}`).origin; } catch { /* Reject malformed local authority. */ }
  }
  if (!sameOrigin || (request.headers.has("sec-fetch-site") && request.headers.get("sec-fetch-site") !== "same-origin")) throw new RequestError(403, "此驗證入口只接受本機網站的同來源操作。");
}
export function createValidationHandler(options: Options = {}) {
  const operation = options.operation ?? "validate";
  let occupied = false;
  return async function POST(request: Request): Promise<Response> {
    let ownsSlot = false; let directory: string | undefined;
    try {
      if (request.method !== "POST") return Response.json({ error: "只接受 POST 驗證。" }, { status: 405, headers: { ...headers, Allow: "POST" } });
      assertLocalRequest(request);
      const contentType = request.headers.get("content-type") ?? "";
      if (!contentType.toLowerCase().startsWith("multipart/form-data;")) throw new RequestError(415, "請以網站的 CSV 與設定表單送出。");
      const limit = options.bodyLimit ?? VALIDATION_BODY_LIMIT;
      const length = request.headers.get("content-length");
      if (length !== null && (!/^\d+$/.test(length) || Number(length) > limit)) throw new RequestError(413, "傳送內容過大或長度不合法。");
      if (occupied) return Response.json({ error: "本機正在處理另一份資料，請稍後再試。" }, { status: 429, headers: { ...headers, "Retry-After": "2" } });
      occupied = true; ownsSlot = true;
      const body = await readBounded(request, limit);
      let form: FormData;
      try { form = await new Request(request.url, { method: "POST", headers: { "Content-Type": contentType }, body: body as BodyInit }).formData(); }
      catch { throw new RequestError(400, "無法讀取上傳表單，請重新選擇 CSV。"); }
      const keys = [...form.keys()];
      const expectedKeys = operation !== "validate" ? ["csv", "config", "review", "experiment"] : ["csv", "config", "review"];
      if (keys.length !== expectedKeys.length || new Set(keys).size !== expectedKeys.length || keys.some(k => !expectedKeys.includes(k))) throw new RequestError(400, "表單必須包含一份 CSV、設定及確認摘要。");
      const file = form.get("csv"); const config = form.get("config"); const review = form.get("review");
      if (!(file instanceof File) || !file.name.toLowerCase().endsWith(".csv") || !file.size || file.size > 25 * 1024 * 1024) throw new RequestError(400, "CSV 必須非空、為 .csv 且不超過 25 MiB。");
      if (typeof config !== "string" || !config || Buffer.byteLength(config) > 128 * 1024 || typeof review !== "string" || !review || Buffer.byteLength(review) > 1024 * 1024) throw new RequestError(400, "設定或確認摘要缺少、過大或格式不合法。");
      const experiment = form.get("experiment");
      if (operation !== "validate" && (typeof experiment !== "string" || !experiment || Buffer.byteLength(experiment) > 4096)) throw new RequestError(400, "缺少或超過上限的訓練確認。");
      const root = options.repoRoot ?? resolve(process.cwd(), "..");
      const python = options.pythonPath ?? process.env.AEP_PYTHON ?? join(root, ".venv", "bin", "python");
      if (!isAbsolute(python)) throw new RequestError(503, "AEP_PYTHON 必須指向本機 Python 的絕對路徑。");
      try { await access(python, constants.X_OK); } catch { throw new RequestError(503, "找不到本機 Python 環境，請先建立專案 .venv 或設定 AEP_PYTHON。"); }
      directory = await mkdtemp(join(options.temporaryRoot ?? tmpdir(), operation === "compare" ? "aep-compare-" : operation === "train" ? "aep-train-" : "aep-validate-"));
      const raw = Buffer.from(await file.arrayBuffer());
      await writeFile(join(directory, "dataset.csv"), raw, { mode: 0o600 });
      await writeFile(join(directory, "config.json"), config, { mode: 0o600 });
      await writeFile(join(directory, "review.json"), review, { mode: 0o600 });
      if (operation !== "validate") await writeFile(join(directory, "experiment.json"), experiment as string, { mode: 0o600 });
      const output = await runPython(python, root, directory, request.signal, options.timeoutMs ?? (operation !== "validate" ? 120_000 : 30_000), operation, undefined, options.retainModels === true);
      const report = operation === "compare" ? parseComparisonReport(output) : operation === "train" ? parseTrainingReport(output) : parseValidationReport(output);
      const validated = report.status === "completed" || report.status === "compared" ? report.validation : report.status === "valid" ? report : null;
      if (validated && (!validated.review_checked || validated.dataset.sha256 !== createHash("sha256").update(raw).digest("hex") || validated.dataset.bytes !== raw.length)) throw new RequestError(500, "驗證結果與傳送的資料版本不一致。");
      if(report.status==="completed" && options.retainModels) {
        if(request.signal.aborted)throw new RequestError(499,"訓練已取消。");
        const missing=JSON.parse(config).missing_numeric;
        if(!["mean","median","none"].includes(missing))throw new RequestError(500,"模型補值設定不符。");
        report.prediction_model=await (options.store??modelStore()).register(join(directory,"model.joblib"),{target:report.target,features:report.selection.final_features,missing_numeric:missing,dataset_sha256:report.validation.dataset.sha256,config_sha256:report.validation.config_sha256});
        if(request.signal.aborted){await (options.store??modelStore()).delete(report.prediction_model.id);throw new RequestError(499,"訓練已取消。");}
      }
      return Response.json(report, { status: report.status === "valid" || report.status === "completed" || report.status === "compared" ? 200 : 422, headers });
    } catch (error) {
      const failure = error instanceof RequestError ? error : new RequestError(500, "本機驗證服務暫時無法完成，請稍後重試。");
      return Response.json({ error: failure.message }, { status: failure.status, headers });
    } finally {
      try { if (directory) await rm(directory, { recursive: true, force: true }); }
      finally { if (ownsSlot) occupied = false; }
    }
  };
}

export function createTrainingHandler(options: Omit<Options, "operation"> = {}) { return createValidationHandler({ ...options, operation: "train" }); }

export function createComparisonHandler(options: Omit<Options, "operation" | "retainModels"> = {}) { return createValidationHandler({ ...options, operation: "compare", retainModels: false }); }
