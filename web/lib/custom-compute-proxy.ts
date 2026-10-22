// Pure server transport; imported through custom-compute-route's server-only boundary.
import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { headers, readBounded, RequestError } from "./custom-validation-service.ts";
type Operation = "validate" | "compare" | "train" | "predict" | "predict-batch";
type Environment = Record<string, string | undefined>;
type Options = { env?: Environment; fetcher?: typeof fetch; now?: () => number; timeoutMs?: number };
const limit = 4 * 1024 * 1024, cookieName = "aep-compute-session", ttl = 1800;
const keys = ["AEP_BACKEND_URL", "AEP_BACKEND_TOKEN", "AEP_SESSION_SECRET", "AEP_PUBLIC_ORIGIN"] as const;
const loopback = (u: URL) => ["localhost", "127.0.0.1", "[::1]"].includes(u.hostname);
function settings(env: Environment) {
  const [endpoint, token, secret, publicOrigin] = keys.map(k => env[k]);
  if (!endpoint || !token || !secret || !publicOrigin || !/^[A-Za-z0-9_-]{32,256}$/.test(token) || !/^[A-Za-z0-9_-]{32,256}$/.test(secret) || token === secret) throw new Error("configuration");
  const backend = new URL(endpoint), origin = new URL(publicOrigin);
  for (const u of [backend, origin]) if (u.username || u.password || u.pathname !== "/" || u.search || u.hash || (u.protocol !== "https:" && !(u.protocol === "http:" && loopback(u) && loopback(origin)))) throw new Error("configuration");
  return { backend, origin, token, secret };
}
function session(cookie: string | null, secret: string, origin: string, now: number) {
  const sign = (id: string, expires: string) => createHmac("sha256", secret).update(`${origin}\n${id}.${expires}`).digest("base64url");
  const values = (cookie ?? "").split(";").map(v => v.trim()).filter(v => v.startsWith(cookieName + "="));
  const parts = values.length === 1 ? values[0].slice(cookieName.length + 1).split(".") : [];
  const [id, expires, signature] = parts;
  let valid = false;
  if (parts.length === 3 && /^[a-f0-9]{64}$/.test(id) && /^\d{10}$/.test(expires) && /^[A-Za-z0-9_-]{43}$/.test(signature) && Number(expires) > now && Number(expires) <= now + ttl) {
    valid = timingSafeEqual(Buffer.from(signature), Buffer.from(sign(id, expires)));
  }
  const selected = valid ? id : randomBytes(32).toString("hex"), expiry = String(now + ttl);
  return { valid, id: selected, value: `${selected}.${expiry}.${sign(selected, expiry)}` };
}
async function responseBody(response: Response, signal: AbortSignal) {
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader(); const chunks: Uint8Array[] = []; let size = 0;
  const abort = () => { void reader.cancel().catch(() => {}); };
  signal.addEventListener("abort", abort, { once: true });
  try {
    while (true) { if (signal.aborted) throw new Error("aborted"); const next = await reader.read(); if (next.done) break; size += next.value.length; if (size > limit) { await reader.cancel(); throw new RequestError(502, "運算結果超過連線層上限，請縮小比較範圍。"); } chunks.push(next.value); }
    if (signal.aborted) throw new Error("aborted");
    const body = new Uint8Array(size); let offset = 0; for (const chunk of chunks) { body.set(chunk, offset); offset += chunk.length; } return body;
  } finally { signal.removeEventListener("abort", abort); reader.releaseLock(); }
}
export function createComputeRoute(local: (request: Request) => Promise<Response>, operation: Operation, options: Options = {}) {
  return async (request: Request): Promise<Response> => {
    const env = options.env ?? process.env;
    if (!keys.some(k => env[k] !== undefined) && !env.VERCEL) return local(request);
    let config: ReturnType<typeof settings>;
    try { config = settings(env); } catch { return Response.json({ error: "遠端運算連線尚未設定完整，請聯絡網站管理者。" }, { status: 503, headers }); }
    const controller = new AbortController(); let expired = false;
    const abort = () => controller.abort(); request.signal.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(() => { expired = true; controller.abort(); }, options.timeoutMs ?? (operation === "compare" ? 180_000 : operation === "train" ? 140_000 : 45_000));
    try {
      const allowed = operation === "predict" ? ["POST", "DELETE"] : ["POST"];
      if (!allowed.includes(request.method)) return Response.json({ error: "不支援此操作方式。" }, { status: 405, headers: { ...headers, Allow: allowed.join(", ") } });
      if (new URL(request.url).search) throw new RequestError(400, "此運算入口不接受網址參數。");
      if (request.headers.get("origin") !== config.origin.origin || (request.headers.get("host") ?? new URL(request.url).host) !== config.origin.host || (request.headers.has("sec-fetch-site") && request.headers.get("sec-fetch-site") !== "same-origin")) throw new RequestError(403, "只接受本站的同來源操作。");
      const current = session(request.headers.get("cookie"), config.secret, config.origin.origin, Math.floor((options.now?.() ?? Date.now()) / 1000));
      if (!current.valid && (operation === "predict" || operation === "predict-batch")) throw new RequestError(410, "工作階段已到期或無法驗證，請重新訓練。");
      const length = request.headers.get("content-length");
      if (length !== null && (!/^\d+$/.test(length) || Number(length) > limit)) throw new RequestError(413, "遠端傳送內容超過 4 MiB，請縮小檔案或設定。");
      if (request.signal.aborted) controller.abort();
      const body = await readBounded(new Request(request, { signal: controller.signal }), limit);
      const contentType = request.headers.get("content-type") ?? "";
      if (["validate", "train", "compare"].includes(operation)) {
        if (!contentType.toLowerCase().startsWith("multipart/form-data;")) throw new RequestError(415, "請使用 CSV 與設定表單。");
        let form: FormData;
        try { form = await new Request(request.url, { method: "POST", headers: { "Content-Type": contentType }, body: body as BodyInit }).formData(); } catch { throw new RequestError(400, "無法讀取上傳表單。"); }
        const csv = form.get("csv"); if (csv instanceof File && csv.size > 3 * 1024 * 1024) throw new RequestError(413, "遠端 CSV 上限為 3 MiB，請縮小檔案。");
      } else if (contentType.split(";")[0].trim().toLowerCase() !== "application/json") throw new RequestError(415, "請使用網站的預測表單。");
      const upstream = await (options.fetcher ?? fetch)(new URL(`/api/custom-dataset/${operation}`, config.backend), { method: request.method, headers: { "Content-Type": contentType, Authorization: `Bearer ${config.token}`, "X-AEP-Session": current.id }, body: body as BodyInit, signal: controller.signal, redirect: "error", cache: "no-store" });
      if ([401, 403].includes(upstream.status)) throw new RequestError(503, "運算服務連線驗證失敗，請聯絡網站管理者。");
      if (![200, 204, 400, 410, 413, 415, 422, 429, 499, 504].includes(upstream.status)) throw new RequestError(502, "運算服務暫時無法完成，請稍後重試。");
      const output = await responseBody(upstream, controller.signal);
      if (upstream.status !== 204) { try { JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(output)); } catch { throw new RequestError(502, "運算服務回傳格式不符，請稍後重試。"); } }
      const out = new Headers(headers); out.set("Content-Type", "application/json;charset=utf-8");
      const retry = upstream.headers.get("retry-after"); if (retry && /^\d{1,4}$/.test(retry)) out.set("Retry-After", retry);
      out.set("Set-Cookie", `${cookieName}=${current.value}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${ttl}${config.origin.protocol === "https:" ? "; Secure" : ""}`);
      return new Response(upstream.status === 204 ? null : output, { status: upstream.status, headers: out });
    } catch (e) {
      const failure = expired ? new RequestError(504, "等待運算服務逾時，請縮小範圍後重試。") : request.signal.aborted ? new RequestError(499, "操作已取消。") : e instanceof RequestError ? e : new RequestError(503, "暫時無法連線到運算服務，請稍後重試。");
      return Response.json({ error: failure.message }, { status: failure.status, headers });
    } finally { clearTimeout(timer); request.signal.removeEventListener("abort", abort); }
  };
}
