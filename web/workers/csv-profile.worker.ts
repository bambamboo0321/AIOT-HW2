import { CSV_LIMITS, decodeCsv, profileCsv } from "../lib/custom-dataset";
self.onmessage = async (event: MessageEvent<{ file: File }>) => {
  try {
    const { file } = event.data;
    if (!file.name.toLowerCase().endsWith(".csv")) throw new Error("請選擇 .csv 檔案。");
    if (!file.size || file.size > CSV_LIMITS.bytes) throw new Error("CSV 必須為非空檔案，且不超過 25 MiB。");
    const buffer = await file.arrayBuffer(); const profile = profileCsv(decodeCsv(buffer));
    const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)), b => b.toString(16).padStart(2, "0")).join("");
    self.postMessage({ ok: true, data: { ...profile, fileName: file.name, bytes: file.size, sha256 } });
  } catch (e) { self.postMessage({ ok: false, error: e instanceof Error ? e.message : "CSV 無法解析。" }); }
};
