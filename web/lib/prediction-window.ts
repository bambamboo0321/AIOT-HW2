import { WINDOW_SIZES, type WindowSize, type PredictionRow, type PredictionWindow, type PredictionMeta, type WindowScores } from "./prediction-types.ts";

export function windowBounds(start: number, size: number, total: number): { start: number; size: WindowSize } {
  if (!Number.isInteger(start) || start < 0 || start >= total || !Number.isInteger(total) || total < 1 || !WINDOW_SIZES.includes(size as WindowSize)) throw new Error("視窗範圍不合法");
  return { start: Math.min(start, Math.max(0, total - size)), size: size as WindowSize };
}
export function parseWindowQuery(params: URLSearchParams, total: number): { start: number; size: WindowSize } {
  if ([...params.keys()].some(k => k !== "start" && k !== "size") || params.getAll("start").length > 1 || params.getAll("size").length > 1) throw new Error("查詢參數不合法");
  const start = params.get("start") ?? "0"; const size = params.get("size") ?? "144";
  if (!/^(0|[1-9]\d*)$/.test(start) || !/^[1-9]\d*$/.test(size)) throw new Error("視窗位置／筆數必須是整數");
  return windowBounds(Number(start), Number(size), total);
}
export function windowScores(rows: PredictionRow[], id: "baseline" | "selected"): WindowScores {
  if (!rows.length) throw new Error("視窗沒有資料");
  const n = rows.length; const averageActual = rows.reduce((s, r) => s + r.actual, 0) / n;
  let squareError = 0; let absoluteError = 0; let totalVariation = 0; let covered = 0; let below = 0; let above = 0; let width = 0;
  for (const row of rows) {
    const model = row[id]; const error = row.actual - model.prediction;
    squareError += error ** 2; absoluteError += Math.abs(error); totalVariation += (row.actual - averageActual) ** 2; width += model.upper - model.lower;
    if (row.actual < model.lower) below++; else if (row.actual > model.upper) above++; else covered++;
  }
  return { rmse: Math.sqrt(squareError / n), mae: absoluteError / n, r2: n < 2 || totalVariation === 0 ? null : 1 - squareError / totalVariation,
    coverage: covered / n, averageWidth: width / n, covered, below, above };
}
export function predictionWindow(rows: PredictionRow[], meta: PredictionMeta, requestedStart: number, size: number): PredictionWindow {
  const bounds = windowBounds(requestedStart, size, rows.length); const selected = rows.slice(bounds.start, bounds.start + bounds.size);
  return { ...bounds, totalRows: meta.totalRows, sourceDigest: meta.sourceDigest, rows: selected, stats: { baseline: windowScores(selected, "baseline"), selected: windowScores(selected, "selected") } };
}
export function timePosition(value: string, meta: Pick<PredictionMeta, "start" | "samplingSeconds" | "totalRows">): number {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(value)) throw new Error("請輸入完整日期與時間");
  const date = Date.parse(value + "Z"); const start = Date.parse(meta.start + "Z");
  const position = (date - start) / (meta.samplingSeconds * 1000);
  if (!Number.isInteger(position) || position < 0 || position >= meta.totalRows) throw new Error("請選擇 Holdout 內的十分鐘觀測時間");
  return position;
}
function object(v: unknown): Record<string, unknown> { if (!v || typeof v !== "object" || Array.isArray(v)) throw new Error("視窗格式錯誤"); return v as Record<string, unknown>; }
export function validatePredictionRows(value: unknown, start: number, expectedCount: number): PredictionRow[] {
  if (!Array.isArray(value) || value.length !== expectedCount) throw new Error("視窗預測筆數不符");
  value.forEach((v, i) => {
    const row = object(v);
    if (row.position !== start + i + 1 || typeof row.timestamp !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(row.timestamp) || !Number.isFinite(Date.parse(row.timestamp + "Z")) || typeof row.actual !== "number" || !Number.isFinite(row.actual) || row.actual < 0) throw new Error("逐筆預測列格式錯誤");
    if (i > 0 && Date.parse(row.timestamp + "Z") - Date.parse(object(value[i - 1]).timestamp + "Z") !== 600000) throw new Error("逐筆時間順序錯誤");
    for (const id of ["baseline", "selected"]) {
      const m = object(row[id]);
      for (const key of ["prediction", "lower", "upper"]) if (typeof m[key] !== "number" || !Number.isFinite(m[key])) throw new Error("逐筆預測不是有限值");
      if (Number(m.lower) > Number(m.prediction) || Number(m.prediction) > Number(m.upper) || Math.abs((Number(m.lower) + Number(m.upper)) / 2 - Number(m.prediction)) > 1e-7) throw new Error("區間順序／中心不一致");
    }
  });
  return value as PredictionRow[];
}
export function parsePredictionWindow(value: unknown): PredictionWindow {
  const v = object(value); const total = Number(v.totalRows); const bounds = windowBounds(Number(v.start), Number(v.size), total);
  if (bounds.start !== v.start || typeof v.sourceDigest !== "string" || !/^[a-f0-9]{64}$/.test(v.sourceDigest)) throw new Error("視窗版本／範圍不符");
  const rows = validatePredictionRows(v.rows, bounds.start, Math.min(bounds.size, total - bounds.start)); const stats = object(v.stats);
  for (const id of ["baseline", "selected"] as const) {
    const saved = object(stats[id]); const computed = windowScores(rows, id);
    for (const key of Object.keys(computed) as (keyof WindowScores)[]) {
      if (computed[key] === null ? saved[key] !== null : typeof saved[key] !== "number" || Math.abs(Number(saved[key]) - Number(computed[key])) > 1e-9 * Math.max(1, Math.abs(Number(computed[key])))) throw new Error("視窗統計與逐筆資料不一致");
    }
  }
  return value as PredictionWindow;
}
