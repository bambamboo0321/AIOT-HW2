import { readFile, realpath } from "node:fs/promises";
import { createHash } from "node:crypto";
import { isAbsolute, relative, resolve } from "node:path";
import { readSource } from "./research-data.ts";
import type { EvaluationData, EvaluationModel, EvaluationScores, EvaluationSnapshot } from "./evaluation-types.ts";

type Obj = Record<string, unknown>;
export const EVALUATION_SOURCES = {
  dashboardModels: "dashboard_data/model_comparison.json",
  dashboardIntervals: "dashboard_data/prediction_intervals.json",
  metrics: "reports/phase3/phase3c/prediction_metrics.json",
  intervals: "reports/phase3/phase3c/prediction_interval_metrics.json",
  plan: "reports/phase3/phase3c/evaluation_plan.json",
  figures: "reports/phase3/phase3c/figure_manifest.json",
  predictions: "reports/phase3/phase3c/prediction_interval_predictions.csv",
} as const;
const metricKeys = ["rmse", "mae", "mse", "r2"] as const;
function obj(v: unknown): Obj { if (!v || typeof v !== "object" || Array.isArray(v)) throw new Error("缺少 object"); return v as Obj; }
function arr(v: unknown): unknown[] { if (!Array.isArray(v)) throw new Error("缺少陣列"); return v; }
function str(v: unknown): string { if (typeof v !== "string" || !v.trim()) throw new Error("缺少文字"); return v; }
function num(v: unknown): number { if (typeof v !== "number" || !Number.isFinite(v)) throw new Error("缺少有限數值"); return v; }
function integer(v: unknown, positive = false): number { const n = num(v); if (!Number.isInteger(n) || n < (positive ? 1 : 0)) throw new Error("不合法的計數"); return n; }
function fraction(v: unknown): number { const n = num(v); if (n < 0 || n > 1) throw new Error("比例超出 0–1"); return n; }
function nonnegative(v: unknown): number { const n = num(v); if (n < 0) throw new Error("負值誤差／寬度"); return n; }
function equal(a: unknown, b: unknown, at: string) { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${at} 不一致`); }
function close(a: number, b: number, at: string) { if (Math.abs(a - b) > 1e-8 * Math.max(1, Math.abs(b))) throw new Error(`${at} 與逐筆預測不一致`); }
function scores(value: unknown): EvaluationScores {
  const v = obj(value); const result = { rmse: nonnegative(v.rmse), mae: nonnegative(v.mae), mse: nonnegative(v.mse), r2: num(v.r2) };
  if (result.r2 > 1) throw new Error("R² 大於 1"); close(result.rmse ** 2, result.mse, "RMSE²／MSE"); return result;
}
function timestamp(v: unknown): string {
  const s = str(v); if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(s) || !Number.isFinite(Date.parse(s + "Z"))) throw new Error("來源時間格式錯誤"); return s;
}
function period(v: unknown, rows: number) { const p = obj(v); if (p.rows !== rows) throw new Error("時段樣本數不一致"); const start = timestamp(p.start); const end = timestamp(p.end); if (start >= end) throw new Error("時段順序錯誤"); return { start, end }; }
export type SavedPrediction = { index: number; timestamp: string; actual: number;
  baseline: { prediction: number; lower: number; upper: number };
  selected: { prediction: number; lower: number; upper: number } };

// This parser accepts only the fixed, unquoted numeric export produced by Phase 3C.
export function parseSavedPredictions(text: string): SavedPrediction[] {
  const lines = text.replace(/\r\n/g, "\n").trimEnd().split("\n");
  const header = ["Row Index", "Timestamp", "Actual", "Baseline Prediction", "Baseline Lower 95", "Baseline Upper 95", "Baseline Leverage", "Selected Prediction", "Selected Lower 95", "Selected Upper 95", "Selected Leverage"];
  equal(lines.shift()?.split(","), header, "預測 CSV 欄位");
  return lines.map(line => {
    if (line.includes('"')) throw new Error("非預期的 CSV 引號格式");
    const cells = line.split(","); if (cells.length !== header.length || cells.some(v => !v.trim())) throw new Error("預測 CSV 列不完整");
    const value = (i: number) => num(Number(cells[i]));
    const index = integer(value(0)); const time = timestamp(cells[1]); const actual = nonnegative(value(2));
    const model = (offset: number) => {
      const prediction = value(offset); const lower = value(offset + 1); const upper = value(offset + 2);
      nonnegative(value(offset + 3));
      if (lower > prediction || prediction > upper) throw new Error("Prediction Interval 順序錯誤");
      close((lower + upper) / 2, prediction, "區間中心"); return { prediction, lower, upper };
    };
    return { index, timestamp: time, actual, baseline: model(3), selected: model(7) };
  });
}
function quantile(sorted: number[], q: number) { const position = (sorted.length - 1) * q; const i = Math.floor(position); return sorted[i] + (sorted[Math.ceil(position)] - sorted[i]) * (position - i); }

export function adaptEvaluation(modelsValue: unknown, intervalsValue: unknown, metricsValue: unknown, piValue: unknown, planValue: unknown, rows: SavedPrediction[]): EvaluationData {
  const modelsDoc = obj(modelsValue); const intervalsDoc = obj(intervalsValue);
  if (modelsDoc.schema_version !== 1 || modelsDoc.artifact_type !== "model_comparison" || intervalsDoc.schema_version !== 1 || intervalsDoc.artifact_type !== "prediction_intervals") throw new Error("Dashboard schema 錯誤");
  const metrics = obj(metricsValue); const pi = obj(piValue); const officialPlan = obj(planValue); const plan = obj(officialPlan.plan);
  const dm = obj(modelsDoc.data); const di = obj(intervalsDoc.data);
  for (const value of [dm.plan, di.plan, metrics.plan, pi.plan]) equal(value, plan, "Phase 3C 計畫");
  for (const v of [obj(modelsDoc.units), obj(intervalsDoc.units), obj(metrics.metric_units)]) {
    if (v.rmse !== "Wh" || v.mae !== "Wh" || v.mse !== "Wh^2" || v.r2 !== "dimensionless") throw new Error("指標單位不一致");
  }
  if (di.interval_type !== "individual_observation_prediction_interval" || pi.sigma_source !== "Training residuals only" || di.sigma_source !== pi.sigma_source || plan.method_or_parameter_selection_using_holdout !== false || plan.intervals_clipped !== false || plan.no_fit_guard !== true || metrics.evaluation !== "fixed chronological holdout, descriptive" || dm.evaluation !== metrics.evaluation) throw new Error("評估／區間定義不符");
  const trainingRows = integer(plan.training_rows, true); const holdoutRows = integer(plan.holdout_rows, true); const nominalCoverage = fraction(plan.nominal_coverage);
  if (nominalCoverage !== 0.95 || Math.abs(num(plan.alpha) + nominalCoverage - 1) > 1e-12) throw new Error("來源並非已保存的 95% PI");
  if (rows.length !== holdoutRows) throw new Error("完整 Holdout 預測筆數不符");
  rows.forEach((r, i) => {
    if (r.index !== trainingRows + i || i > 0 && Date.parse(r.timestamp + "Z") - Date.parse(rows[i - 1].timestamp + "Z") !== 600000) throw new Error("Holdout 列索引／時間順序不一致");
  });
  let trainingPeriod: EvaluationData["trainingPeriod"] | undefined; let holdoutPeriod: EvaluationData["holdoutPeriod"] | undefined;
  const residuals: { baseline: number[]; selected: number[] } = { baseline: [], selected: [] };
  const actualMean = rows.reduce((sum, r) => sum + r.actual, 0) / rows.length;
  const totalVariation = rows.reduce((sum, r) => sum + (r.actual - actualMean) ** 2, 0);
  if (totalVariation <= 0) throw new Error("常數 Target 無法核對 R²");
  const models = (["baseline", "selected"] as const).map(id => {
    const m = obj(obj(metrics.models)[id]); const dashboard = obj(obj(dm.models)[id]);
    const p = obj(obj(pi.models)[id]); const dashboardPI = obj(obj(di.models)[id]);
    const diagnostics = obj(p.training_diagnostics);
    if (diagnostics.holdout_residuals_used !== false || diagnostics.bounds_clipped_to_zero !== false || diagnostics.training_rows !== trainingRows || p.width_std_ddof !== 0) throw new Error("區間變異來源／寬度定義錯誤");
    equal(dashboardPI, p, "Dashboard PI／正式 PI");
    const training = scores(m.training_metrics); const holdout = scores(m.holdout_metrics);
    equal(scores(dashboard.training_metrics), training, "Training 指標"); equal(scores(dashboard.holdout_metrics), holdout, "Holdout 指標");
    const features = arr(m.selected_features).map(str);
    if (features.length !== integer(m.feature_count, true) || dashboard.feature_count !== features.length || new Set(features).size !== features.length) throw new Error("模型 Features 不一致");
    equal(features, dashboard.selected_features, "選中特徵");
    const tp = period(m.training_period, trainingRows); const hp = period(m.holdout_period, holdoutRows);
    if (trainingPeriod) equal(tp, trainingPeriod, "兩模型 Training 時段"); if (holdoutPeriod) equal(hp, holdoutPeriod, "兩模型 Holdout 時段");
    trainingPeriod = tp; holdoutPeriod = hp;
    if (tp.end >= hp.start || rows[0].timestamp !== hp.start || rows.at(-1)!.timestamp !== hp.end) throw new Error("Training／Holdout 時間邊界錯誤");
    let covered = 0; let below = 0; let above = 0;
    const widths: number[] = [];
    for (const row of rows) {
      const interval = row[id]; residuals[id].push(row.actual - interval.prediction); widths.push(interval.upper - interval.lower);
      if (row.actual < interval.lower) below++; else if (row.actual > interval.upper) above++; else covered++;
    }
    const errors = residuals[id]; const mse = errors.reduce((sum, e) => sum + e * e, 0) / holdoutRows;
    const mae = errors.reduce((sum, e) => sum + Math.abs(e), 0) / holdoutRows;
    const recalculated = { mse, mae, rmse: Math.sqrt(mse), r2: 1 - mse * holdoutRows / totalVariation };
    for (const key of metricKeys) close(recalculated[key], holdout[key], `Holdout ${key}`);
    const averageWidth = widths.reduce((sum, v) => sum + v, 0) / holdoutRows;
    const medianWidth = quantile([...widths].sort((a, b) => a - b), 0.5);
    const widthSD = Math.sqrt(widths.reduce((sum, w) => sum + (w - averageWidth) ** 2, 0) / holdoutRows);
    for (const [computed, saved] of [[covered, p.covered_count], [below, p.below_lower_count], [above, p.above_upper_count], [holdoutRows, p.total_holdout_samples]]) if (computed !== integer(saved)) throw new Error("區間覆蓋計數與逐筆預測不一致");
    if (p.nominal_coverage !== nominalCoverage) throw new Error("名目 Coverage 不一致");
    close(covered / holdoutRows, fraction(p.empirical_coverage), "Coverage");
    close(averageWidth, nonnegative(p.average_width), "平均區間寬度"); close(medianWidth, nonnegative(p.median_width), "中位區間寬度"); close(widthSD, nonnegative(p.width_standard_deviation), "區間寬度 SD");
    const sorted = [...errors].sort((a, b) => a - b);
    return { id, label: id === "baseline" ? "Baseline LR" : "MI + LR", features, training, holdout,
      interval: { coverage: fraction(p.empirical_coverage), averageWidth: num(p.average_width), medianWidth: num(p.median_width), widthSD: num(p.width_standard_deviation), covered, below, above },
      residual: { mean: errors.reduce((sum, e) => sum + e, 0) / holdoutRows, median: quantile(sorted, 0.5), p05: quantile(sorted, 0.05), p95: quantile(sorted, 0.95), maxAbsolute: Math.max(...errors.map(Math.abs)),
        underpredictions: errors.filter(e => e > 0).length, overpredictions: errors.filter(e => e < 0).length, exact: errors.filter(e => e === 0).length } } satisfies EvaluationModel;
  }) as [EvaluationModel, EvaluationModel];
  const limit = Math.max(50, Math.ceil(Math.max(...residuals.baseline.map(Math.abs), ...residuals.selected.map(Math.abs)) / 50) * 50);
  const binCount = 20; const step = 2 * limit / binCount;
  const histogram = Array.from({ length: binCount }, (_, i) => ({ lower: -limit + i * step, upper: -limit + (i + 1) * step, baseline: 0, selected: 0 }));
  for (const id of ["baseline", "selected"] as const) for (const value of residuals[id]) histogram[Math.min(binCount - 1, Math.floor((value + limit) / step))][id]++;
  return { trainingRows, holdoutRows, nominalCoverage, trainingPeriod: trainingPeriod!, holdoutPeriod: holdoutPeriod!, holdoutRole: str(plan.holdout_role), formula: str(pi.formula), models, histogram };
}

export async function readPredictionSource(repo: string) {
  const base = await realpath(repo); const path = EVALUATION_SOURCES.predictions;
  const full = await realpath(resolve(base, path)); const local = relative(base, full);
  if (local.startsWith("..") || isAbsolute(local)) throw new Error("預測來源超出 Repository");
  const bytes = await readFile(full); return { rows: parseSavedPredictions(bytes.toString("utf8")), source: { path, sha256: createHash("sha256").update(bytes).digest("hex") } };
}
export async function buildEvaluationSnapshot(repo: string): Promise<EvaluationSnapshot> {
  const sources: EvaluationSnapshot["sources"] = [];
  try {
    const values: Record<string, unknown> = {};
    for (const key of ["dashboardModels", "dashboardIntervals", "metrics", "intervals", "plan", "figures"] as const) {
      const result = await readSource(repo, EVALUATION_SOURCES[key]); sources.push(result.source); values[key] = result.value;
    }
    const predictions = await readPredictionSource(repo); sources.push(predictions.source);
    if (obj(obj(values.figures).sources)["prediction_interval_predictions.csv"] !== predictions.source.sha256) throw new Error("逐筆預測與既有 Figure Manifest 雜湊不符");
    for (const [document, key] of [["dashboardModels", "metrics"], ["dashboardIntervals", "intervals"]]) {
      const source = sources.find(s => s.path === EVALUATION_SOURCES[key as "metrics" | "intervals"])!;
      const recorded = arr(obj(values[document]).sources).map(obj).find(s => s.path === source.path);
      if (recorded?.sha256 !== source.sha256) throw new Error("Dashboard 與正式成果雜湊不符");
    }
    const data = adaptEvaluation(values.dashboardModels, values.dashboardIntervals, values.metrics, values.intervals, values.plan, predictions.rows);
    return { schemaVersion: 1, status: "ready", sources, data };
  } catch (error) { return { schemaVersion: 1, status: "unavailable", sources, issues: ["Model Evaluation 資料缺少或定義不符。", error instanceof Error ? error.message : "資料錯誤"] }; }
}

export function parseEvaluationSnapshot(value: unknown): EvaluationSnapshot {
  const root = obj(value); if (root.schemaVersion !== 1) throw new Error("Snapshot 版本錯誤");
  for (const value of arr(root.sources)) { const source = obj(value); if (!Object.values(EVALUATION_SOURCES).includes(str(source.path) as typeof EVALUATION_SOURCES[keyof typeof EVALUATION_SOURCES]) || !/^[a-f0-9]{64}$/.test(str(source.sha256))) throw new Error("來源格式錯誤"); }
  if (root.status === "unavailable") { if (!arr(root.issues).map(str).length) throw new Error("缺少錯誤訊息"); return value as EvaluationSnapshot; }
  if (root.status !== "ready") throw new Error("Snapshot 狀態錯誤");
  const d = obj(root.data); integer(d.trainingRows, true); const n = integer(d.holdoutRows, true); fraction(d.nominalCoverage); str(d.holdoutRole); str(d.formula);
  for (const key of ["trainingPeriod", "holdoutPeriod"]) { const p = obj(d[key]); if (timestamp(p.start) >= timestamp(p.end)) throw new Error("時段順序錯誤"); }
  const models = arr(d.models).map(obj); if (models.length !== 2 || models[0].id !== "baseline" || models[1].id !== "selected") throw new Error("兩模型角色錯誤");
  for (const m of models) {
    str(m.label); if (!arr(m.features).map(str).length) throw new Error("缺少 Features"); scores(m.training); scores(m.holdout);
    const p = obj(m.interval); fraction(p.coverage); for (const key of ["averageWidth", "medianWidth", "widthSD"]) nonnegative(p[key]);
    if (integer(p.covered) + integer(p.below) + integer(p.above) !== n) throw new Error("區間計數錯誤"); close(num(p.coverage), num(p.covered) / n, "Coverage");
    const residual = obj(m.residual); for (const key of ["mean", "median", "p05", "p95"]) num(residual[key]); nonnegative(residual.maxAbsolute);
    if (integer(residual.underpredictions) + integer(residual.overpredictions) + integer(residual.exact) !== n) throw new Error("殘差計數錯誤");
  }
  const bins = arr(d.histogram).map(obj); if (bins.length !== 20) throw new Error("殘差分箱不完整");
  for (const [i, b] of bins.entries()) { if (num(b.lower) >= num(b.upper) || i > 0 && b.lower !== bins[i - 1].upper) throw new Error("殘差分箱邊界不一致"); integer(b.baseline); integer(b.selected); }
  for (const id of ["baseline", "selected"]) if (bins.reduce((sum, b) => sum + num(b[id]), 0) !== n) throw new Error("殘差分布未涵蓋完整 Holdout");
  return value as EvaluationSnapshot;
}
