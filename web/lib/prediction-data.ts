import { readSource } from "./research-data.ts";
import { buildEvaluationSnapshot, readPredictionSource, EVALUATION_SOURCES } from "./evaluation-data.ts";
import { WINDOW_SIZES, type WindowSize, type PredictionMeta, type PredictionSnapshot } from "./prediction-types.ts";
import { predictionWindow, parsePredictionWindow, validatePredictionRows, windowScores } from "./prediction-window.ts";

export async function buildPredictionSnapshot(repo: string): Promise<PredictionSnapshot> {
  const evaluation = await buildEvaluationSnapshot(repo);
  if (evaluation.status !== "ready") return { schemaVersion: 1, status: "unavailable", sources: evaluation.sources, issues: ["Prediction Explorer 無法核實完整評估成果。", ...evaluation.issues] };
  try {
    const predictions = await readPredictionSource(repo);
    const plan = await readSource(repo, EVALUATION_SOURCES.plan);
    if (evaluation.sources.find(s => s.path === predictions.source.path)?.sha256 !== predictions.source.sha256 || evaluation.sources.find(s => s.path === plan.source.path)?.sha256 !== plan.source.sha256) throw new Error("讀取期間成果版本改變，請重新同步");
    const saved = plan.value as { plan: { plot_window: { holdout_start_position: number; consecutive_samples: number } } };
    const declared = saved.plan.plot_window;
    if (!Number.isInteger(declared.holdout_start_position) || !WINDOW_SIZES.includes(declared.consecutive_samples as WindowSize)) throw new Error("預先固定視窗格式不符");
    const e = evaluation.data;
    const rows = predictions.rows.map((r, i) => ({ position: i + 1, timestamp: r.timestamp, actual: r.actual, baseline: r.baseline, selected: r.selected }));
    const meta: PredictionMeta = { totalRows: e.holdoutRows, trainingRows: e.trainingRows, nominalCoverage: e.nominalCoverage, samplingSeconds: 600,
      start: e.holdoutPeriod.start, end: e.holdoutPeriod.end, sourceDigest: predictions.source.sha256,
      defaultStart: declared.holdout_start_position, defaultSize: declared.consecutive_samples as WindowSize,
      models: e.models.map(m => ({ id: m.id, label: m.label, featureCount: m.features.length, holdout: m.holdout, coverage: m.interval.coverage })) };
    predictionWindow(rows, meta, meta.defaultStart, meta.defaultSize);
    return { schemaVersion: 1, status: "ready", sources: evaluation.sources, data: { meta, rows } };
  } catch (error) { return { schemaVersion: 1, status: "unavailable", sources: evaluation.sources, issues: ["逐筆預測資料缺少或格式不符。", error instanceof Error ? error.message : "資料錯誤"] }; }
}
export function parsePredictionSnapshot(value: unknown): PredictionSnapshot {
  if (!value || typeof value !== "object") throw new Error("快照格式錯誤");
  const s = value as PredictionSnapshot;
  if (s.schemaVersion !== 1 || !Array.isArray(s.sources) || new Set(s.sources.map(v => v.path)).size !== s.sources.length || s.sources.some(v => !Object.values(EVALUATION_SOURCES).includes(v.path as typeof EVALUATION_SOURCES[keyof typeof EVALUATION_SOURCES]) || !/^[a-f0-9]{64}$/.test(v.sha256))) throw new Error("快照來源格式錯誤");
  if (s.status === "unavailable") { if (!Array.isArray(s.issues) || !s.issues.length || s.issues.some(i => typeof i !== "string" || !i)) throw new Error("缺少資料錯誤原因"); return s; }
  if (s.status !== "ready" || s.sources.length !== Object.keys(EVALUATION_SOURCES).length || !s.data?.meta) throw new Error("快照狀態錯誤");
  const { meta, rows } = s.data;
  if (!Number.isInteger(meta.totalRows) || meta.totalRows < 1 || !Number.isInteger(meta.trainingRows) || meta.trainingRows < 1 || meta.nominalCoverage !== 0.95 || meta.samplingSeconds !== 600 || s.sources.find(v => v.path === EVALUATION_SOURCES.predictions)?.sha256 !== meta.sourceDigest) throw new Error("快照評估條件不符");
  validatePredictionRows(rows, 0, meta.totalRows);
  if (rows[0].timestamp !== meta.start || rows.at(-1)!.timestamp !== meta.end || meta.models.length !== 2 || meta.models[0].id !== "baseline" || meta.models[1].id !== "selected") throw new Error("快照時段／模型不一致");
  for (const model of meta.models) {
    if (!model.label || !Number.isInteger(model.featureCount) || model.featureCount < 1 || !Number.isFinite(model.coverage) || model.coverage < 0 || model.coverage > 1) throw new Error("模型摘要不完整");
    const actual = windowScores(rows, model.id);
    for (const key of ["rmse", "mae", "r2"] as const) if (!Number.isFinite(model.holdout[key]) || actual[key] === null || Math.abs(Number(actual[key]) - model.holdout[key]) > 1e-8 * Math.max(1, Math.abs(model.holdout[key]))) throw new Error("完整 Holdout 指標不符");
    if (!Number.isFinite(model.holdout.mse) || Math.abs(model.holdout.rmse ** 2 - model.holdout.mse) > 1e-6 || actual.coverage !== model.coverage) throw new Error("完整 Holdout MSE／Coverage 不符");
  }
  const initial = predictionWindow(rows, meta, meta.defaultStart, meta.defaultSize); if (initial.start !== meta.defaultStart) throw new Error("預設視窗超出範圍");
  parsePredictionWindow(initial); return s;
}
