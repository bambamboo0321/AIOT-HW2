import { readFile, realpath } from "node:fs/promises";
import { createHash } from "node:crypto";
import { isAbsolute, relative, resolve } from "node:path";

export type Provenance = { path: string; sha256: string };
export type Model = {
  id: "baseline" | "selected"; label: string; features: number;
  selectedFeatures: string[]; rmse: number; mae: number; r2: number;
  coverage: number; averageWidth: number;
};
export type ResearchData = {
  dataset: { name: string; records: number; rawFeatures: number; modelingFeatures: number;
    target: string; targetUnit: string; samplingSeconds: number; start: string; end: string };
  trainingRows: number; holdoutRows: number; holdoutRole: string;
  nominalCoverage: number; errorUnit: string; widthUnit: string;
  models: [Model, Model];
  exploration: { available: false; reason: string } | {
    available: true; sampleCount: number; bestRmse: number; fixedMiRmse: number;
    bestFeatures: string[]; betterCount: number; rank: number; rankPopulation: number;
    folds: number; gap: number; warning: string;
  };
};
export type Snapshot = {
  schemaVersion: 1; status: "ready"; sources: Provenance[]; data: ResearchData;
} | { schemaVersion: 1; status: "unavailable"; sources: Provenance[]; issues: string[] };

export const SOURCE_PATHS = {
  dataset: "dashboard_data/dataset_summary.json",
  models: "dashboard_data/model_comparison.json",
  intervals: "dashboard_data/prediction_intervals.json",
  exploration: "reports/phase3/phase3d/comparison_summary.json",
} as const;

type Obj = Record<string, unknown>;
function obj(value: unknown, at: string): Obj {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${at}：缺少 object`);
  return value as Obj;
}
function num(value: unknown, at: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${at}：缺少有限數值`);
  return value;
}
function count(value: unknown, at: string): number {
  const n = num(value, at);
  if (!Number.isInteger(n) || n <= 0) throw new Error(`${at}：必須是正整數`);
  return n;
}
function text(value: unknown, at: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${at}：缺少文字`);
  return value;
}
function strings(value: unknown, at: string): string[] {
  if (!Array.isArray(value) || !value.every(v => typeof v === "string" && v.length > 0)) throw new Error(`${at}：必須是文字陣列`);
  return value as string[];
}
function fraction(value: unknown, at: string): number {
  const n = num(value, at);
  if (n < 0 || n > 1) throw new Error(`${at}：比例超出 0–1`);
  return n;
}
function envelope(value: unknown, name: string): { data: Obj; units: Obj } {
  const root = obj(value, name);
  if (root.schema_version !== 1 || root.artifact_type !== name) throw new Error(`${name}：不相容的 schema 或 artifact_type`);
  return { data: obj(root.data, `${name}.data`), units: obj(root.units, `${name}.units`) };
}

// 明確 allowlist；不讀取原始 CSV、不反序列化模型，也不執行 Python。
export async function readSource(root: string, path: string): Promise<{ value: unknown; source: Provenance }> {
  const base = await realpath(root);
  const full = await realpath(resolve(base, path));
  const local = relative(base, full);
  if (local.startsWith("..") || isAbsolute(local)) throw new Error(`${path}：來源超出 Repository`);
  const bytes = await readFile(full);
  return { value: JSON.parse(bytes.toString("utf8")) as unknown,
    source: { path, sha256: createHash("sha256").update(bytes).digest("hex") } };
}

function explorationAdapter(value: unknown): ResearchData["exploration"] {
  const root = obj(value, "Phase 3D");
  const conditions = obj(root.conditions, "Phase 3D.conditions");
  if (root.status !== "complete" || conditions.holdout_used !== false) throw new Error("Phase 3D：非完整 Training-only 成果");
  const random = obj(root.random_search, "random_search");
  const best = obj(random.best, "random_search.best");
  const reference = obj(random.mi_reference, "mi_reference");
  const fixed = obj(obj(root.fixed_combinations, "fixed_combinations").mutual_information, "fixed MI");
  const features = strings(best.features, "best.features");
  if (features.length !== count(conditions.k, "k")) throw new Error("Phase 3D：最佳組合欄數不符");
  if (random.sample_count !== random.unique_count) throw new Error("Phase 3D：抽樣組合有重複");
  return { available: true, sampleCount: count(random.sample_count, "sample_count"),
    bestRmse: num(best.rmse_wh, "best.rmse_wh"), fixedMiRmse: num(fixed.rmse_wh, "fixed_mi.rmse_wh"),
    bestFeatures: features, betterCount: num(reference.random_better_than_mi, "better count"),
    rank: count(reference.competition_rank_among_random_and_mi, "MI rank"),
    rankPopulation: count(reference.rank_population, "rank population"),
    folds: count(conditions.cv_splits, "CV folds"), gap: num(conditions.gap, "CV gap"),
    warning: text(root.selection_bias, "selection_bias") };
}

export function adaptCore(datasetValue: unknown, modelsValue: unknown, intervalsValue: unknown): Omit<ResearchData, "exploration"> {
  const datasetDoc = envelope(datasetValue, "dataset_summary");
  const modelDoc = envelope(modelsValue, "model_comparison");
  const intervalDoc = envelope(intervalsValue, "prediction_intervals");
  const ds = obj(datasetDoc.data.dataset, "dataset");
  const shape = obj(ds.shape, "dataset.shape");
  const range = obj(ds.date_range, "date_range");
  const plan = obj(modelDoc.data.plan, "model.plan");
  const piPlan = obj(intervalDoc.data.plan, "interval.plan");
  const trainingRows = count(plan.training_rows, "training_rows");
  const holdoutRows = count(plan.holdout_rows, "holdout_rows");
  const nominalCoverage = fraction(piPlan.nominal_coverage, "nominal coverage");
  if (piPlan.training_rows !== trainingRows || piPlan.holdout_rows !== holdoutRows || piPlan.nominal_coverage !== plan.nominal_coverage) throw new Error("模型與 PI 的資料切分／名目機率不一致");
  if (intervalDoc.data.interval_type !== "individual_observation_prediction_interval") throw new Error("區間不是單筆觀測 Prediction Interval");
  const models = obj(modelDoc.data.models, "models");
  const intervals = obj(intervalDoc.data.models, "interval.models");
  const buildModel = (id: Model["id"], label: string): Model => {
    const m = obj(models[id], id); const metrics = obj(m.holdout_metrics, `${id}.holdout_metrics`);
    const pi = obj(intervals[id], `${id}.interval`);
    const features = count(m.feature_count, `${id}.feature_count`);
    const names = strings(m.selected_features, `${id}.features`);
    if (names.length !== features || pi.nominal_coverage !== nominalCoverage || pi.total_holdout_samples !== holdoutRows) throw new Error(`${id}：Features／PI 樣本或機率不一致`);
    const rmse = num(metrics.rmse, `${id}.rmse`); const mae = num(metrics.mae, `${id}.mae`);
    const width = num(pi.average_width, `${id}.average_width`);
    if (rmse < 0 || mae < 0 || width < 0) throw new Error(`${id}：誤差／寬度不能為負`);
    return { id, label, features, selectedFeatures: names, rmse, mae,
      r2: num(metrics.r2, `${id}.r2`), coverage: fraction(pi.empirical_coverage, `${id}.coverage`), averageWidth: width };
  };
  const baseline = buildModel("baseline", "Baseline Linear Regression");
  const selected = buildModel("selected", "MI + Linear Regression");
  const records = count(shape.rows, "records");
  const modelingFeatures = count(datasetDoc.data.baseline_feature_count, "modeling features");
  if (records !== trainingRows + holdoutRows || baseline.features !== modelingFeatures) throw new Error("Dataset 與模型的樣本／欄數不一致");
  return { dataset: { name: text(ds.name, "dataset name"), records,
    rawFeatures: count(ds.original_input_feature_count, "raw features"), modelingFeatures,
    target: text(ds.target, "target"), targetUnit: text(modelDoc.units.rmse, "Wh unit"),
    samplingSeconds: count(ds.sampling_interval_seconds, "sampling interval"),
    start: text(range.start, "start date"), end: text(range.end, "end date") },
    trainingRows, holdoutRows, nominalCoverage,
    holdoutRole: text(plan.holdout_role, "holdout role"),
    errorUnit: text(modelDoc.units.rmse, "error unit"), widthUnit: text(intervalDoc.units.interval_width, "width unit"),
    models: [baseline, selected] };
}

export async function buildSnapshot(root: string): Promise<Snapshot> {
  const sources: Provenance[] = []; const issues: string[] = [];
  const values: unknown[] = [];
  for (const key of ["dataset", "models", "intervals"] as const) {
    try {
      const result = await readSource(root, SOURCE_PATHS[key]);
      sources.push(result.source); values.push(result.value);
    } catch {
      issues.push(`${SOURCE_PATHS[key]}：檔案缺少、無法解析或路徑不合法。`);
    }
  }
  if (issues.length) return { schemaVersion: 1, status: "unavailable", sources, issues };
  try {
    const core = adaptCore(values[0], values[1], values[2]);
    let exploration: ResearchData["exploration"];
    try {
      const result = await readSource(root, SOURCE_PATHS.exploration);
      exploration = explorationAdapter(result.value); sources.push(result.source);
    } catch {
      exploration = { available: false, reason: "Phase 3D 成果缺少或格式不符，無法顯示探索性比較。" };
    }
    return { schemaVersion: 1, status: "ready", sources, data: { ...core, exploration } };
  } catch (error) {
    return { schemaVersion: 1, status: "unavailable", sources,
      issues: [error instanceof Error ? error.message : "成果格式不符。"] };
  }
}

export function parseSnapshot(value: unknown): Snapshot {
  const root = obj(value, "snapshot");
  if (root.schemaVersion !== 1 || !Array.isArray(root.sources)) throw new Error("Snapshot 版本／來源格式錯誤");
  if (root.status === "unavailable") { strings(root.issues, "issues"); return value as Snapshot; }
  if (root.status !== "ready") throw new Error("Snapshot 狀態錯誤");
  const data = obj(root.data, "snapshot.data");
  const dataset = obj(data.dataset, "snapshot.dataset");
  for (const key of ["records", "rawFeatures", "modelingFeatures", "samplingSeconds"]) count(dataset[key], key);
  for (const key of ["name", "target", "targetUnit", "start", "end"]) text(dataset[key], key);
  for (const key of ["trainingRows", "holdoutRows"]) count(data[key], key);
  for (const key of ["errorUnit", "widthUnit", "holdoutRole"]) text(data[key], key);
  fraction(data.nominalCoverage, "nominalCoverage");
  if (dataset.records !== Number(data.trainingRows) + Number(data.holdoutRows)) throw new Error("Snapshot 切分大小不一致");
  if (!Array.isArray(data.models) || data.models.length !== 2) throw new Error("Snapshot 缺少兩個固定模型");
  for (const value of data.models) {
    const model = obj(value, "snapshot.model");
    for (const key of ["rmse", "mae", "r2", "averageWidth"]) num(model[key], key);
    fraction(model.coverage, "coverage"); count(model.features, "features"); text(model.label, "label");
    if (strings(model.selectedFeatures, "selectedFeatures").length !== model.features) throw new Error("Snapshot Features 不一致");
  }
  if (obj(data.models[0], "baseline").id !== "baseline" || obj(data.models[1], "selected").id !== "selected") throw new Error("Snapshot 模型角色不一致");
  const exploration = obj(data.exploration, "exploration");
  if (exploration.available === false) text(exploration.reason, "exploration.reason");
  else if (exploration.available === true) {
    for (const key of ["sampleCount", "rank", "rankPopulation", "folds"]) count(exploration[key], key);
    for (const key of ["bestRmse", "fixedMiRmse", "betterCount", "gap"]) num(exploration[key], key);
    strings(exploration.bestFeatures, "bestFeatures"); text(exploration.warning, "selection bias");
  } else throw new Error("Snapshot 缺少探索性成果狀態");
  return value as Snapshot;
}
