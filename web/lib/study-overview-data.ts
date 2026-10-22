import { DATASET_REGISTRY, type DatasetId } from "./dataset-registry.ts";

export type StudyDatasetId = Exclude<DatasetId, "appliances-energy">;
export type Metrics = { mae: number; rmse: number; r2: number };
export type StudyInterval = { status: "ready"; nominal: number; coverage: number; averageWidth: number; covered: number; total: number }
  | { status: "unavailable"; reason: string };
export type StudyModel = { id: "baseline" | "mutual_information"; label: string } & (
  { status: "ready"; features: string[]; k: number; holdout: Metrics; cv: Metrics; interval: StudyInterval }
  | { status: "unavailable"; reason: string });
export type StudyOverview = {
  schemaVersion: 1; datasetId: StudyDatasetId;
  dataset: { name: string; target: string; unit: string; rawRows: number; rows: number; candidates: string[]; removedTargets: number; rawFile: string; rawSha256: string };
  protocol: { trainRows: number; holdoutRows: number; seed: 42; outerFolds: 5; innerFolds: 3; fixedK: 7; split: "random"; scaling: "standard"; imputation: "median"; selectionLockedBeforeHoldout: true; selectedMethod: "mutual_information" };
  sources: { path: string; sha256: string }[];
  models: [StudyModel, StudyModel];
};
export type StudySnapshot = { status: "ready"; data: StudyOverview } | { status: "unavailable"; issues: string[] };
const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
function requireValue(condition: unknown, field: string): asserts condition {
  if (!condition) throw new Error(`研究快照欄位不完整或不相容：${field}`);
}
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const count = (v: unknown): v is number => finite(v) && Number.isSafeInteger(v) && v >= 0;
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const hash = (v: unknown) => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const stringList = (v: unknown): v is string[] => Array.isArray(v) && v.length > 0 && v.every(text) && new Set(v).size === v.length;
function metrics(v: unknown) {
  requireValue(isObject(v) && finite(v.mae) && v.mae >= 0 && finite(v.rmse) && v.rmse >= 0 && finite(v.r2) && v.r2 <= 1, "metrics");
}
/** Fail closed: never substitute Energy data for a missing or invalid study snapshot. */
export function parseStudyOverview(value: unknown, id: StudyDatasetId): StudySnapshot {
  try {
    requireValue(isObject(value) && value.schemaVersion === 1 && value.datasetId === id, "schemaVersion/datasetId");
    const d = value.dataset, p = value.protocol, registry = DATASET_REGISTRY[id];
    requireValue(isObject(d) && d.name === registry.name && d.target === registry.target && d.unit === registry.targetUnit && d.rows === registry.rows, "dataset");
    requireValue(count(d.rawRows) && count(d.removedTargets) && d.rawRows - d.removedTargets === d.rows && stringList(d.candidates) && d.candidates.length === registry.candidateFeatures && !d.candidates.includes(d.target as string), "rows/candidates");
    requireValue(text(d.rawFile) && !/[\\/]/.test(d.rawFile) && hash(d.rawSha256), "raw provenance");
    requireValue(isObject(p) && p.trainRows === registry.protocol.trainingRows && p.holdoutRows === registry.protocol.holdoutRows && p.seed === 42 && p.outerFolds === 5 && p.innerFolds === 3 && p.fixedK === 7 && p.split === "random" && p.scaling === "standard" && p.imputation === "median" && p.selectionLockedBeforeHoldout === true && p.selectedMethod === "mutual_information", "protocol");
    const prefix = id === "automobile-price" ? "experiments/multi-dataset-phase2/" : "experiments/student-performance-phase2/";
    requireValue(Array.isArray(value.sources) && value.sources.length >= 4 && value.sources.every(s => isObject(s) && text(s.path) && s.path.startsWith(prefix) && !s.path.includes("..") && hash(s.sha256)), "sources");
    requireValue(Array.isArray(value.models) && value.models.length === 2, "models");
    const candidates = d.candidates;
    for (const [i, m] of value.models.entries()) {
      requireValue(isObject(m) && m.id === (i === 0 ? "baseline" : "mutual_information") && m.label === (i === 0 ? "Baseline LR" : "MI + LR"), "model identity");
      if (m.status === "unavailable") { requireValue(text(m.reason), "model reason"); continue; }
      requireValue(m.status === "ready" && stringList(m.features) && m.features.every(f => candidates.includes(f)) && m.k === m.features.length && m.k === (i === 0 ? registry.candidateFeatures : 7), "selected features/k");
      if (i === 0) requireValue(JSON.stringify(m.features) === JSON.stringify(d.candidates), "baseline features");
      metrics(m.holdout); metrics(m.cv);
      const pi = m.interval;
      requireValue(isObject(pi), "interval");
      if (pi.status === "unavailable") { requireValue(text(pi.reason), "interval reason"); continue; }
      requireValue(pi.status === "ready" && pi.nominal === 0.95 && finite(pi.coverage) && pi.coverage >= 0 && pi.coverage <= 1 && finite(pi.averageWidth) && pi.averageWidth > 0 && count(pi.covered) && count(pi.total) && pi.total === p.holdoutRows && pi.covered <= pi.total && Math.abs(pi.coverage - pi.covered / pi.total) < 1e-12, "interval coverage");
    }
    return { status: "ready", data: value as StudyOverview };
  } catch (e) {
    return { status: "unavailable", issues: [e instanceof Error ? e.message : "研究快照無法讀取。"] };
  }
}
