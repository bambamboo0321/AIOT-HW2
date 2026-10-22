import { METHOD_NAMES, type MethodId, type Scores, type Metric } from "./feature-selection-types.ts";
import { parseStudyOverview, type StudyDatasetId, type StudyOverview } from "./study-overview-data.ts";
export const STUDY_METHODS = Object.keys(METHOD_NAMES) as MethodId[];
export type StudyCV = { status: "ready"; mean: Scores; std: Scores; foldCount: 5; stdDDoF: 0 } | { status: "unavailable"; reason: string };
export type StudySelectionMethod = { id: MethodId; mode: "fixed" | "auto"; requestedK: 7 | null } & (
  { status: "ready"; finalK: number; features: string[]; cv: StudyCV; folds: { fold: number; k: number; features: string[]; scores: Scores | null }[]; warnings: string[] }
  | { status: "unavailable"; reason: string });
export type StudySelectionData = {
  schemaVersion: 1; artifactType: "study-feature-selection"; datasetId: StudyDatasetId; overviewSha256: string;
  sources: StudyOverview["sources"]; methods: StudySelectionMethod[];
  study: Pick<StudyOverview, "dataset" | "protocol">;
};
export type StudySelectionSnapshot = { status: "ready"; data: StudySelectionData } | { status: "unavailable"; issues: string[] };
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
function requireValue(v: unknown, field: string): asserts v { if (!v) throw new Error(`特徵選擇快照缺失或格式不符：${field}`); }
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const count = (v: unknown): v is number => finite(v) && Number.isSafeInteger(v) && v > 0;
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const scores = (v: unknown): v is Scores => object(v) && finite(v.rmse) && v.rmse >= 0 && finite(v.mae) && v.mae >= 0 && finite(v.r2) && v.r2 <= 1;
const deviations = (v: unknown): v is Scores => object(v) && finite(v.rmse) && v.rmse >= 0 && finite(v.mae) && v.mae >= 0 && finite(v.r2) && v.r2 >= 0;
const near = (a: number,b: number) => Math.abs(a-b) <= 1e-10 * Math.max(1,Math.abs(a),Math.abs(b));
function features(v: unknown, candidates: string[]): v is string[] {
  return Array.isArray(v) && v.length > 0 && new Set(v).size === v.length && v.every(f => typeof f === "string" && candidates.includes(f));
}
/** The linked Overview adapter owns dataset/protocol validation; no Energy fallback. */
export function parseStudyFeatureSelection(value: unknown, id: StudyDatasetId, overviewHash: string, overview: unknown): StudySelectionSnapshot {
  try {
    const base = parseStudyOverview(overview,id);
    requireValue(base.status === "ready", "linked Overview unavailable");
    requireValue(object(value) && value.schemaVersion === 1 && value.artifactType === "study-feature-selection" && value.datasetId === id && /^[a-f0-9]{64}$/.test(overviewHash) && value.overviewSha256 === overviewHash,"version/dataset/Overview hash");
    requireValue(JSON.stringify(value.sources) === JSON.stringify(base.data.sources), "source hashes");
    requireValue(Array.isArray(value.methods) && value.methods.length === 6, "six methods");
    const candidates = base.data.dataset.candidates;
    for (const [i,m] of value.methods.entries()) {
      const auto = STUDY_METHODS[i] === "lasso_cv" || STUDY_METHODS[i] === "rfecv";
      requireValue(object(m) && m.id === STUDY_METHODS[i] && m.mode === (auto ? "auto" : "fixed") && m.requestedK === (auto ? null : 7), "method/config");
      if (m.status === "unavailable") { requireValue(text(m.reason), "method reason"); continue; }
      requireValue(m.status === "ready" && features(m.features,candidates) && count(m.finalK) && m.finalK === m.features.length && (auto || m.finalK === 7), "final selection");
      requireValue(Array.isArray(m.warnings) && m.warnings.every(text), "warnings");
      requireValue(Array.isArray(m.folds) && m.folds.length <= 5 && new Set(m.folds.map(f => object(f) ? f.fold : null)).size === m.folds.length, "folds");
      for (const f of m.folds) requireValue(object(f) && count(f.fold) && f.fold <= 5 && count(f.k) && features(f.features,candidates) && f.features.length === f.k && (auto || f.k === 7) && (f.scores === null || scores(f.scores)), "fold result");
      const cv = m.cv;
      requireValue(object(cv), "CV");
      if (cv.status === "unavailable") { requireValue(text(cv.reason), "CV reason"); continue; }
      requireValue(cv.status === "ready" && cv.foldCount === 5 && cv.stdDDoF === 0 && scores(cv.mean) && deviations(cv.std) && m.folds.length === 5 && m.folds.every(f => object(f) && scores(f.scores)), "complete CV scores");
      const folds = m.folds as {scores: Scores}[];
      for (const k of ["rmse","mae","r2"] as Metric[]) {
        const mean = folds.reduce((s,f) => s+f.scores[k],0)/5;
        const std = Math.sqrt(folds.reduce((s,f) => s+(f.scores[k]-mean)**2,0)/5);
        requireValue(near(cv.mean[k],mean) && near(cv.std[k] as number,std), "CV mean/std versus folds");
      }
    }
    return { status:"ready", data:{ ...(value as Omit<StudySelectionData,"study">), study:{dataset:base.data.dataset,protocol:base.data.protocol} } };
  } catch (error) {
    return { status:"unavailable", issues:[error instanceof Error ? error.message : "特徵選擇快照無法讀取。"] };
  }
}
export function studyComparisonRows(data: StudySelectionData, metric: Metric) {
  return data.methods.map(m => ({id:m.id,label:METHOD_NAMES[m.id],value:m.status === "ready" && m.cv.status === "ready" ? m.cv.mean[metric] : null}));
}
