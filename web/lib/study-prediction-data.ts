import { predictionPoints, type StudyEvaluationModel } from "./study-evaluation-data.ts";

export type StudyPredictionRow = {
  position: number; actual: number; predicted: number; residual: number; absoluteError: number;
  lower: number | null; upper: number | null;
};
export const PREDICTION_SORTS = ["sample", "error-desc", "error-asc"] as const;
export type PredictionSort = typeof PREDICTION_SORTS[number];
/** Same verified Phase 3D arrays as Model Evaluation. Position is Holdout order, not source CSV row ID. */
export function studyPredictionRows(model: StudyEvaluationModel): StudyPredictionRow[] {
  const pi = model.interval;
  return predictionPoints(model).map(p => ({ ...p, absoluteError: Math.abs(p.residual),
    lower: pi.status === "ready" ? pi.lower[p.position - 1] : null,
    upper: pi.status === "ready" ? pi.upper[p.position - 1] : null }));
}
/** Search is an exact positive integer ordinal, never a match on a rounded prediction. */
export function visibleStudyPredictions(rows: readonly StudyPredictionRow[], query: string, sort: PredictionSort): StudyPredictionRow[] {
  const search = query.trim();
  const filtered = rows.filter(p => !search || (/^[1-9]\d*$/.test(search) && p.position === Number(search)));
  return filtered.sort((a, b) => sort === "sample" ? a.position - b.position :
    (sort === "error-desc" ? b.absoluteError - a.absoluteError : a.absoluteError - b.absoluteError) || a.position - b.position);
}
