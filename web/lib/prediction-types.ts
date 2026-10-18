import type { EvaluationScores } from "./evaluation-types.ts";
export const WINDOW_SIZES = [72, 144, 288] as const;
export type WindowSize = typeof WINDOW_SIZES[number];
export type PredictionModelId = "baseline" | "selected";
export type PredictionRow = { position: number; timestamp: string; actual: number } & Record<PredictionModelId, { prediction: number; lower: number; upper: number }>;
export type PredictionMeta = {
  totalRows: number; trainingRows: number; nominalCoverage: number; samplingSeconds: number;
  start: string; end: string; sourceDigest: string; defaultStart: number; defaultSize: WindowSize;
  models: { id: PredictionModelId; label: string; featureCount: number; holdout: EvaluationScores; coverage: number }[];
};
export type WindowScores = { rmse: number; mae: number; r2: number | null; coverage: number; averageWidth: number; covered: number; above: number; below: number };
export type PredictionWindow = { start: number; size: WindowSize; totalRows: number; sourceDigest: string; rows: PredictionRow[]; stats: Record<PredictionModelId, WindowScores> };
export type PredictionSnapshot = { schemaVersion: 1; sources: { path: string; sha256: string }[] } & (
  { status: "ready"; data: { meta: PredictionMeta; rows: PredictionRow[] } } | { status: "unavailable"; issues: string[] }
);
