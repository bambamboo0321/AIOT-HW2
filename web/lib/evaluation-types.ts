export type EvaluationMetric = "rmse" | "mae" | "r2" | "mse";
export type EvaluationScores = Record<EvaluationMetric, number>;
export type EvaluationModel = {
  id: "baseline" | "selected"; label: string; features: string[];
  training: EvaluationScores; holdout: EvaluationScores;
  interval: { coverage: number; averageWidth: number; medianWidth: number; widthSD: number;
    covered: number; below: number; above: number };
  residual: { mean: number; median: number; p05: number; p95: number; maxAbsolute: number;
    underpredictions: number; overpredictions: number; exact: number };
};
export type EvaluationData = {
  trainingRows: number; holdoutRows: number; nominalCoverage: number;
  trainingPeriod: { start: string; end: string }; holdoutPeriod: { start: string; end: string };
  holdoutRole: string; formula: string;
  models: [EvaluationModel, EvaluationModel];
  histogram: { lower: number; upper: number; baseline: number; selected: number }[];
};
export type EvaluationSnapshot = { schemaVersion: 1; sources: { path: string; sha256: string }[] } & (
  { status: "ready"; data: EvaluationData } | { status: "unavailable"; issues: string[] }
);
