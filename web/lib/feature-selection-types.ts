export const METHOD_NAMES = {
  f_test: "F-test", mutual_information: "Mutual Information", forward_sfs: "Forward SFS",
  rfe: "RFE", lasso_cv: "LassoCV", rfecv: "RFECV",
} as const;
export type MethodId = keyof typeof METHOD_NAMES;
export type Metric = "rmse" | "mae" | "r2";
export type Scores = Record<Metric, number>;
export type CurvePoint = { k: number; scores: Scores | null; validFolds: number };
export type FoldResult = { fold: number; k: number; scores: Scores };
export type SelectionMethod = {
  id: MethodId; automatic: boolean; finalK: number; features: string[];
  transformedFeatures: string[]; finalCV: Scores; meanFinalK: number;
  nestedCV: Scores; outerK: number[]; outerResults: FoldResult[]; curve: CurvePoint[];
};
export type ExplorationRow = { label: string; scores: Scores; features: string[] };
export type FeatureData = {
  trainingRows: number; holdoutRows: number; outerFolds: number; innerFolds: number;
  gap: number; candidateFeatures: number; winner: MethodId; selectionRule: string;
  methods: SelectionMethod[];
  exploration: { available: false; reason: string } | {
    available: true; k: number; folds: number; sampleCount: number;
    selectorRows: { id: MethodId; scores: Scores }[]; fixedRows: ExplorationRow[];
    warning: string;
  };
};
export type FeatureSnapshot = { schemaVersion: 1; sources: { path: string; sha256: string }[] } & (
  { status: "ready"; data: FeatureData } | { status: "unavailable"; issues: string[] }
);
