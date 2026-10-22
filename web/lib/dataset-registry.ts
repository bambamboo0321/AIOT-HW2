/** Research metadata only. No model results, filesystem imports, or Custom Dataset state. */
export const DATASET_IDS = ["appliances-energy", "automobile-price", "student-performance"] as const;
export type DatasetId = typeof DATASET_IDS[number];
export const DEFAULT_DATASET_ID: DatasetId = "appliances-energy";
export type ResearchStatus = "connected" | "not-connected";
export type DatasetDefinition = {
  readonly id: DatasetId;
  readonly name: string;
  readonly shortName: string;
  readonly target: string;
  readonly targetUnit: string;
  readonly rows: number;
  readonly candidateFeatures: number;
  readonly protocol: {
    readonly split: "chronological" | "random";
    readonly trainingRows: number;
    readonly holdoutRows: number;
    readonly outerFolds: number;
    readonly innerFolds: number;
    readonly gap: number;
    readonly kPolicy: "inner-cv-search-and-auto" | "fixed-7-and-auto";
    readonly label: string;
  };
  readonly features: {
    readonly overview: ResearchStatus;
    readonly featureSelection: ResearchStatus;
    readonly modelEvaluation: ResearchStatus;
    readonly researchPages: ResearchStatus;
    readonly livePrediction: ResearchStatus;
  };
  readonly note: string;
};

export const DATASET_REGISTRY = {
  "appliances-energy": {
    id: "appliances-energy", name: "Appliances Energy Prediction", shortName: "Appliances Energy",
    target: "Appliances", targetUnit: "Wh", rows: 19735, candidateFeatures: 25,
    protocol: { split: "chronological", trainingRows: 15788, holdoutRows: 3947, outerFolds: 5, innerFolds: 3, gap: 6,
      kPolicy: "inner-cv-search-and-auto", label: "時間切分 · 外層 5-fold／內層 3-fold · gap 6" },
    features: { overview: "connected", featureSelection: "connected", modelEvaluation: "connected", researchPages: "connected", livePrediction: "connected" },
    note: "保留既有能源研究流程與成果；研究頁瀏覽已保存的預測。",
  },
  "automobile-price": {
    id: "automobile-price", name: "Automobile Price", shortName: "Automobile Price",
    target: "price", targetUnit: "原始價格單位（幣別未核實）", rows: 201, candidateFeatures: 14,
    protocol: { split: "random", trainingRows: 160, holdoutRows: 41, outerFolds: 5, innerFolds: 3, gap: 0,
      kPolicy: "fixed-7-and-auto", label: "固定 seed 42 隨機切分 · 外層 5-fold／內層 3-fold · 固定 k=7 與 auto" },
    features: { overview: "connected", featureSelection: "connected", modelEvaluation: "connected", researchPages: "connected", livePrediction: "connected" },
    note: "201 筆有效資料；14 個候選欄位包含 12 個數值欄位與 2 個固定二元類別編碼。",
  },
  "student-performance": {
    id: "student-performance", name: "Student Performance", shortName: "Student Performance",
    target: "G3", targetUnit: "分數（0–20）", rows: 649, candidateFeatures: 15,
    protocol: { split: "random", trainingRows: 519, holdoutRows: 130, outerFolds: 5, innerFolds: 3, gap: 0,
      kPolicy: "fixed-7-and-auto", label: "固定 seed 42 隨機切分 · 外層 5-fold／內層 3-fold · 固定 k=7 與 auto" },
    features: { overview: "connected", featureSelection: "connected", modelEvaluation: "connected", researchPages: "connected", livePrediction: "connected" },
    note: "15 個數值候選欄位包含 G1、G2；使用前兩期成績已知後的條件估計 G3，非開學前預測。",
  },
} as const satisfies Record<DatasetId, DatasetDefinition>;

export function isDatasetId(value: unknown): value is DatasetId {
  return typeof value === "string" && Object.hasOwn(DATASET_REGISTRY, value);
}
export function resolveDatasetId(value: unknown): DatasetId {
  return isDatasetId(value) ? value : DEFAULT_DATASET_ID;
}
export function getDataset(value: unknown): DatasetDefinition {
  return DATASET_REGISTRY[resolveDatasetId(value)];
}
