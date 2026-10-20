import { isAutoMethod, isNestedMethod, parseSelectionDetails, type SelectionDetails } from "./custom-selection-policy.ts";
import { parsePredictionModel, parseTrainingInputRanges, type TrainingInputRanges, type PredictionModel } from "./custom-prediction-types.ts";
import { parseValidationReport, type ValidationReport } from "./custom-validation-result.ts";
export type SelectionMethod = "none" | "f_test" | "mutual_information" | "rfe" | "forward_sfs" | "rfecv" | "lasso_cv";
export const SELECTION_LABELS: Record<SelectionMethod, string> = { none: "不自動選欄", f_test: "F-test", mutual_information: "Mutual Information", rfe: "RFE", forward_sfs: "Forward SFS", rfecv: "RFECV", lasso_cv: "LassoCV" };
export type Scores = { rmse: number; mae: number; r2: number | null };
export type Average = Record<keyof Scores, { mean: number | null; std: number | null; defined_folds: number }>;
type Fold = { fold: number; training_rows: number; validation_rows: number; linear_regression: Scores; all_features: Scores; training_mean: Scores };
export type TrainingReport = { schema_version: 1; scope: "custom_baseline_training" } & (
  { status: "rejected" | "failed"; training_started: boolean; errors: { stage: string; message: string }[] } |
  { status: "completed"; training_started: true; validation: Extract<ValidationReport, { status: "valid" }>; method: "linear_regression"; target: string; features: string[];
    prediction_model?: PredictionModel;
    prediction_input_ranges?: TrainingInputRanges;
    selection: { method: SelectionMethod; requested_k: number | null; ranking_performed: boolean; final_features: string[]; details?: SelectionDetails; fold_features: { fold: number; features: string[]; details?: SelectionDetails }[] };
    cv: { folds: Fold[]; linear_regression: Average; all_features: Average; training_mean: Average }; holdout: { rows: number; linear_regression: Scores; all_features: Scores; training_mean: Scores };
    preview: { position: number; source_row: number; timestamp: string | null; actual: number; prediction: number; baseline_prediction: number; training_mean: number }[]; warnings: string[]; limits: string[] }
);
function object(v: unknown): Record<string, unknown> { if (!v || typeof v !== "object" || Array.isArray(v)) throw new Error("訓練結果格式錯誤。"); return v as Record<string, unknown>; }
function str(v: unknown, max = 4000): string { if (typeof v !== "string" || !v || v.length > max) throw new Error("訓練訊息格式錯誤。"); return v; }
function finite(v: unknown): number { if (typeof v !== "number" || !Number.isFinite(v)) throw new Error("訓練指標非有限數值。"); return v; }
function count(v: unknown, min = 1, max = 50000): number { const n = finite(v); if (!Number.isInteger(n) || n < min || n > max) throw new Error("訓練筆數不符。"); return n; }
function strings(v: unknown, max: number): string[] { if (!Array.isArray(v) || v.length > max) throw new Error("訓練清單格式錯誤。"); return v.map(s => str(s)); }
function scores(v: unknown): Scores { const s = object(v); const rmse = finite(s.rmse), mae = finite(s.mae), r2 = s.r2 === null ? null : finite(s.r2); if (rmse < 0 || mae < 0 || mae > rmse + 1e-8 * Math.max(1, rmse) || (r2 !== null && r2 > 1 + 1e-10)) throw new Error("訓練指標範圍不符。"); return { rmse, mae, r2 }; }
function average(v: unknown, folds: (Scores|{rmse:null;mae:null;r2:null})[]): Average {
  const input = object(v); const result = {} as Average;
  for (const key of ["rmse", "mae", "r2"] as const) {
    const s = object(input[key]); const values = folds.map(f => f[key]).filter((n): n is number => n !== null);
    const n = count(s.defined_folds, 0, folds.length); if (n !== values.length) throw new Error("CV 有效 folds 不符。");
    if (!n) { if (s.mean !== null || s.std !== null) throw new Error("未定義 CV 指標不可補值。"); result[key] = { mean: null, std: null, defined_folds: 0 }; continue; }
    const mean = finite(s.mean), std = finite(s.std); const expected = values.reduce((a,b) => a+b,0) / n;
    const expectedStd = Math.sqrt(values.reduce((sum,x) => sum + (x-expected)**2,0) / n);
    if (std < 0 || Math.abs(mean-expected) > 1e-8*Math.max(1,Math.abs(expected)) || Math.abs(std-expectedStd) > 1e-8*Math.max(1,expectedStd)) throw new Error("CV 平均或標準差不一致。");
    result[key] = { mean, std, defined_folds: n };
  }
  return result;
}
export function parseTrainingReport(value: unknown): TrainingReport {
  const v = object(value); if (v.schema_version !== 1 || v.scope !== "custom_baseline_training") throw new Error("訓練結果版本或範圍不符。");
  const common = { schema_version: 1 as const, scope: "custom_baseline_training" as const };
  if (v.status === "rejected" || v.status === "failed") {
    if (v.training_started !== (v.status === "failed") || !Array.isArray(v.errors) || !v.errors.length || v.errors.length > 20) throw new Error("訓練錯誤格式不符。");
    return { ...common, status: v.status, training_started: v.training_started, errors: v.errors.map(row => { const e=object(row); const stage=str(e.stage); if (!["confirmation","config","csv","data","review","split","training"].includes(stage)) throw new Error("未知錯誤階段。"); return { stage, message: str(e.message) }; }) };
  }
  if (v.status !== "completed" || v.training_started !== true || v.method !== "linear_regression") throw new Error("未知訓練狀態或方法。");
  const validation = parseValidationReport(v.validation); if (validation.status !== "valid" || !validation.review_checked) throw new Error("訓練缺少已確認的有效資料。");
  const features = strings(v.features, 199); const target = str(v.target, 128); if (!features.length || features.some(f => f.length > 128 || f === target) || new Set(features).size !== features.length || features.length >= validation.dataset.columns) throw new Error("訓練輸入欄位不符。");
  const rawSelection = object(v.selection); const selectionMethod = str(rawSelection.method) as SelectionMethod;
  if (!["none","f_test","mutual_information","rfe","forward_sfs","rfecv","lasso_cv"].includes(selectionMethod)) throw new Error("未知特徵選擇方法。");
  const requestedK = (selectionMethod === "none" || isAutoMethod(selectionMethod)) ? null : count(rawSelection.requested_k,1,features.length);
  if ((selectionMethod === "none" || isAutoMethod(selectionMethod)) && rawSelection.requested_k !== null) throw new Error("不選欄不可指定 k。");
  const details=isNestedMethod(selectionMethod)?parseSelectionDetails(rawSelection.details,selectionMethod,validation.split.training_rows,features.length,requestedK):undefined;
  const rankingPerformed = details ? !details.search_bypassed : selectionMethod !== "none" && requestedK !== features.length;
  if (rawSelection.ranking_performed !== rankingPerformed) throw new Error("特徵排名狀態不符。");
  const selectedNames = (value: unknown) => { const names=strings(value,features.length); const expected=requestedK ?? features.length; if ((isAutoMethod(selectionMethod)?names.length<1:names.length!==expected) || new Set(names).size!==names.length || names.some(n=>!features.includes(n)) || JSON.stringify(names)!==JSON.stringify(features.filter(n=>names.includes(n)))) throw new Error("選中特徵數量、欄名或順序不符。"); return names; };
  const finalFeatures=selectedNames(rawSelection.final_features);
  if (!Array.isArray(rawSelection.fold_features) || rawSelection.fold_features.length!==validation.split.cv_folds.length) throw new Error("缺少逐 fold 選欄。");
  const foldFeatures=rawSelection.fold_features.map((row,i)=>{ const f=object(row); if (f.fold!==i+1) throw new Error("選欄 fold 順序不符。"); const details=isNestedMethod(selectionMethod)?parseSelectionDetails(f.details,selectionMethod,validation.split.cv_folds[i].training_rows,features.length,requestedK):undefined;return {fold:i+1,features:selectedNames(f.features),...(details?{details}:{})}; });
  const selection={method:selectionMethod,requested_k:requestedK,ranking_performed:rankingPerformed,final_features:finalFeatures,fold_features:foldFeatures,...(details?{details}:{})};
  const c = object(v.cv); if (!Array.isArray(c.folds) || c.folds.length !== validation.split.cv_folds.length) throw new Error("CV folds 數量不符。");
  const folds = c.folds.map((row,i) => { const f=object(row), expected=validation.split.cv_folds[i]; if (f.fold !== expected.fold || f.training_rows !== expected.training_rows || f.validation_rows !== expected.validation_rows) throw new Error("CV 切分不一致。"); return { ...expected, linear_regression: scores(f.linear_regression), all_features: scores(f.all_features), training_mean: scores(f.training_mean) }; });
  const h=object(v.holdout); if (h.rows !== validation.split.holdout_rows) throw new Error("Holdout 筆數不符。");
  if (!Array.isArray(v.preview) || v.preview.length !== Math.min(144,h.rows as number)) throw new Error("預覽筆數不符。");
  const preview=v.preview.map((row,i) => { const p=object(row); if (p.position !== i+1 || (validation.split.strategy === "random" ? p.timestamp !== null : typeof p.timestamp !== "string" || !/^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?)?$/.test(p.timestamp))) throw new Error("預覽順序或時間格式不符。"); return { position:i+1, source_row:count(p.source_row,1,validation.dataset.rows), timestamp:p.timestamp as string|null, actual:finite(p.actual), prediction:finite(p.prediction), baseline_prediction:finite(p.baseline_prediction), training_mean:finite(p.training_mean) }; });
  if (new Set(preview.map(p=>p.source_row)).size !== preview.length || preview.some(p=>p.training_mean!==preview[0].training_mean)) throw new Error("預覽原列或平均基準不一致。");
  const predictionModel=v.prediction_model===undefined?undefined:parsePredictionModel(v.prediction_model);
  if(predictionModel && (predictionModel.target!==target || predictionModel.dataset_sha256!==validation.dataset.sha256 || predictionModel.config_sha256!==validation.config_sha256 || JSON.stringify(predictionModel.features)!==JSON.stringify(finalFeatures))) throw new Error("暫存模型與訓練結果不符。");
  const inputRanges=v.prediction_input_ranges===undefined?undefined:parseTrainingInputRanges(v.prediction_input_ranges,finalFeatures,validation.split.training_rows);
  return { ...common,...(inputRanges?{prediction_input_ranges:inputRanges}:{}),...(predictionModel?{prediction_model:predictionModel}:{}),status:"completed",training_started:true,validation,method:"linear_regression",target,features,selection,cv:{folds,linear_regression:average(c.linear_regression,folds.map(f=>f.linear_regression)),training_mean:average(c.training_mean,folds.map(f=>f.training_mean)),all_features:average(c.all_features,folds.map(f=>f.all_features))},holdout:{rows:count(h.rows),linear_regression:scores(h.linear_regression),all_features:scores(h.all_features),training_mean:scores(h.training_mean)},preview,warnings:strings(v.warnings,250),limits:strings(v.limits,20) };
}

// Shared invariant checks for numeric regression metrics in comparison reports.
export { scores as parseScores, average as parseAverage };
