import type { PreprocessingPlan } from "./custom-dataset.ts";
import { parseTrainingReport, type TrainingReport, type Scores } from "./custom-training-result.ts";
import { parsePredictionModel, parsePredictionResult, type PredictionModel, type PredictionResult } from "./custom-prediction-types.ts";
export type DownloadArtifact = { filename: string; mime: string; content: string };
type Completed = Extract<TrainingReport,{status:"completed"}>;
type ModelKey = "linear_regression" | "all_features" | "training_mean";
function completed(value: Completed): Completed {
  const report=parseTrainingReport(value);if(report.status!=="completed")throw new Error("只能下載已完成的評估。");return report;
}
function json(filename: string, value: unknown): DownloadArtifact {return {filename,mime:"application/json;charset=utf-8",content:JSON.stringify(value,null,2)+"\n"};}
// Text cells may contain user-chosen column names. Neutralize spreadsheet formulas
// without changing numeric negatives, and quote every string for commas/newlines.
export function csvCell(value: string | number | null): string {
  if(value===null)return "";
  if(typeof value==="number"){if(!Number.isFinite(value))throw new Error("無法匯出非有限數字。");return String(value);}
  const safe=/^[\s\u0000-\u001f]*[=+\-@]/u.test(value)||/^[\t\r\n]/.test(value)?"'"+value:value;
  return '"'+safe.replaceAll('"','""')+'"';
}
function csv(filename:string,rows:(string|number|null)[][]):DownloadArtifact{return {filename,mime:"text/csv;charset=utf-8",content:"\uFEFF"+rows.map(row=>row.map(csvCell).join(",")).join("\r\n")+"\r\n"};}
function models(r:Completed):ModelKey[]{return r.selection.method==="none"?["linear_regression","training_mean"]:["linear_regression","all_features","training_mean"];}
function planSnapshot(config:PreprocessingPlan,r:Completed):PreprocessingPlan {
  if(config.target_column!==r.target || JSON.stringify(config.feature_columns)!==JSON.stringify(r.features) || config.split_strategy!==r.validation.split.strategy || config.cv_splits!==r.cv.folds.length)throw new Error("設定與評估版本不一致。");
  // Whitelist the supported plan fields; never spread arbitrary caller properties.
  return {schema_version:1,target_column:config.target_column,feature_columns:[...config.feature_columns],column_types:Object.fromEntries(Object.entries(config.column_types).filter(([n])=>n===r.target||r.features.includes(n))),missing_numeric:config.missing_numeric,missing_categorical:config.missing_categorical,missing_target:config.missing_target,categorical_constant:config.categorical_constant,categorical_encoding:config.categorical_encoding,numerical_scaling:config.numerical_scaling,outlier_strategy:config.outlier_strategy,iqr_multiplier:config.iqr_multiplier,datetime_strategy:config.datetime_strategy,datetime_formats:{},split_strategy:config.split_strategy,split_column:config.split_column,test_size:config.test_size,random_state:config.random_state,cv_splits:config.cv_splits,cv_gap:config.cv_gap};
}
export function evaluationJson(value:Completed,config:PreprocessingPlan):DownloadArtifact {
  const r=completed(value);
  return json("custom-evaluation-report.json",{schema_version:1,artifact_type:"custom_dataset_evaluation",scope:"evaluation_summary",source:{dataset:r.validation.dataset,config_sha256:r.validation.config_sha256},preprocessing:planSnapshot(config,r),predictor:r.method,target:r.target,candidate_features:r.features,selection:r.selection,split:r.validation.split,cv:r.cv,holdout:r.holdout,displayed_comparators:models(r),preview:{included:false,available_rows:r.preview.length,holdout_rows:r.holdout.rows,limit:144},warnings:r.warnings,limits:[...r.limits,"此檔案是評估摘要與設定，不含原始 CSV、逐筆預覽、模型或可用於推論的模型代碼。","CV 平均／標準差按每 fold 等權計算，ddof=0；不是信賴區間。","R² 無法定義時保持 null；輸出的成績來自全部 Holdout，並非只計預覽資料。"]});
}
export function evaluationCsv(value:Completed):DownloadArtifact {
  const r=completed(value);const rows:(string|number|null)[][]=[["evaluation","model","fold","training_rows","evaluation_rows","aggregation","rmse","rmse_std","mae","mae_std","r2","r2_std","r2_defined_folds","total_folds","r2_status","target","experiment_selection_method","experiment_requested_k","split_strategy","dataset_sha256","config_sha256"]];
  const append=(scope:string,model:ModelKey,fold:number|null,train:number|null,size:number|null,scores:Scores)=>rows.push([scope,model,fold,train,size,"all_evaluation_rows",scores.rmse,null,scores.mae,null,scores.r2,null,null,null,scores.r2===null?"undefined":"defined",r.target,r.selection.method,r.selection.requested_k,r.validation.split.strategy,r.validation.dataset.sha256,r.validation.config_sha256]);
  for(const model of models(r))append("holdout",model,null,r.validation.split.training_rows,r.holdout.rows,r.holdout[model]);
  for(const model of models(r)){const a=r.cv[model];rows.push(["training_cv_summary",model,null,null,null,"equal_fold_mean_std_ddof0",a.rmse.mean,a.rmse.std,a.mae.mean,a.mae.std,a.r2.mean,a.r2.std,a.r2.defined_folds,r.cv.folds.length,a.r2.defined_folds===r.cv.folds.length?"all_folds_defined":a.r2.defined_folds?"partially_defined":"undefined",r.target,r.selection.method,r.selection.requested_k,r.validation.split.strategy,r.validation.dataset.sha256,r.validation.config_sha256]);}
  for(const fold of r.cv.folds)for(const model of models(r))append("training_cv_fold",model,fold.fold,fold.training_rows,fold.validation_rows,fold[model]);
  rows[0].push("experiment_final_k","cv_fold_selected_k","selector_alpha","selector_features_json");
  for(const row of rows.slice(1)){const fold=typeof row[2]==="number"?r.selection.fold_features.find(f=>f.fold===row[2]):undefined;const final=row[0]==="holdout";row.push(r.selection.final_features.length,fold?.features.length??null,(final?r.selection.details?.alpha:fold?.details?.alpha)??null,final?JSON.stringify(r.selection.final_features):fold?JSON.stringify(fold.features):null);}
  return csv("custom-evaluation-metrics.csv",rows);
}
export function previewCsv(value:Completed):DownloadArtifact {
  const r=completed(value);
  return csv("custom-holdout-preview.csv",[["scope","holdout_total_rows","exported_preview_rows","preview_limit","position","source_csv_row","timestamp","actual","active_lr_prediction","all_candidate_lr_prediction","training_mean_prediction","target","experiment_selection_method","experiment_requested_k","split_strategy","dataset_sha256","config_sha256"],...r.preview.map(p=>["holdout_preview",r.holdout.rows,r.preview.length,144,p.position,p.source_row,p.timestamp,p.actual,p.prediction,p.baseline_prediction,p.training_mean,r.target,r.selection.method,r.selection.requested_k,r.validation.split.strategy,r.validation.dataset.sha256,r.validation.config_sha256])]);
}
export function predictionJson(modelValue:PredictionModel,value:PredictionResult,values:Record<string,number|null>):DownloadArtifact {
  const model=parsePredictionModel(modelValue);const r=parsePredictionResult(value,model);const names=model.features;
  if(Object.keys(values).length!==names.length || Object.keys(values).some(n=>!names.includes(n)) || Object.values(values).some(n=>n!==null&&(typeof n!=="number"||!Number.isFinite(n)||Math.abs(n)>Number.MAX_SAFE_INTEGER)) || !Object.values(values).some(n=>n!==null))throw new Error("預測缺少送出當時的有效輸入。");
  const missing=names.filter(n=>values[n]===null);
  if(JSON.stringify(missing)!==JSON.stringify(r.imputed.map(row=>row.feature)))throw new Error("輸入與補值紀錄不符。");
  return json("custom-single-prediction.json",{schema_version:1,artifact_type:"custom_dataset_single_prediction",scope:"single_row_inference",source:{dataset_sha256:r.dataset_sha256,config_sha256:r.config_sha256},predictor:"linear_regression",target:r.target,features:names,missing_numeric:model.missing_numeric,inputs:names.map(name=>({feature:name,submitted_value:values[name],training_imputation_value:r.imputed.find(row=>row.feature===name)?.value??null})),prediction:r.prediction,out_of_range:r.out_of_range,warnings:r.warnings,limits:["輸入與預測保留原始單位，沿用訓練時的前處理；沒有重新訓練。","submitted_value 為原始輸入；training_imputation_value 僅記錄空白欄位的 Training 補值，不代表真實觀測值。","這筆資料沒有實際目標可評分，不提供單筆 R²／RMSE／MAE 或預測區間。","此檔不含模型或可用於推論的模型代碼，無法匯入網站還原模型。"]});
}
