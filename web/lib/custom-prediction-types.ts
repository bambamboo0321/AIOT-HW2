export type PredictionModel = { id: string; expires_at: number; target: string; features: string[]; missing_numeric: "mean" | "median" | "none"; dataset_sha256: string; config_sha256: string };
export type PredictionResult = { schema_version: 1; status: "predicted"; training_started: false; target: string; features: string[]; prediction: number; dataset_sha256: string; config_sha256: string; imputed: { feature: string; value: number }[]; out_of_range: string[]; warnings: string[] };
function object(v: unknown): Record<string,unknown> { if (!v || typeof v!=="object" || Array.isArray(v)) throw new Error("預測結果格式不符。");return v as Record<string,unknown>; }
function names(v: unknown): string[] { if (!Array.isArray(v) || !v.length || v.length>199 || v.some(n=>typeof n!=="string" || !n || n.length>128) || new Set(v).size!==v.length) throw new Error("模型欄位格式不符。"); return v; }
const digest=(v:unknown)=>typeof v==="string"&&/^[a-f0-9]{64}$/.test(v);
export function parsePredictionModel(value: unknown): PredictionModel {
  const v=object(value); const features=names(v.features);
  if (!digest(v.id) || !digest(v.dataset_sha256) || !digest(v.config_sha256) || typeof v.target!=="string" || !v.target || v.target.length>128 || features.includes(v.target) || typeof v.expires_at!=="number" || !Number.isSafeInteger(v.expires_at) || v.expires_at<=0 || !["mean","median","none"].includes(String(v.missing_numeric))) throw new Error("暫存模型資訊不符。");
  return {id:v.id as string,expires_at:v.expires_at,target:v.target,features,missing_numeric:v.missing_numeric as PredictionModel["missing_numeric"],dataset_sha256:v.dataset_sha256 as string,config_sha256:v.config_sha256 as string};
}
export function parsePredictionResult(value: unknown, model: PredictionModel): PredictionResult {
  const v=object(value); const features=names(v.features);
  if (v.schema_version!==1 || v.status!=="predicted" || v.training_started!==false || v.target!==model.target || v.dataset_sha256!==model.dataset_sha256 || v.config_sha256!==model.config_sha256 || JSON.stringify(features)!==JSON.stringify(model.features) || typeof v.prediction!=="number" || !Number.isFinite(v.prediction)) throw new Error("預測結果與目前模型不一致。");
  if (!Array.isArray(v.imputed) || v.imputed.length>features.length) throw new Error("補值結果格式不符。");
  const imputed=v.imputed.map(row=>{const r=object(row);if(typeof r.feature!=="string" || !features.includes(r.feature) || typeof r.value!=="number" || !Number.isFinite(r.value))throw new Error("補值結果格式不符。");return {feature:r.feature,value:r.value};});
  if (new Set(imputed.map(r=>r.feature)).size!==imputed.length || (model.missing_numeric==="none"&&imputed.length)) throw new Error("補值狀態不符。");
  if (!Array.isArray(v.out_of_range) || v.out_of_range.some(n=>typeof n!=="string" || !features.includes(n)) || new Set(v.out_of_range).size!==v.out_of_range.length || !Array.isArray(v.warnings) || v.warnings.length>20 || v.warnings.some(n=>typeof n!=="string"||n.length>4000)) throw new Error("預測提醒格式不符。");
  return {schema_version:1,status:"predicted",training_started:false,target:model.target,features,prediction:v.prediction,dataset_sha256:model.dataset_sha256,config_sha256:model.config_sha256,imputed,out_of_range:v.out_of_range,warnings:v.warnings};
}
export function parseFeatureInput(text: string): number | null {
  const value=text.trim(); if (!value) return null;
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value)) throw new Error("請輸入數字；空白才會依訓練設定補值。");
  const n=Number(value);if(!Number.isFinite(n)||Math.abs(n)>Number.MAX_SAFE_INTEGER)throw new Error("數值必須有限，且絕對值不超過 9,007,199,254,740,991。");return n;
}

export type TrainingInputRanges = {source:"full_training_raw";training_rows:number;features:{feature:string;min:number|null;max:number|null}[]};
export function parseTrainingInputRanges(value:unknown, features:string[], rows:number):TrainingInputRanges {
 const v=object(value);
 if(v.source!=="full_training_raw"||v.training_rows!==rows||!Array.isArray(v.features)||v.features.length!==features.length)throw new Error("Training 輸入範圍來源不符。");
 const ranges=v.features.map((raw,i)=>{const r=object(raw);if(r.feature!==features[i]||!((r.min===null&&r.max===null)||(typeof r.min==="number"&&typeof r.max==="number"&&Number.isFinite(r.min)&&Number.isFinite(r.max)&&r.min<=r.max&&Math.max(Math.abs(r.min),Math.abs(r.max))<=Number.MAX_SAFE_INTEGER)))throw new Error("Training 輸入範圍不符。");return {feature:features[i],min:r.min as number|null,max:r.max as number|null};});
 return {source:"full_training_raw",training_rows:rows,features:ranges};
}
