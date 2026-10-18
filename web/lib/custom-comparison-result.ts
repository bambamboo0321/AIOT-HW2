import { parseValidationReport, type ValidationReport } from "./custom-validation-result.ts";
import { parseScores, parseAverage, type Scores, type Average } from "./custom-training-result.ts";
import { isAutoMethod, isNestedMethod, parseSelectionDetails, type SelectionDetails, type AutoMethod } from "./custom-selection-policy.ts";
export const COMPARISON_LABELS = { f_test:"F-test", mutual_information:"Mutual Information", rfe:"RFE", forward_sfs:"Forward SFS", rfecv:"RFECV", lasso_cv:"LassoCV" } as const;
export type ComparisonMethod=keyof typeof COMPARISON_LABELS;
export const COMPARISON_METHODS:ComparisonMethod[]=["f_test","mutual_information","rfe","forward_sfs","rfecv","lasso_cv"];
type Fold={fold:number;training_rows:number;validation_rows:number;features:string[];scores:Scores;details?:SelectionDetails};
export type ComparisonPoint={method:ComparisonMethod;k:number;ranking_performed:boolean;folds:Fold[];metrics:Average};
export type AutomaticComparison={method:AutoMethod;folds:(Omit<Fold,"scores">&{scores:Scores|null;details:SelectionDetails})[];metrics:Average};
export type ComparisonReport={schema_version:1;scope:"custom_feature_comparison"}&(
  {status:"rejected"|"failed";comparison_started:boolean;errors:{stage:string;message:string}[]}|
  {status:"compared";comparison_started:true;validation:Extract<ValidationReport,{status:"valid"}>;target:string;features:string[];methods:ComparisonMethod[];k_values:number[];scaling:"standard";points:ComparisonPoint[];automatic:AutomaticComparison[];warnings:string[];limits:string[]}
);
export type Compared=Extract<ComparisonReport,{status:"compared"}>;
function obj(v:unknown):Record<string,unknown>{if(!v||typeof v!=="object"||Array.isArray(v))throw new Error("比較格式不符。");return v as Record<string,unknown>;}
function text(v:unknown,max=4000):string{if(typeof v!=="string"||!v||v.length>max)throw new Error("比較訊息不合法。");return v;}
function list(v:unknown,max:number):unknown[]{if(!Array.isArray(v)||v.length>max)throw new Error("比較清單不合法。");return v;}
function names(v:unknown,max:number):string[]{return list(v,max).map(n=>text(n,128));}
export function parseComparisonReport(value:unknown):ComparisonReport {
  const v=obj(value);if(v.schema_version!==1||v.scope!=="custom_feature_comparison")throw new Error("比較版本或範圍不符。");const base={schema_version:1 as const,scope:"custom_feature_comparison" as const};
  if(v.status==="rejected"||v.status==="failed"){const errors=list(v.errors,20);if(!errors.length||v.comparison_started!==(v.status==="failed"))throw new Error("比較錯誤狀態不符。");return {...base,status:v.status,comparison_started:v.comparison_started,errors:errors.map(e=>{const r=obj(e),stage=text(r.stage);if(!["confirmation","config","csv","data","review","split","comparison"].includes(stage))throw new Error("比較錯誤階段不符。");return {stage,message:text(r.message)};})};}
  if(v.status!=="compared"||v.comparison_started!==true||v.scaling!=="standard")throw new Error("比較未完成或缺少標準化。");
  const validation=parseValidationReport(v.validation);if(validation.status!=="valid"||!validation.review_checked||validation.dataset.rows>5000)throw new Error("比較缺少有效驗證。");
  const features=names(v.features,25),target=text(v.target,128);if(!features.length||new Set(features).size!==features.length||features.includes(target)||features.length>=validation.dataset.columns||validation.split.cv_folds.some(f=>f.training_rows<4))throw new Error("比較欄位或 fold 不符。");
  const methods=names(v.methods,6) as ComparisonMethod[];if(!methods.length||JSON.stringify(methods)!==JSON.stringify(COMPARISON_METHODS.filter(m=>methods.includes(m))))throw new Error("比較方法不符。");
  const fixed=methods.filter(m=>!isAutoMethod(m)),autos=methods.filter(isAutoMethod);
  const ks=list(v.k_values,25).map(k=>{if(typeof k!=="number"||!Number.isInteger(k)||k<1||k>features.length)throw new Error("比較 k 不合法。");return k;});
  if(!!fixed.length!==!!ks.length||ks.some((k,i)=>i>0&&k<=ks[i-1]))throw new Error("固定 k 範圍不符。");
  function fold(value:unknown,i:number,k:number|null,method:ComparisonMethod,auto=false){const f=obj(value),expected=validation.status==="valid"?validation.split.cv_folds[i]:undefined;if(!expected)throw new Error("Fold 不符。");const selected=names(f.features,features.length);if(f.fold!==expected.fold||f.training_rows!==expected.training_rows||f.validation_rows!==expected.validation_rows||new Set(selected).size!==selected.length||JSON.stringify(selected)!==JSON.stringify(features.filter(n=>selected.includes(n)))||(!auto&&selected.length!==k)||(auto&&method==="rfecv"&&selected.length<1))throw new Error("比較切分或選中特徵不符。");const details=isNestedMethod(method)?parseSelectionDetails(f.details,method,expected.training_rows,features.length,k):undefined;return {...expected,features:selected,scores:auto&&selected.length===0?(f.scores===null?null:(()=>{throw new Error("零特徵不可計分。");})()):parseScores(f.scores),...(details?{details}:{})};}
  const raw=list(v.points,100);if(raw.length!==fixed.length*ks.length)throw new Error("固定比較點缺少。");
  const points=raw.map((row,i):ComparisonPoint=>{const p=obj(row),method=fixed[Math.floor(i/ks.length)],k=ks[i%ks.length];if(p.method!==method||p.k!==k||p.ranking_performed!==(k<features.length))throw new Error("比較點方法或 k 不符。");const rows=list(p.folds,10);if(rows.length!==validation.split.cv_folds.length)throw new Error("比較 folds 缺少。");const folds=rows.map((f,j)=>fold(f,j,k,method)) as Fold[];return {method,k,ranking_performed:k<features.length,folds,metrics:parseAverage(p.metrics,folds.map(f=>f.scores))};});
  const autoRaw=list(v.automatic??[],2);if(autoRaw.length!==autos.length)throw new Error("自動選 k 結果缺少。");
  const automatic=autoRaw.map((row,i):AutomaticComparison=>{const p=obj(row),method=autos[i];if(p.method!==method)throw new Error("自動選 k 方法不符。");const rows=list(p.folds,10);if(rows.length!==validation.split.cv_folds.length)throw new Error("自動選 k folds 缺少。");const folds=rows.map((f,j)=>fold(f,j,null,method,true)) as AutomaticComparison['folds'];return {method,folds,metrics:parseAverage(p.metrics,folds.map(f=>f.scores??{rmse:null,mae:null,r2:null}))};});
  return {...base,status:"compared",comparison_started:true,validation,target,features,methods,k_values:ks,scaling:"standard",points,automatic,warnings:list(v.warnings,250).map(v=>text(v)),limits:list(v.limits,20).map(v=>text(v))};
}
