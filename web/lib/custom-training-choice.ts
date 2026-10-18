import { isAutoMethod, isNestedMethod, nestedSelectionError } from "./custom-selection-policy.ts";
import type { PreprocessingPlan } from "./custom-dataset.ts";
import type { ValidationReport } from "./custom-validation-result.ts";
import type { SelectionMethod } from "./custom-training-result.ts";
type Validated=Extract<ValidationReport,{status:"valid"}>;
export function trainingSelectionError(method:SelectionMethod,k:number|null,config:PreprocessingPlan,validation:Validated):string|null {
  if(!["none","f_test","mutual_information","rfe","forward_sfs","rfecv","lasso_cv"].includes(method))return "未知特徵選擇方法。";
  const p=config.feature_columns.length;
  if(method==="none")return k===null?null:"不選欄時不可指定 k。";
  if(isAutoMethod(method)&&k!==null)return "自動選 k 時，不可指定固定 k。";
  if(isNestedMethod(method)&&config.numerical_scaling!=="standard")return "SFS／RFECV／LassoCV 需要 StandardScaler；請調整設定、確認並重新驗證。";
  if(isAutoMethod(method))return nestedSelectionError(method,k,p,validation.dataset.rows,[...validation.split.cv_folds.map(f=>f.training_rows),validation.split.training_rows],config.split_strategy,config.cv_gap);
  if(typeof k!=="number"||!Number.isSafeInteger(k)||k<1||k>p)return `k 必須是 1 到 ${p} 之間的整數。`;
  if(k<p&&validation.split.cv_folds.some(f=>f.training_rows<4))return "選特徵需要每個 CV Training 至少 4 筆，請調整切分。";
  if(method==="rfe"){
    if(config.numerical_scaling!=="standard")return "RFE 需要 StandardScaler；請調整設定、確認並重新驗證。";
    const rows=validation.split.training_rows+validation.split.cv_folds.reduce((sum,f)=>sum+f.training_rows,0),fits=k<p?p-k+2:1;
    if(p>25||validation.dataset.rows>5000||fits*p*rows>100_000_000)return "RFE 限 5,000 筆、25 個候選欄位及有限運算量；請減少資料／特徵／folds 或增加 k。";
  }
  return nestedSelectionError(method,k,p,validation.dataset.rows,[...validation.split.cv_folds.map(f=>f.training_rows),validation.split.training_rows],config.split_strategy,config.cv_gap);
}
