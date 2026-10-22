/** Fixed research inference only. No fitting, network, Worker or Node dependency. */
export const FIXED_DATASETS = ["appliances-energy", "automobile-price", "student-performance"] as const;
export const FIXED_MODELS = ["baseline", "mutual_information"] as const;
export type FixedDatasetId = typeof FIXED_DATASETS[number];
export type FixedModelId = typeof FIXED_MODELS[number];
export type ModelBinding = { datasetId: FixedDatasetId; modelId: FixedModelId; modelVersion: "1.0.0" };
export type FixedFeature = {
 name:string; kind:"number"|"integer"|"category"; unit:string; description:string;
 trainingMin:number; trainingMax:number; domainMin:number|null; domainMax:number|null;
 allowedValues:number[]|null; encoding:Record<string,number>|null;
 imputer:{strategy:"median";value:number}|null;
 scaler:{mean:number;scale:number;withMean:true;withStd:true}|null; outliers:"keep";
};
export type FixedModel = ModelBinding & {
 schemaVersion:1;target:string;unit:string;features:FixedFeature[];featureOrder:string[];
 originalPreprocessorSchema:string[];coefficients:number[];intercept:number;
 preprocessingOrder:["validate","encode","impute","scale","linear"];outlierStrategy:"keep";predictionClipped:false;
 trainingSplit:{strategy:string;randomSeed:number|null;trainingRows:number;holdoutRows:number;trainingIndexSha256:string;holdoutIndexSha256:string;holdoutUsedForFitting:false};
 versions:Record<string,string>;sourceHashes:{path:string;sha256:string}[];
 acquisition:"original-saved-model-no-fit"|"training-only-refit-with-frozen-selected-columns-no-selector";limitations:string[];
};
export type FixedPrediction = ModelBinding & {target:string;unit:string;prediction:number;imputed:{feature:string;value:number}[];outOfRange:string[];warnings:string[]};
const object=(x:unknown):x is Record<string,unknown>=>!!x&&typeof x==="object"&&!Array.isArray(x);
const finite=(x:unknown):x is number=>typeof x==="number"&&Number.isFinite(x);
const text=(x:unknown):x is string=>typeof x==="string"&&x.length>0;
const hash=(x:unknown):x is string=>typeof x==="string"&&/^[a-f0-9]{64}$/.test(x);
const names=(x:unknown):x is string[]=>Array.isArray(x)&&x.length>0&&x.every(n=>text(n)&&!["__proto__","prototype","constructor"].includes(n))&&new Set(x).size===x.length;
function valid(x:unknown,field:string):asserts x {if(!x)throw new Error(`固定模型不相容或損壞：${field}`);}
const contract={
 "appliances-energy":{target:"Appliances",train:15788,holdout:3947,baseline:25,mutual_information:6},
 "automobile-price":{target:"price",train:160,holdout:41,baseline:14,mutual_information:7},
 "student-performance":{target:"G3",train:519,holdout:130,baseline:15,mutual_information:7},
} as const;
function freezeDeep<T>(v:T):T {if(v&&typeof v==="object"){for(const child of Object.values(v))freezeDeep(child);Object.freeze(v);}return v;}
/** A detached frozen copy; parsing does not mutate the caller's model JSON. */
export function parseFixedModel(raw:unknown,binding:ModelBinding):FixedModel {
 valid(binding.modelVersion==="1.0.0"&&FIXED_DATASETS.includes(binding.datasetId)&&FIXED_MODELS.includes(binding.modelId),"expected binding");
 valid(object(raw)&&raw.schemaVersion===1&&raw.modelVersion===binding.modelVersion&&raw.datasetId===binding.datasetId&&raw.modelId===binding.modelId,"version / dataset / model ID");
 const c=contract[binding.datasetId];
 valid(raw.target===c.target&&text(raw.unit)&&names(raw.featureOrder)&&names(raw.originalPreprocessorSchema)&&!raw.featureOrder.includes(c.target)&&raw.featureOrder.every(n=>(raw.originalPreprocessorSchema as string[]).includes(n)),"input schema / target");
 valid(raw.featureOrder.length===c[binding.modelId]&&Array.isArray(raw.features)&&raw.features.length===raw.featureOrder.length&&Array.isArray(raw.coefficients)&&raw.coefficients.length===raw.featureOrder.length&&raw.coefficients.every(finite)&&finite(raw.intercept),"coefficients / order");
 valid(JSON.stringify(raw.preprocessingOrder)===JSON.stringify(["validate","encode","impute","scale","linear"])&&raw.outlierStrategy==="keep"&&raw.predictionClipped===false,"pipeline / clipping");
 const s=raw.trainingSplit;
 valid(object(s)&&s.trainingRows===c.train&&s.holdoutRows===c.holdout&&hash(s.trainingIndexSha256)&&hash(s.holdoutIndexSha256)&&s.holdoutUsedForFitting===false&&text(s.strategy)&&s.randomSeed===(binding.datasetId==="appliances-energy"?null:42),"Training split");
 valid(raw.acquisition===(binding.datasetId==="appliances-energy"?"original-saved-model-no-fit":"training-only-refit-with-frozen-selected-columns-no-selector")&&object(raw.versions)&&raw.versions["scikit-learn"]==="1.7.2"&&Object.values(raw.versions).every(text),"origin / package versions");
 valid(Array.isArray(raw.sourceHashes)&&raw.sourceHashes.length>0&&raw.sourceHashes.every(s=>object(s)&&text(s.path)&&!s.path.startsWith("/")&&!s.path.includes("..")&&hash(s.sha256))&&new Set(raw.sourceHashes.map(s=>s.path)).size===raw.sourceHashes.length,"source hashes");
 valid(Array.isArray(raw.limitations)&&raw.limitations.length>0&&raw.limitations.every(text),"limitations");
 for(const [i,f] of raw.features.entries()){
  valid(object(f)&&f.name===raw.featureOrder[i]&&["number","integer","category"].includes(String(f.kind))&&text(f.unit)&&text(f.description)&&finite(f.trainingMin)&&finite(f.trainingMax)&&f.trainingMin<=f.trainingMax&&f.outliers==="keep","feature metadata");
  valid((f.domainMin===null||finite(f.domainMin))&&(f.domainMax===null||finite(f.domainMax))&&(f.domainMin===null||f.domainMax===null||(f.domainMin as number)<=(f.domainMax as number)),"legal domain");
  valid(f.allowedValues===null||(Array.isArray(f.allowedValues)&&f.allowedValues.length>0&&f.allowedValues.every(finite)&&new Set(f.allowedValues).size===f.allowedValues.length),"legal codes");
  valid(f.encoding===null?f.kind!=="category":f.kind==="category"&&object(f.encoding)&&Object.keys(f.encoding).length===2&&Object.keys(f.encoding).every(text)&&Object.values(f.encoding).every(finite)&&new Set(Object.values(f.encoding)).size===2,"category encoding");
  valid(f.imputer===null||(object(f.imputer)&&f.imputer.strategy==="median"&&finite(f.imputer.value)),"imputer");
  valid(f.scaler===null||(object(f.scaler)&&finite(f.scaler.mean)&&finite(f.scaler.scale)&&f.scaler.scale>0&&f.scaler.withMean===true&&f.scaler.withStd===true),"scaler");
  if(binding.datasetId==="appliances-energy"&&binding.modelId==="baseline")valid(f.imputer===null&&f.scaler===null,"raw Energy Baseline");
  else valid(f.scaler!==null&&(f.kind==="category"?f.imputer===null:f.imputer!==null),"frozen preprocessing required");
 }
 return freezeDeep(structuredClone(raw) as FixedModel);
}
/** Verify exact bytes against a trusted bundled manifest before parsing. */
export async function verifyFixedModelJson(text:string,binding:ModelBinding,expectedSha256:string):Promise<FixedModel>{
 valid(hash(expectedSha256)&&new TextEncoder().encode(text).byteLength<=128*1024,"JSON size / expected SHA-256");
 const actual=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(text))),b=>b.toString(16).padStart(2,"0")).join("");
 valid(actual===expectedSha256,"SHA-256 mismatch");return parseFixedModel(JSON.parse(text),binding);
}
/** Pure inference: no input/model mutation, fitting, clipping or automatic transport. */
export function predictFixedModel(model:FixedModel,inputs:unknown):FixedPrediction {
 if(!object(inputs)||Object.keys(inputs).some(n=>!model.featureOrder.includes(n)))throw new Error("輸入必須只包含目前模型使用的欄位。");
 const imputed:FixedPrediction["imputed"]=[],outside:string[]=[];let linear=0;
 for(const [i,f] of model.features.entries()){
  const raw=Object.hasOwn(inputs,f.name)?inputs[f.name]:null;let value:number;
  if(raw===null){if(!f.imputer)throw new Error(`${f.name} 必填；此欄不接受缺失值。`);value=f.imputer.value;imputed.push({feature:f.name,value});}
  else{
   if(f.encoding){if(typeof raw==="string"&&Object.hasOwn(f.encoding,raw))value=f.encoding[raw];else if(finite(raw)&&Object.values(f.encoding).includes(raw))value=raw;else throw new Error(`${f.name}：未知類別或非法編碼。`);}
   else{if(!finite(raw)||Math.abs(raw)>Number.MAX_SAFE_INTEGER)throw new Error(`${f.name} 必須是安全範圍內的有限數字。`);value=raw;}
   if((f.kind==="integer"&&!Number.isSafeInteger(value))||(f.allowedValues&&!f.allowedValues.includes(value))||(f.domainMin!==null&&value<f.domainMin)||(f.domainMax!==null&&value>f.domainMax))throw new Error(`${f.name} 超出合法值或整數／序數規則。`);
   if(value<f.trainingMin||value>f.trainingMax)outside.push(f.name);
  }
  // Frozen keep-outliers -> imputer -> StandardScaler -> LinearRegression.
  const z=f.scaler?(value-f.scaler.mean)/f.scaler.scale:value;
  if(!Number.isFinite(z))throw new Error(`${f.name} 前處理無法產生有限數值。`);
  linear+=model.coefficients[i]*z;
 }
 const prediction=linear+model.intercept;
 if(!Number.isFinite(prediction))throw new Error("線性推論無法產生有限數值。");
 return {datasetId:model.datasetId,modelId:model.modelId,modelVersion:model.modelVersion,target:model.target,unit:model.unit,prediction,imputed,outOfRange:outside,
  warnings:[...model.limitations,...(imputed.length?["空白使用保存的 Training median；不代表已知該筆真實輸入。"]:[]),...(outside.length?[`外插警告：${outside.join("、")} 超出 Training 範圍，保留原始輸入與預測。`]:[]),...(model.datasetId==="student-performance"&&(prediction<0||prediction>20)?["預測超出 G3 的 0–20 分範圍，未裁切。"]:[]),...(model.datasetId!=="student-performance"&&prediction<0?["預測為負值，未裁切；請依目標意義判讀。"]:[])]};
}
