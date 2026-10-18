import { MISSING_TOKENS, parseCsv } from "./custom-dataset.ts";
import { parseFeatureInput, type PredictionModel } from "./custom-prediction-types.ts";
import { BATCH_LIMITS, parseBatchResult, validateBatchRows, type BatchRows, type BatchResult } from "./custom-batch-types.ts";
import { csvCell, type DownloadArtifact } from "./custom-result-export.ts";
export type BatchPrepared={rows:BatchRows;ignored_columns:string[];input_sha256:string};
export function prepareBatchCsv(text:string,model:PredictionModel):Omit<BatchPrepared,"input_sha256"> {
 if(new TextEncoder().encode(text).byteLength>BATCH_LIMITS.bytes)throw new Error("批次 CSV 超過 2 MiB。");
 const {headers,rows}=parseCsv(text);const missing=model.features.filter(n=>!headers.includes(n));if(missing.length)throw new Error(`缺少模型欄位：${missing.join("、")}。欄名大小寫必須相同。`);
 if(rows.length>BATCH_LIMITS.rows||rows.length*model.features.length>BATCH_LIMITS.cells)throw new Error("批次最多 500 筆、10,000 個模型輸入值，請拆成較小檔案。");
 const indices=model.features.map(n=>headers.indexOf(n));
 const values=rows.map((row,i)=>indices.map((j,k)=>{const s=row[j].trim();try{return MISSING_TOKENS.includes(s)?null:parseFeatureInput(s);}catch{throw new Error(`第 ${i+1} 筆的「${model.features[k]}」不是有效數字。`);}}));
 return {rows:validateBatchRows(values,model),ignored_columns:headers.filter(n=>!model.features.includes(n))};
}
export function batchCsv(model:PredictionModel,value:BatchResult,prepared:BatchPrepared):DownloadArtifact {
 if(!/^[a-f0-9]{64}$/.test(prepared.input_sha256))throw new Error("缺少批次檔案版本。");
 const r=parseBatchResult(value,model,prepared.rows);
 const rows:(string|number|null)[][]=[["source_csv_data_row","target","prediction",...model.features.map(n=>`input:${n}`),"training_imputation_values","out_of_range_features","input_csv_sha256","training_dataset_sha256","config_sha256"],...r.rows.map((row,i)=>[row.position,r.target,row.prediction,...prepared.rows[i],JSON.stringify(r.imputation_values.filter(f=>row.imputed_features.includes(f.feature))),JSON.stringify(row.out_of_range_features),prepared.input_sha256,r.dataset_sha256,r.config_sha256])];
 return {filename:"custom-batch-predictions.csv",mime:"text/csv;charset=utf-8",content:"\uFEFF"+rows.map(row=>row.map(csvCell).join(",")).join("\r\n")+"\r\n"};
}
