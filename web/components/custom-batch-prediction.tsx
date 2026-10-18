"use client";
import { useEffect, useRef, useState } from "react";
import { decodeCsv } from "@/lib/custom-dataset";
import { BATCH_LIMITS, parseBatchResult, type BatchResult } from "@/lib/custom-batch-types";
import { batchCsv, prepareBatchCsv, type BatchPrepared } from "@/lib/custom-batch-input";
import { downloadArtifact } from "@/lib/browser-download";
import type { PredictionModel } from "@/lib/custom-prediction-types";
export function CustomBatchPrediction({model,unavailable,onModelExpired}:{model:PredictionModel;unavailable:boolean;onModelExpired:()=>void}) {
 const [prepared,setPrepared]=useState<BatchPrepared|null>(null);const [filename,setFilename]=useState("");const [busy,setBusy]=useState(false);const [reading,setReading]=useState(false);const [error,setError]=useState<string|null>(null);const [result,setResult]=useState<BatchResult|null>(null);const [notice,setNotice]=useState<string|null>(null);
 const generation=useRef(0);const controller=useRef<AbortController|null>(null);
 useEffect(()=>()=>{generation.current++;controller.current?.abort();},[]);
 function reset(){generation.current++;controller.current?.abort();controller.current=null;setBusy(false);setReading(false);setPrepared(null);setFilename("");setError(null);setResult(null);setNotice(null);}
 async function open(file:File|undefined){
  reset();if(!file)return;const token=generation.current;setReading(true);setFilename(file.name);
  try{
   if(!file.name.toLowerCase().endsWith(".csv")||!file.size||file.size>BATCH_LIMITS.bytes)throw new Error("請選擇非空的 UTF-8 CSV，上限 2 MiB。");
   const buffer=await file.arrayBuffer();if(token!==generation.current)return;
   const data=prepareBatchCsv(decodeCsv(buffer),model);const sha=await crypto.subtle.digest("SHA-256",buffer);const hash=[...new Uint8Array(sha)].map(n=>n.toString(16).padStart(2,"0")).join("");
   if(token===generation.current)setPrepared({...data,input_sha256:hash});
  }catch(e){if(token===generation.current)setError(e instanceof Error?e.message:"無法檢查批次 CSV。");}
  finally{if(token===generation.current)setReading(false);}
 }
 function cancel(){generation.current++;controller.current?.abort();controller.current=null;setBusy(false);setResult(null);setError("已取消批次預測，不顯示未完成結果。");}
 async function predict(){
  if(!prepared||busy||reading||unavailable)return;const token=++generation.current;const abort=new AbortController();controller.current=abort;setBusy(true);setResult(null);setError(null);setNotice(null);const timer=setTimeout(()=>abort.abort(),35_000);
  try{
   if(Date.now()>=model.expires_at){onModelExpired();return;}
   const response=await fetch("/api/custom-dataset/predict-batch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model_id:model.id,features:model.features,rows:prepared.rows}),cache:"no-store",signal:abort.signal});const payload=await response.json();
   if(!response.ok){if(response.status===410&&token===generation.current)onModelExpired();throw new Error(typeof payload?.error==="string"?payload.error:"批次預測暫時無法完成。");}
   const parsed=parseBatchResult(payload,model,prepared.rows);if(token===generation.current)setResult(parsed);
  }catch(e){if(token===generation.current)setError(e instanceof Error&&e.name!=="AbortError"?e.message:"批次等待已取消或逾時，請重試。");}
  finally{clearTimeout(timer);if(token===generation.current){setBusy(false);controller.current=null;}}
 }
 const maxRows=Math.min(BATCH_LIMITS.rows,Math.floor(BATCH_LIMITS.cells/model.features.length));
 const imputedRows=result?.rows.filter(r=>r.imputed_features.length).length??0;const outsideRows=result?.rows.filter(r=>r.out_of_range_features.length).length??0;
 return <section className="cd-batch-prediction" aria-labelledby="custom-batch-title" aria-busy={busy||reading}><div className="section-kicker">09 · PREDICT A NEW FILE</div><h3 id="custom-batch-title">新 CSV · 批次預測</h3><p>用同一個已訓練模型一次預測多筆「{model.target}」，不重新訓練。新檔不需要目標欄位；本模型單次最多 {maxRows} 筆、2 MiB（整體上限 500 筆、10,000 個模型輸入值）。</p>
 {unavailable?<p className="cd-validation" role="status">模型已清除或到期，請重新訓練後再使用批次預測。</p>:<><p className="cd-confirmation">CSV 必須包含全部 {model.features.length} 個模型欄名（大小寫相同），順序可不同。{model.missing_numeric==="none"?"此模型不補值，每筆都須填完整。":"欄位內空白或 NA 記號沿用 Training 補值；每筆至少填一個數值。"}多出的欄位會列出並忽略；即使有目標值，也不會拿來評分。</p><details><summary>查看模型需要的欄名</summary><p>{model.features.join("、")}</p></details>
 <label className="cd-file-button cd-batch-file" aria-disabled={busy} htmlFor="custom-batch-file">{filename?"更換要預測的 CSV":"選擇要預測的 CSV"}<input id="custom-batch-file" type="file" aria-label="選擇批次預測 CSV" accept=".csv,text/csv" disabled={busy} onChange={e=>{const file=e.currentTarget.files?.[0];e.currentTarget.value="";void open(file);}}/></label>
 {reading&&<p role="status">正在瀏覽器檢查檔案…</p>}{filename&&<p className="cd-confirmation">檔案：{filename}</p>}
 {prepared&&<div className="cd-batch-ready"><strong>檢查通過 · {prepared.rows.length} 筆 · {model.features.length} 個模型輸入</strong>{prepared.ignored_columns.length>0&&<p>忽略欄位：{prepared.ignored_columns.join("、")}</p>}<p>目前只在瀏覽器檢查；按下「開始批次預測」才傳送模型需要的數值至本機服務。若有一筆資料無效，整批先停止，不會偷偷跳列。</p></div>}
 <div className="cd-download-actions"><button id="custom-batch-predict" className="cd-button cd-primary" disabled={!prepared||busy||reading} onClick={()=>void predict()}>{busy?"批次預測中…":"開始批次預測"}</button>{busy?<button className="cd-button" onClick={cancel}>取消批次預測</button>:<button className="cd-button" disabled={!prepared&&!filename} onClick={reset}>清除批次資料</button>}</div></>}
 {error&&<p className="cd-error" role="alert">{error}</p>}
 {result&&prepared&&<div className="cd-batch-result" role="status"><h4>批次預測完成 · {result.row_count} 筆</h4><p>{imputedRows} 筆使用補值 · {outsideRows} 筆含超出 Training 範圍的輸入。這是新資料的估計，沒有 R²／RMSE／MAE 或預測區間。</p>{result.warnings.length>0&&<ul>{result.warnings.map((w,i)=><li key={i}>{w}</li>)}</ul>}<div className="fs-table-scroll"><table className="metrics-table"><caption>前 {Math.min(10,result.row_count)} 筆預覽；下載包含本批全部 {result.row_count} 筆</caption><thead><tr><th>CSV 資料列</th><th>預測 {result.target}</th><th>補值欄位</th><th>超出範圍</th></tr></thead><tbody>{result.rows.slice(0,10).map(row=><tr key={row.position}><th>{row.position}</th><td>{row.prediction.toLocaleString("en-US",{maximumFractionDigits:6})}</td><td>{row.imputed_features.join("、")||"—"}</td><td>{row.out_of_range_features.join("、")||"—"}</td></tr>)}</tbody></table></div><button id="custom-batch-download" className="cd-button" onClick={()=>{try{downloadArtifact(batchCsv(model,result,prepared));setNotice("已產生本批全部預測的 CSV，請查看下載項目。");}catch{setError("無法產生批次結果，請重新預測後再試。");}}}>下載全部 {result.row_count} 筆預測 CSV</button>{notice&&<p role="status">{notice}</p>}<p className="cd-confirmation">下載包含模型使用的原始輸入（空白仍為空白）、實際 Training 補值、預測與超出範圍欄位；不含被忽略欄位或模型代碼。資料列從 1 起算，不含標頭與略過的完全空白行，並非檔案的實體行號。</p></div>}
 <p className="cd-confirmation">批次與單筆共用同一個約 30 分鐘到期的模型。換檔、清除資料、改訓練設定或離頁會清除舊結果。伺服器只暫存這次需要的數值，運算結束後移除；不會保存新 CSV。</p>
 </section>;
}
