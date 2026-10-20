"use client";
import { useCustomCompute } from "./custom-compute-provider";
import { useEffect, useRef, useState } from "react";
import { CustomBatchPrediction } from "./custom-batch-prediction";
import { downloadArtifact } from "@/lib/browser-download";
import { predictionJson } from "@/lib/custom-result-export";
import { parseFeatureInput, parsePredictionResult, type PredictionModel, type PredictionResult } from "@/lib/custom-prediction-types";
const format=(n:number)=>n.toLocaleString("en-US",{maximumFractionDigits:6});
export function CustomPredictionForm({model}:{model:PredictionModel}) {
  const compute=useCustomCompute();
  const [inputs,setInputs]=useState<Record<string,string>>({});const [completed,setCompleted]=useState<{result:PredictionResult;values:Record<string,number|null>}|null>(null);
  const result=completed?.result??null;
  const [busy,setBusy]=useState(false);const [error,setError]=useState<string|null>(null);const [unavailable,setUnavailable]=useState(false);
  const [expiry,setExpiry]=useState("");const controller=useRef<AbortController|null>(null);const generation=useRef(0);
  useEffect(()=>{const timer=setTimeout(()=>setExpiry(new Date(model.expires_at).toLocaleTimeString("zh-TW")),0);return()=>clearTimeout(timer);},[model.expires_at]);
  useEffect(()=>()=>{generation.current++;controller.current?.abort();},[]);
  let invalid:string|null=null;let filled=0;
  try {for(const name of model.features)if(parseFeatureInput(inputs[name]??"")!==null)filled++;}catch(e){invalid=e instanceof Error?e.message:"請輸入有效數字。";}
  const isUnavailable=unavailable||(compute.mode==="browser"&&!compute.hasModel(model.id));
  const complete=model.missing_numeric!=="none"||filled===model.features.length;
  function reset(){generation.current++;controller.current?.abort();controller.current=null;setBusy(false);setCompleted(null);setError(null);}
  async function predict(){
    if(busy||isUnavailable||invalid||!filled||!complete)return;
    const token=++generation.current;const abort=new AbortController();controller.current=abort;setBusy(true);setCompleted(null);setError(null);
    const timer=setTimeout(()=>abort.abort(),15_000);
    try{
      if(Date.now()>=model.expires_at){setUnavailable(true);throw new Error("暫存模型已到期，請重新訓練後再預測。");}
      const values=Object.fromEntries(model.features.map(name=>[name,parseFeatureInput(inputs[name]??"")]));
      const response=await compute.request("/api/custom-dataset/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model_id:model.id,values}),signal:abort.signal,cache:"no-store"});
      const payload=await response.json();
      if(!response.ok){if(response.status===410&&token===generation.current)setUnavailable(true);throw new Error(typeof payload?.error==="string"?payload.error:"暫時無法預測。");}
      const parsed=parsePredictionResult(payload,model);
      if(JSON.stringify(parsed.imputed.map(r=>r.feature))!==JSON.stringify(model.features.filter(n=>values[n]===null)))throw new Error("預測補值與本次輸入不符。");
      if(token===generation.current)setCompleted({result:parsed,values});
    }catch(e){if(token===generation.current)setError(e instanceof Error&&e.name!=="AbortError"?e.message:"預測已取消或等待逾時，請重試。");}
    finally{clearTimeout(timer);if(token===generation.current){setBusy(false);controller.current=null;}}
  }
  async function remove(){
    if(busy||isUnavailable)return;reset();const token=++generation.current;const abort=new AbortController();controller.current=abort;setBusy(true);const timer=setTimeout(()=>abort.abort(),15_000);
    try{const response=await compute.request("/api/custom-dataset/predict",{method:"DELETE",headers:{"Content-Type":"application/json"},body:JSON.stringify({model_id:model.id}),signal:abort.signal,cache:"no-store"});if(!response.ok)throw new Error("暫時無法清除模型，請重試。");if(token===generation.current)setUnavailable(true);}
    catch(e){if(token===generation.current)setError(e instanceof Error&&e.name!=="AbortError"?e.message:"清除等待逾時，請重試。");}
    finally{clearTimeout(timer);if(token===generation.current){setBusy(false);controller.current=null;}}
  }
  return <><section className="cd-single-prediction" aria-labelledby="custom-predict-title" aria-busy={busy}>
    <div className="section-kicker">08 · TRY A NEW INPUT</div><h3 id="custom-predict-title">自訂輸入 · 單筆預測</h3>
    <p>輸入一筆新資料，用剛才完整 Training 訓練好的多元線性迴歸估計 <strong>{model.target}</strong>。按下預測只做推論，不會重新訓練或更新評估成績。</p>
    <p className="cd-confirmation">只需填模型最終使用的 {model.features.length} 個特徵，請使用原始單位，不需自己標準化。{model.missing_numeric==="none"?"訓練時選擇不補值，以下欄位都必填。":`可只填部分欄位；空白會使用完整 Training 學到的${model.missing_numeric==="mean"?"平均值":"中位數"}，至少填一欄。補值會影響結果，已知的欄位請盡量填完整。`}</p>
    {!isUnavailable?<form noValidate onSubmit={e=>{e.preventDefault();void predict();}}>
      <div className="cd-config-grid cd-predict-fields">{model.features.map((name,index)=><label key={name} htmlFor={`custom-predict-${index}`}><span>{name}{model.missing_numeric==="none"?" · 必填":""}</span><input id={`custom-predict-${index}`} type="text" inputMode="decimal" autoComplete="off" maxLength={80} value={inputs[name]??""} placeholder={model.missing_numeric==="none"?"請輸入數值":"留空依 Training 補值"} disabled={busy} onChange={e=>{reset();setInputs({...inputs,[name]:e.target.value});}}/></label>)}</div>
      {invalid&&<p className="cd-error" role="alert">{invalid}</p>}
      {!complete&&filled>0&&<p className="cd-confirmation">目前設定不補值，請填完全部 {model.features.length} 欄。</p>}
      <div className="cd-download-actions"><button id="dataset-predict" type="submit" className="cd-button cd-primary" disabled={busy||!!invalid||!filled||!complete}>{busy?"處理中…":"預測這筆資料"}</button>{busy?<button type="button" className="cd-button" onClick={reset}>取消等待</button>:<button type="button" className="cd-button" onClick={()=>{reset();setInputs({});}}>清空輸入</button>}<button type="button" className="cd-button" disabled={busy} onClick={()=>void remove()}>清除此模型</button></div>
    </form>:<p className="cd-validation" role="status">暫存模型已無法使用；評估成績仍保留。請重新訓練後再做單筆預測。</p>}
    {error&&<p className="cd-error" role="alert">{error}</p>}
    {result&&completed&&<div className="cd-prediction-output" role="status" aria-live="polite"><span>{result.target} · 預測值（原始單位）</span><strong>{format(result.prediction)}</strong><p>這是單筆估計，未提供預測區間；沒有這筆的實際答案，就無法計算它的 R²、RMSE 或 MAE。</p>{result.imputed.length>0&&<p>本次補值：{result.imputed.map(r=>`${r.feature} = ${format(r.value)}`).join("；")}</p>}{result.out_of_range.length>0&&<p>超出 Training 範圍：{result.out_of_range.join("、")}</p>}{result.warnings.length>0&&<ul>{result.warnings.map((w,i)=><li key={i}>{w}</li>)}</ul>}<div className="cd-download-actions"><button id="custom-prediction-json" type="button" className="cd-button" onClick={()=>{try{downloadArtifact(predictionJson(model,result,completed.values));}catch{setError("無法產生單筆預測檔案，請重新預測後再試。");}}}>下載這筆預測 JSON</button></div><p className="cd-confirmation">包含送出當時的原始輸入、實際補值與預測；不含模型。</p></div>}
    <p className="cd-confirmation">模型在 {compute.mode==="browser"?"瀏覽器記憶體":"運算服務私有暫存"}，約 30 分鐘內可用{expiry?`（最晚至 ${expiry}）`:""}，最多保留最近 3 個；可手動清除。{compute.mode==="browser"?"取消、逾時、換檔、改設定、離頁或切換模式會清除記憶體模型，需要重新訓練。":"服務重啟或模型被替換後需重新訓練；伺服器模型依到期時間清除。"}</p>
  </section><CustomBatchPrediction key={`${model.id}-${unavailable}`} model={model} unavailable={isUnavailable} onModelExpired={()=>setUnavailable(true)}/></>;
}
