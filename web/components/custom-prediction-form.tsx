"use client";
import { useCustomCompute } from "./custom-compute-provider";
import { useEffect, useRef, useState } from "react";
import { CustomBatchPrediction } from "./custom-batch-prediction";
import { downloadArtifact } from "@/lib/browser-download";
import { predictionJson } from "@/lib/custom-result-export";
import { parseFeatureInput, parsePredictionResult, type PredictionModel, type PredictionResult, type TrainingInputRanges } from "@/lib/custom-prediction-types";
import {initialPredictionInputs,sliderPresentation} from "@/lib/live-prediction-input";
import {LivePredictionController,type PredictionValues} from "@/lib/live-prediction-controller";
const format=(n:number)=>n.toLocaleString("en-US",{maximumFractionDigits:6});
export function CustomPredictionForm({model,ranges,paused=false}:{model:PredictionModel;ranges?:TrainingInputRanges;paused?:boolean}) {
  const compute=useCustomCompute(),request=compute.request;
  const [inputs,setInputs]=useState<Record<string,string>>(()=>initialPredictionInputs(ranges));
  const [completed,setCompleted]=useState<{result:PredictionResult;values:PredictionValues}|null>(null);
  const [busy,setBusy]=useState(false),[deleting,setDeleting]=useState(false),[batchBusy,setBatchBusy]=useState(false),[live,setLive]=useState(true);
  const [error,setError]=useState<string|null>(null),[unavailable,setUnavailable]=useState(false),[expiry,setExpiry]=useState("");
  const runner=useRef<LivePredictionController<PredictionResult>|null>(null),deleteController=useRef<AbortController|null>(null);
  const result=completed?.result??null;
  useEffect(()=>{
    const own=new LivePredictionController<PredictionResult>({
      request:async(values,signal)=>{
        if(Date.now()>=model.expires_at)throw Object.assign(new Error("暫存模型已到期，請重新訓練。"),{expired:true});
        const response=await request("/api/custom-dataset/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model_id:model.id,values}),signal,cache:"no-store"});
        const payload=await response.json();
        if(!response.ok)throw Object.assign(new Error(typeof payload?.error==="string"?payload.error:"暫時無法預測，請手動重試。"),{expired:response.status===410});
        const parsed=parsePredictionResult(payload,model);
        if(JSON.stringify(parsed.imputed.map(r=>r.feature))!==JSON.stringify(model.features.filter(n=>values[n]===null)))throw new Error("預測補值與本次輸入不符。");
        return parsed;
      },
      onStart:()=>{setBusy(true);setCompleted(null);setError(null);},
      onResult:(result,values)=>setCompleted({result,values}),
      onError:failure=>{if(failure&&typeof failure==="object"&&"expired" in failure&&failure.expired)setUnavailable(true);setError(failure instanceof Error&&failure.name!=="AbortError"?failure.message:"預測已取消或等待逾時，請重試。");},
      onIdle:()=>setBusy(false),
    });runner.current=own;
    const display=setTimeout(()=>setExpiry(new Date(model.expires_at).toLocaleTimeString("zh-TW")),0);
    const expiryTimer=setTimeout(()=>{setUnavailable(true);own.cancel();},Math.max(0,model.expires_at-Date.now()));
    return()=>{clearTimeout(display);clearTimeout(expiryTimer);own.dispose();deleteController.current?.abort();if(runner.current===own)runner.current=null;};
  },[model,request]);
  let invalid:string|null=null;let filled=0;let values:PredictionValues={};
  try{values=Object.fromEntries(model.features.map(name=>[name,parseFeatureInput(inputs[name]??"")]));filled=Object.values(values).filter(v=>v!==null).length;}catch(e){invalid=e instanceof Error?e.message:"請輸入有效數字。";}
  const isUnavailable=unavailable||(compute.mode==="browser"&&!compute.hasModel(model.id));
  const complete=model.missing_numeric!=="none"||filled===model.features.length;
  useEffect(()=>{
    if(!live||paused||batchBusy||deleting||isUnavailable){runner.current?.schedule(null);return;}
    try{const next=Object.fromEntries(model.features.map(n=>[n,parseFeatureInput(inputs[n]??"")]));const filled=Object.values(next).filter(v=>v!==null).length;runner.current?.schedule(filled&&(model.missing_numeric!=="none"||filled===model.features.length)?next:null);}catch{runner.current?.schedule(null);}
  },[inputs,live,paused,batchBusy,deleting,isUnavailable,model]);
  function edit(name:string,value:string){runner.current?.schedule(null);setCompleted(null);setError(null);setInputs(v=>({...v,[name]:value}));}
  function clear(){runner.current?.schedule(null);setCompleted(null);setError(null);setInputs({});}
  function cancel(){runner.current?.cancel();setCompleted(null);setError("已取消即時預測；可修改輸入或手動重試。Browser 取消會清除記憶體模型。");}
  async function remove(){
    if(paused||busy||batchBusy||isUnavailable||deleting)return;runner.current?.schedule(null);setDeleting(true);setCompleted(null);setError(null);const abort=new AbortController();deleteController.current=abort;const timer=setTimeout(()=>abort.abort(),15000);
    try{const response=await request("/api/custom-dataset/predict",{method:"DELETE",headers:{"Content-Type":"application/json"},body:JSON.stringify({model_id:model.id}),signal:abort.signal,cache:"no-store"});if(!response.ok)throw new Error("暫時無法清除模型，請重試。");setUnavailable(true);}
    catch(e){if(!abort.signal.aborted)setError(e instanceof Error?e.message:"清除模型失敗。");else setError("清除模型已取消或逾時。");}
    finally{clearTimeout(timer);if(deleteController.current===abort){deleteController.current=null;setDeleting(false);}}
  }
  return <><section className="cd-single-prediction" aria-labelledby="custom-predict-title" aria-busy={busy||deleting}>
    <div className="section-kicker">08 · LIVE PREDICTOR</div><h3 id="custom-predict-title">即時預測 · 單筆輸入</h3>
    <p>使用剛才訓練好的模型估計 <strong>{model.target}</strong>。停止輸入 500 毫秒後自動預測，只做推論，不重新訓練或更新評估成績。</p>
    <p className="cd-confirmation">滑桿 Min／Max 僅來自完整 Training 的原始有效值，不含 Holdout；起始輸入為範圍中點。Number Input 可輸入範圍外數值，預測保留你輸入的原值。{model.missing_numeric==="none"?"此模型不補值，每個欄位都必填。":`空白沿用 Training 的${model.missing_numeric==="mean"?"平均值":"中位數"}，至少填一欄。`}</p>
    {!ranges&&<p className="cd-confirmation">此舊結果沒有 Training 範圍，滑桿暫不可用；可輸入數值，或重新訓練以取得範圍。不會使用全資料或 Holdout 代替。</p>}
    {!isUnavailable?<form noValidate onSubmit={e=>{e.preventDefault();if(!invalid&&filled&&complete&&!paused&&!batchBusy&&!deleting)runner.current?.schedule(values,true);}}>
      <label className="cd-live-consent"><input id="custom-live-enabled" type="checkbox" checked={live} onChange={e=>{runner.current?.schedule(null);setLive(e.target.checked);}}/> 自動預測（停止輸入後更新）</label>
      <div className="cd-config-grid cd-predict-fields">{model.features.map((name,index)=>{
        const range=ranges?.features.find(r=>r.feature===name),view=sliderPresentation(inputs[name]??"",range?.min??null,range?.max??null);
        return <div className="cd-live-feature" key={name}><label htmlFor={`custom-predict-${index}`}><span id={`custom-feature-label-${index}`}>{name}{model.missing_numeric==="none"?" · 必填":""}</span><input id={`custom-predict-${index}`} type="number" inputMode="decimal" step="any" autoComplete="off" value={inputs[name]??""} placeholder="請輸入原始數值" disabled={deleting} onChange={e=>edit(name,e.target.value)} aria-describedby={`custom-range-${index}`}/></label>
          <input id={`custom-slider-${index}`} type="range" aria-label={`${name} 滑桿`} min={view.min} max={view.max} step={view.step} value={view.value} disabled={view.disabled||deleting} onChange={e=>edit(name,e.target.value)} aria-describedby={`custom-range-${index}`}/>
          <p id={`custom-range-${index}`} className="cd-live-range">{range?.min!==null&&range?.min!==undefined&&range.max!==null?`Training Min ${format(range.min)} · Max ${format(range.max)}`:"Training 無有效觀測範圍，滑桿不可用。"}</p>
          {view.outside&&<p className="cd-live-warning" role="status">外插預測警告：{name} 超出 Training 範圍。滑桿僅顯示邊界；預測仍使用 Number Input 原值，可靠性可能降低。</p>}
        </div>;
      })}</div>
      {invalid&&<p className="cd-error" role="alert">{invalid}</p>}{!complete&&filled>0&&<p className="cd-confirmation">請填完全部 {model.features.length} 欄。</p>}
      {paused&&<p role="status">比較進行中，即時預測暫停；完成或取消後會沿用原模型恢復。</p>}{batchBusy&&<p role="status">批次推論進行中；即時預測暫停，完成後使用最新輸入。</p>}
      <div className="cd-download-actions"><button id="dataset-predict" type="submit" className="cd-button cd-primary" disabled={!!invalid||!filled||!complete||paused||batchBusy||deleting}>{busy?"更新最新預測":"立即預測／重試"}</button>{busy&&<button type="button" className="cd-button" onClick={cancel}>取消等待</button>}<button type="button" className="cd-button" onClick={clear}>清空輸入</button><button type="button" className="cd-button" disabled={paused||busy||batchBusy||deleting} onClick={()=>void remove()}>清除此模型</button></div>
    </form>:<p className="cd-validation" role="status">暫存模型已無法使用；評估成績仍保留。請重新訓練後再做單筆預測。</p>}
    {error&&<p className="cd-error" role="alert">{error}</p>}
    {result&&completed&&<div className="cd-prediction-output" role="status" aria-live="polite"><span>{result.target} · 預測值（原始單位）</span><strong>{format(result.prediction)}</strong><p>這是單筆估計，未提供預測區間；沒有實際答案，就無法計算這一筆的 R²、RMSE 或 MAE。</p>{result.imputed.length>0&&<p>本次補值：{result.imputed.map(r=>`${r.feature} = ${format(r.value)}`).join("；")}</p>}{result.out_of_range.length>0&&<p>外插預測：超出 Training 範圍 {result.out_of_range.join("、")}</p>}{result.warnings.length>0&&<ul>{result.warnings.map((w,i)=><li key={i}>{w}</li>)}</ul>}<button id="custom-prediction-json" type="button" className="cd-button" onClick={()=>{try{downloadArtifact(predictionJson(model,result,completed.values));}catch{setError("無法產生單筆預測檔案。");}}}>下載這筆預測 JSON</button><p className="cd-confirmation">下載包含送出當時的原始輸入與實際補值，不含模型。</p></div>}
    <p className="cd-confirmation">模型在 {compute.mode==="browser"?"瀏覽器記憶體":"運算服務私有暫存"}，約 30 分鐘內可用{expiry?`（最晚至 ${expiry}）`:""}，最多 3 個。取消、逾時、換檔或切換模式的清理行為維持原設定。</p>
  </section><CustomBatchPrediction key={`${model.id}-${unavailable}`} model={model} unavailable={isUnavailable} disabled={paused||busy||deleting} onBusyChange={active=>{if(active)runner.current?.schedule(null);setBatchBusy(active);}} onModelExpired={()=>setUnavailable(true)}/></>;
}
