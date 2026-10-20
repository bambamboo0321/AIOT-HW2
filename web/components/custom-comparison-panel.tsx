"use client";
import { useCustomCompute } from "./custom-compute-provider";
import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { isAutoMethod, estimatedFits, nestedSelectionError } from "@/lib/custom-selection-policy";
import { CustomSelectionDetails } from "./custom-selection-details";
import type { PreprocessingPlan } from "@/lib/custom-dataset";
import type { ValidationReport } from "@/lib/custom-validation-result";
import { COMPARISON_LABELS as labels, COMPARISON_METHODS as allMethods, parseComparisonReport, type Compared, type ComparisonMethod, type ComparisonReport } from "@/lib/custom-comparison-result";
import { comparisonCsv, comparisonJson } from "@/lib/custom-comparison-export";
import { downloadArtifact } from "@/lib/browser-download";
type Validated=Extract<ValidationReport,{status:"valid"}>;
const format=(n:number|null)=>n===null?"無法定義":n.toLocaleString("en-US",{maximumFractionDigits:4});
const colors:Record<ComparisonMethod,string>={forward_sfs:"var(--comparison-sfs)",rfecv:"var(--selected)",lasso_cv:"var(--baseline)",f_test:"var(--baseline)",mutual_information:"var(--selected)",rfe:"var(--comparison-rfe)"};
export function CustomComparisonPanel({file,config,validation,disabled,holdoutViewed,onBusyChange,onApply}:{file:File;config:PreprocessingPlan;validation:Validated;disabled:boolean;holdoutViewed:boolean;onBusyChange:(busy:boolean)=>void;onApply:(method:ComparisonMethod,k:number|null)=>void}) {
  const compute=useCustomCompute();
  const [methods,setMethods]=useState<ComparisonMethod[]>(["f_test","mutual_information","rfe"]);
  const [min,setMin]=useState("1"),[max,setMax]=useState(String(Math.min(10,config.feature_columns.length)));
  const [consent,setConsent]=useState(false),[busy,setBusy]=useState(false),[result,setResult]=useState<ComparisonReport|null>(null),[error,setError]=useState<string|null>(null);
  const generation=useRef(0),controller=useRef<AbortController|null>(null);
  useEffect(()=>()=>{generation.current++;controller.current?.abort();},[]);
  function reset(){generation.current++;controller.current?.abort();controller.current=null;setBusy(false);onBusyChange(false);setConsent(false);setResult(null);setError(null);}
  function cancel(){reset();setError("比較已取消；未完成的成績不會顯示。");}
  const p=config.feature_columns.length,lo=Number(min),hi=Number(max);
  const rangeValid=/^[1-9]\d*$/.test(min)&&/^[1-9]\d*$/.test(max)&&Number.isSafeInteger(lo)&&Number.isSafeInteger(hi)&&lo<=hi&&hi<=p&&hi-lo<25;
  const hasFixed=methods.some(m=>!isAutoMethod(m));
  const ks=rangeValid&&hasFixed?Array.from({length:hi-lo+1},(_,i)=>lo+i):[];
  const fits=methods.reduce((sum,m)=>sum+(isAutoMethod(m)?[null]:ks).reduce<number>((n,k)=>n+estimatedFits(m,k,p),0),0);
  const work=fits*p*validation.split.cv_folds.reduce((sum,f)=>sum+f.training_rows,0);
  const nestedError=methods.flatMap(m=>(isAutoMethod(m)?[null]:ks).map(k=>nestedSelectionError(m,k,p,validation.dataset.rows,validation.split.cv_folds.map(f=>f.training_rows),config.split_strategy,config.cv_gap))).find(Boolean);
  const blocked=nestedError??(config.numerical_scaling!=="standard"?"比較需要標準化。請回上方將數值縮放設為 StandardScaler，再確認並重新驗證。":p>25||validation.dataset.rows>5000?"本版比較限 5,000 筆與 25 個候選特徵；請縮小資料或候選欄位。":validation.split.cv_folds.some(f=>f.training_rows<4)?"每個 CV Training 至少需要 4 筆，請調整切分或 folds。":hasFixed&&!rangeValid?`k 須為 1～${p} 的整數，起點不可大於終點，最多 25 個 k。`:!methods.length?"請至少選擇一種方法。":work>100_000_000?"比較運算量過大，請縮小 k 範圍、減少方法／folds 或資料筆數。":null);
  async function compare(){
    if(busy||disabled||blocked||!consent)return;
    const token=++generation.current,abort=new AbortController();controller.current=abort;setBusy(true);onBusyChange(true);setResult(null);setError(null);let expired=false;
    const timer=setTimeout(()=>{expired=true;abort.abort();},130_000);
    const body=new FormData();body.append("csv",file);body.append("config",JSON.stringify(config));
    body.append("review",JSON.stringify({schema_version:1,artifact_type:"custom_dataset_review",dataset:{sha256:validation.dataset.sha256},preprocessing:config,review:{confirmed:true,training_started:false,scope:"configuration_only"}}));
    body.append("experiment",JSON.stringify({schema_version:1,confirmed:true,dataset_sha256:validation.dataset.sha256,config_sha256:validation.config_sha256,methods,k_values:ks}));
    try{
      const response=await compute.request("/api/custom-dataset/compare",{method:"POST",body,signal:abort.signal,cache:"no-store"}),payload:unknown=await response.json();
      if(![200,422].includes(response.status))throw new Error(payload&&typeof payload==="object"&&"error" in payload&&typeof payload.error==="string"?payload.error:"本機比較服務無法完成，請稍後重試。");
      const report=parseComparisonReport(payload);
      if((response.status===200)!==(report.status==="compared"))throw new Error("比較回應狀態不一致。");
      if(report.status==="compared"&&(report.validation.dataset.sha256!==validation.dataset.sha256||report.validation.config_sha256!==validation.config_sha256||report.target!==config.target_column||JSON.stringify(report.features)!==JSON.stringify(config.feature_columns)||JSON.stringify(report.methods)!==JSON.stringify(methods)||JSON.stringify(report.k_values)!==JSON.stringify(ks)))throw new Error("比較與目前資料或設定版本不一致，請重新驗證。");
      if(token===generation.current)setResult(report);
    }catch(e){if(token===generation.current)setError(expired?"比較逾時，請縮小範圍後重試。":e instanceof Error&&e.name!=="AbortError"?e.message:"比較已取消。");}
    finally{clearTimeout(timer);if(token===generation.current){setBusy(false);onBusyChange(false);controller.current=null;}}
  }
  return <section className="panel cd-comparison" aria-labelledby="custom-comparison-title" aria-busy={busy}><div className="panel-header"><div><div className="section-kicker">06 · COMPARE BEFORE HOLDOUT</div><h2 id="custom-comparison-title">特徵選擇 × k 比較</h2><p>先用 Training CV 比較方法與特徵數，再決定正式測試的設定。</p></div><span className="neutral-tag">探索性 CV</span></div><div className="cd-body">
    <p>F-test／MI 依關聯排名，RFE 逐次移除特徵，Forward SFS 逐次加入特徵。RFECV／LassoCV 會自動選 k，另外呈現逐 fold 結果。全部方法使用相同外層切分，選欄後另 fit Linear Regression；SFS 與自動方法的內層為 3-fold CV。</p>
    <div className="cd-chart-guide" id="comparison-k-guide"><strong>k 是保留幾個特徵，不是第幾個欄位</strong><p>例如 k＝{Math.min(3,p)}，就是從你勾選的 {p} 個候選欄位中，讓各方法選出 {Math.min(3,p)} 欄，再用線性迴歸預測。實際選到哪些欄位，要看下方的特徵清單。</p><p>起始／結束 k 決定這次要比較的數量範圍，預設先比較 1～{Math.min(10,p)}；可調整到候選欄位數（本版最多 25 欄）。保留更多欄位不一定更準，先看 Training CV 的誤差與穩定性。</p></div>
    <fieldset className="cd-comparison-methods" disabled={busy||disabled}><legend>要比較的方法</legend>{allMethods.map(m=><label key={m}><input type="checkbox" checked={methods.includes(m)} onChange={e=>{reset();setMethods(allMethods.filter(n=>n===m?e.target.checked:methods.includes(n)));}}/>{labels[m]}</label>)}</fieldset>
    <div className="cd-config-grid"><label htmlFor="comparison-k-min">起始特徵數 k<input disabled={busy||disabled||!hasFixed} id="comparison-k-min" type="number" min={1} max={p} step={1} value={min} onChange={e=>{reset();setMin(e.target.value);}}/></label><label htmlFor="comparison-k-max">結束特徵數 k<input disabled={busy||disabled||!hasFixed} id="comparison-k-max" type="number" min={1} max={p} step={1} value={max} onChange={e=>{reset();setMax(e.target.value);}}/></label></div>
    {!hasFixed&&methods.length>0&&<p className="cd-confirmation">目前只選自動方法，不使用上方的起始／結束 k；系統會在各 Training 區自行決定 k。</p>}
    <p className="cd-confirmation">{methods.filter(m=>!isAutoMethod(m)).length} 種固定方法 × {ks.length} 個 k；{methods.filter(isAutoMethod).length} 種自動方法 · {config.cv_splits} 個外層 folds · 候選 {p} 欄 · k 等於候選數時保留全部，未執行排名。上限 {compute.mode==="browser"?90:120} 秒，可取消；較大資料請分段比較。</p>
    {blocked&&<p className="cd-error" role="alert">{blocked}</p>}{holdoutViewed&&<p className="cd-validation cd-cautions">你已看過這份資料的 Holdout。此時調整方法或 k 後再次測試，不能視為全新的獨立驗證。</p>}
    <label className="cd-consent"><input id="comparison-consent" type="checkbox" checked={consent} disabled={busy||disabled||!!blocked} onChange={e=>setConsent(e.target.checked)}/>我確認只在 Training CV 比較，不使用 Holdout 選方法或 k；本次不保存模型。</label>
    <div className="cd-download-actions"><button id="dataset-compare" className="cd-button cd-primary" disabled={busy||disabled||!!blocked||!consent} onClick={()=>void compare()}>{busy?"正在比較…":"開始方法與 k 比較"}</button>{busy&&<button className="cd-button" id="comparison-cancel" onClick={cancel}>取消比較</button>}</div>
    {busy&&<p role="status">Python 正在逐 fold 計算真實成績；完成前不顯示部分結果。</p>}{error&&<p role="alert" className="cd-error">{error}</p>}{result&&result.status!=="compared"&&<div role="alert" className="cd-error">{result.errors.map((e,i)=><p key={i}>{e.message}</p>)}</div>}
    {result?.status==="compared"&&<ComparisonResults key={JSON.stringify([result.methods,result.k_values])} report={result} config={config} disabled={busy||disabled} onApply={onApply}/>}</div></section>;
}
function ComparisonResults({report,config,disabled,onApply}:{report:Compared;config:PreprocessingPlan;disabled:boolean;onApply:(method:ComparisonMethod,k:number|null)=>void}){
  const [metric,setMetric]=useState<"rmse"|"mae"|"r2">("rmse"),[k,setK]=useState(report.k_values[0]),[error,setError]=useState<string|null>(null);
  const fixedMethods=report.methods.filter(m=>!isAutoMethod(m));
  const chart=report.k_values.map(k=>({k,...Object.fromEntries(fixedMethods.map(m=>[m,report.points.find(p=>p.method===m&&p.k===k)!.metrics[metric].mean]))})),label=metric==="r2"?"R²":metric.toUpperCase();
  function download(kind:"csv"|"json"){try{downloadArtifact(kind==="csv"?comparisonCsv(report):comparisonJson(report,config));setError(null);}catch{setError("下載無法完成，請重新執行比較。");}}
  return <div id="custom-comparison-result" className="cd-comparison-result">{fixedMethods.length>0&&<><div className="cd-comparison-heading"><h3>Training CV · 方法與特徵數</h3><div className="segmented" role="group" aria-label="比較指標">{(["rmse","mae","r2"] as const).map(m=><button key={m} aria-pressed={metric===m} onClick={()=>setMetric(m)}>{m==="r2"?"R²":m.toUpperCase()}</button>)}</div></div>
    <p className="cd-confirmation">X 軸：保留特徵數 k · Y 軸：{label} 的各 fold 平均，{metric==="r2"?"越高越好，不是準確率":"越低越好，與目標同單位"}。線只連接實際計算點；無法定義的 R² 保持空白。</p>
    <div className="cd-training-chart" role="img" aria-label={`${label} 隨保留特徵數 k 變化；完整數值及選中特徵見下方表格`}><ResponsiveContainer width="100%" height="100%"><LineChart data={chart} margin={{top:16,right:20,bottom:10,left:0}}><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3"/><XAxis dataKey="k" type="number" domain={[report.k_values[0],report.k_values.at(-1)!]} allowDecimals={false} tick={{fill:"var(--muted)",fontSize:12}}/><YAxis domain={["auto","auto"]} width={76} tickFormatter={n=>Number(n).toLocaleString("en-US",{maximumFractionDigits:3})} tick={{fill:"var(--muted)",fontSize:12}}/><Tooltip labelFormatter={n=>`k = ${n}`} formatter={v=>typeof v==="number"?format(v):"無法定義"} contentStyle={{background:"var(--surface)",borderColor:"var(--border)"}}/><Legend/>{fixedMethods.map((m,i)=><Line key={m} type="linear" dataKey={m} name={labels[m]} stroke={colors[m]} strokeWidth={2} strokeDasharray={i===1?"6 3":i===2?"2 3":undefined} dot={{r:4}} connectNulls={false} isAnimationActive={false}/>)}</LineChart></ResponsiveContainer></div>
    <div className="cd-chart-guide"><p>同一個 k 在不同方法或 fold 中可能選到不同特徵。依曲線挑出的最佳分數可能偏樂觀，需固定設定後，再到下方評估一次 Holdout。</p><p>下方可將目前 k 與指定方法套用到正式訓練；你仍需確認後才會評估 Holdout。RFECV／LassoCV 的自動選 k 結果另列於下方，不套用這個 k 選單。</p></div>
    <div className="fs-table-scroll"><table className="metrics-table"><caption>每 fold 等權平均 ± 標準差（ddof=0）；SFS 內部另做 3-fold 選欄；標準差不是信賴區間</caption><thead><tr><th>k</th>{fixedMethods.map(m=><th key={m}>{labels[m]} · {label}</th>)}</tr></thead><tbody>{report.k_values.map(n=><tr key={n}><th>{n}</th>{fixedMethods.map(m=>{const a=report.points.find(p=>p.method===m&&p.k===n)!.metrics[metric];return <td key={m}>{format(a.mean)}{a.std!==null&&<> ± {format(a.std)}</>}<small className="cd-defined-folds">{a.defined_folds}／{config.cv_splits} folds 有定義</small></td>;})}</tr>)}</tbody></table></div>
    <label className="cd-comparison-inspect" htmlFor="comparison-inspect-k">查看／準備套用的特徵數 k<select id="comparison-inspect-k" aria-describedby="comparison-selected-k-guide" value={k} onChange={e=>setK(Number(e.target.value))}>{report.k_values.map(n=><option key={n} value={n}>k = {n}</option>)}</select></label>
    <p id="comparison-selected-k-guide" className="cd-confirmation">目前查看 k＝{k}：每種方法各保留 {k} 個特徵。這裡只切換查看的成績與特徵清單；選好方法後，按「套用」才會帶入下方正式訓練，並由你再次確認才開始。</p>
    <div className="cd-download-actions" role="group" aria-label="套用比較設定">{fixedMethods.map(m=><button key={m} id={`comparison-apply-${m}`} className="cd-button cd-primary" disabled={disabled} onClick={()=>onApply(m,k)}>套用 {labels[m]} · k = {k} 到正式訓練</button>)}</div>
    <p className="cd-confirmation">只帶入方法與 k，不會直接沿用任何 fold 的特徵清單，也不會自動訓練。正式訓練會在完整 Training 重新選欄。</p>
    {fixedMethods.map(m=><details key={m} className="cd-cv-details"><summary>{labels[m]} · k = {k} · 各 fold 成績與特徵{k===report.features.length?"（全部保留，未排名）":""}</summary><div className="fs-table-scroll"><table className="metrics-table"><thead><tr><th>Fold</th><th>訓練／驗證筆數</th><th>選中特徵</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>{report.points.find(p=>p.method===m&&p.k===k)!.folds.map(f=><tr key={f.fold}><th>{f.fold}</th><td>{f.training_rows}／{f.validation_rows}</td><td>{f.features.join("、")}</td><td>{format(f.scores.rmse)}</td><td>{format(f.scores.mae)}</td><td>{format(f.scores.r2)}</td></tr>)}</tbody></table></div></details>)}
    </>}
    {report.automatic.map(auto=><section key={auto.method} className="cd-auto-comparison"><h3>{labels[auto.method]} · 自動選 k 的外層 CV</h3><p className="cd-confirmation">每個外層 fold 的 Training 內另做 3-fold CV；下表為外層驗證成績。這是整個自動選 k 程序的評估，不是固定 k 曲線。正式訓練會在完整 Training 重新選 k，不能把某個 fold 的 k 當成最終設定。</p><div className="fs-table-scroll"><table className="metrics-table"><caption>外層各 fold 等權平均 ± 標準差（ddof=0）</caption><thead><tr><th>指標</th><th>平均 ± 標準差</th><th>有效 folds</th></tr></thead><tbody>{(["rmse","mae","r2"] as const).map(metric=><tr key={metric}><th>{metric==="r2"?"R²":metric.toUpperCase()}</th><td>{format(auto.metrics[metric].mean)} ± {format(auto.metrics[metric].std)}</td><td>{auto.metrics[metric].defined_folds}／{auto.folds.length}</td></tr>)}</tbody></table></div><div className="fs-table-scroll"><table className="metrics-table"><caption>每個 fold 實際選到的整數 k 與特徵；0 欄時不訓練 LR、不計分</caption><thead><tr><th>Fold</th><th>k</th><th>RMSE</th><th>MAE</th><th>R²</th><th>alpha</th><th>選中特徵</th></tr></thead><tbody>{auto.folds.map(f=><tr key={f.fold}><th>{f.fold}</th><td>{f.features.length}</td><td>{format(f.scores?.rmse??null)}</td><td>{format(f.scores?.mae??null)}</td><td>{format(f.scores?.r2??null)}</td><td>{f.details.alpha??"—"}</td><td>{f.features.length?f.features.join("、"):"0 欄，未計分"}</td></tr>)}</tbody></table></div><details className="cd-cv-details"><summary>查看各 fold 的內部選模設定</summary>{auto.folds.map(f=><div key={f.fold}><strong>Fold {f.fold}</strong><CustomSelectionDetails details={f.details}/></div>)}</details><div className="cd-download-actions"><button id={`comparison-apply-${auto.method}`} className="cd-button cd-primary" disabled={disabled} onClick={()=>onApply(auto.method,null)}>套用 {labels[auto.method]} · 自動選 k 到正式訓練</button></div></section>)}
    <div className="cd-download-actions"><button className="cd-button" id="comparison-download-csv" onClick={()=>download("csv")}>下載完整 CV 成績 CSV</button><button className="cd-button" id="comparison-download-json" onClick={()=>download("json")}>下載比較與設定 JSON</button></div>{error&&<p className="cd-error" role="alert">{error}</p>}
    {report.warnings.length>0&&<div className="cd-validation cd-cautions"><ul>{report.warnings.map((w,i)=><li key={i}>{w}</li>)}</ul></div>}
    <details className="cd-provenance"><summary>比較版本與限制</summary><p>目標：{report.target} · 候選：{report.features.join("、")}</p><p>CSV SHA-256：<code>{report.validation.dataset.sha256}</code></p><p>設定 SHA-256：<code>{report.validation.config_sha256}</code></p><ul>{report.limits.map((l,i)=><li key={i}>{l}</li>)}</ul></details></div>;
}
