"use client";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { METHOD_NAMES, type MethodId, type Metric, type Scores } from "@/lib/feature-selection-types";
import { studyComparisonRows, type StudySelectionData } from "@/lib/study-feature-selection-data";
const descriptions: Record<MethodId,string> = {
  f_test:"F-test 依單欄與目標的線性關聯選出固定 7 欄；分數不代表因果，亦不評估欄位間所有交互作用。",
  mutual_information:"Mutual Information 依互資訊選出固定 7 欄；未掃描 k。此方法由既有 Training CV 規則鎖定，不會隨此頁操作重新選模。",
  forward_sfs:"Forward SFS 在各外層 Training fold 內，透過內層 3-fold RMSE 逐步加入特徵至 k=7；搜尋可能成本高且不保證全域最佳子集。",
  rfe:"RFE 以標準化後線性迴歸係數逐步刪除特徵至 k=7；相關特徵可能影響係數與刪除次序。",
  lasso_cv:"LassoCV 透過內層 3-fold 平均 MSE 選 alpha，以非零係數自動決定 k；最終預測器仍為獨立 Linear Regression。各 fold 與完整 Training 可選出不同 k。",
  rfecv:"RFECV 以內層 3-fold RMSE 評估逐步刪除的特徵數，自動選 k；各 fold 與完整 Training 可選出不同 k。",
};
const shortNames: Record<MethodId,string> = {f_test:"F-test",mutual_information:"MI",forward_sfs:"SFS",rfe:"RFE",lasso_cv:"LassoCV",rfecv:"RFECV"};
const fmt = (n:number) => n.toLocaleString("en-US",{minimumFractionDigits:4,maximumFractionDigits:4});
const metricName = (m:Metric) => m === "r2" ? "R²" : m.toUpperCase();
function ScoreCells({mean,std}:{mean:Scores;std:Scores}) { return <>{(["rmse","mae","r2"] as const).map(k=><td key={k}>{fmt(mean[k])} ± {fmt(std[k])}</td>)}</>; }
export function StudyFeatureExplorer({data}:{data:StudySelectionData}) {
  const [selectedId,setSelectedId]=useState<MethodId>("mutual_information");
  const [metric,setMetric]=useState<Metric>("rmse");
  const selected=data.methods.find(m=>m.id===selectedId)!;
  const unit=metric === "r2" ? "" : ` ${data.study.dataset.unit}`;
  const rows=studyComparisonRows(data,metric).map(row=>({...row,label:shortNames[row.id],color:row.id===selectedId?"var(--selected)":"var(--baseline)"}));
  return <>
    <section className="panel fs-explorer" aria-labelledby="study-explorer-title"><div className="panel-header"><div><div className="section-kicker">SIX METHODS · TRAINING CV</div><h2 id="study-explorer-title">六方法比較與選中特徵</h2><p>四種固定 k=7 方法與兩種 auto 方法；只瀏覽成果，不啟動訓練。</p></div><span className="neutral-tag">固定 k / auto</span></div>
      <div className="fs-methods" role="group" aria-label="特徵選擇方法">{data.methods.map(m=><button key={m.id} aria-pressed={m.id===selectedId} className={m.id===selectedId?"active":""} onClick={()=>setSelectedId(m.id)}>{METHOD_NAMES[m.id]}<small>{m.status === "ready" ? `${m.mode === "auto" ? "Auto → 最終" : "固定"} k=${m.finalK}` : "unavailable"}</small></button>)}</div>
      <div className="fs-explorer-body">
        <div className="chart-controls"><span className="fs-chart-level">5-fold Training CV · 每 fold 等權平均</span><div className="segmented-control" role="group" aria-label="特徵選擇評估指標">{(["rmse","mae","r2"] as const).map(k=><button key={k} aria-pressed={metric===k} className={metric===k?"selected":""} onClick={()=>setMetric(k)}>{metricName(k)}</button>)}</div></div>
        <p className="fs-axis-note">橫軸：{metricName(metric)}{unit}（{metric === "r2" ? "越高越好；不是準確率" : "越低越好"}）；縱軸：方法。紫色為目前檢視的方法。</p>
        {rows.some(r=>r.value!==null) ? <div className="fs-chart" role="img" data-study-metric={metric} aria-label={`六方法 Training CV ${metricName(metric)} 橫條比較；不是 k 曲線，數值見表格`}>
          <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{width:650,height:290}}><BarChart data={rows} layout="vertical" margin={{top:10,right:25,bottom:8,left:0}} accessibilityLayer>
            <CartesianGrid horizontal={false} stroke="var(--chart-grid)" strokeDasharray="3 4"/>
            <XAxis type="number" domain={[Math.min(0,...rows.flatMap(r=>r.value===null?[]:[r.value])),"auto"]} tickFormatter={n=>Number(n).toLocaleString("en-US",{maximumFractionDigits:4})} tick={{fill:"var(--muted)",fontSize:10}}/>
            <YAxis type="category" dataKey="label" width={75} tick={{fill:"var(--text-secondary)",fontSize:11}}/>
            <Tooltip cursor={{fill:"var(--hover)"}} contentStyle={{background:"var(--surface)",border:"1px solid var(--border)",borderRadius:8,color:"var(--text)"}} formatter={v=>[`${fmt(Number(v))}${unit}`,metricName(metric)]}/>
            <Bar dataKey="value" barSize={24} isAnimationActive={false} shape={props=><rect x={props.x} y={props.y} width={props.width} height={props.height} fill={props.payload.color} rx={4}/>}/>
          </BarChart></ResponsiveContainer>
        </div> : <p className="missing-inline" role="status">CV 比較圖 unavailable：沒有完整且可比較的 CV 摘要。</p>}
        {data.methods.filter(m=>m.status!=="ready"||m.cv.status!=="ready").map(m=><p key={m.id} className="missing-inline" role="status">{METHOD_NAMES[m.id]} CV unavailable：{m.status === "unavailable" ? m.reason : m.cv.status === "unavailable" ? m.cv.reason : ""}</p>)}
        <p className="fs-note">此圖是六個方法的程序比較，不是隨 k 變化的曲線。固定方法只有 k=7 的實驗；完整 k 曲線 unavailable（未執行 k sweep）。auto 方法的平均分數也不等於在最終 k 固定評估的分數。</p>
        <div className="fs-table-scroll"><table className="metrics-table" data-study-cv-table><caption>Training CV 平均 ± 標準差（ddof=0），非信賴區間、非 pooled 指標，也不是 Holdout</caption><thead><tr><th>方法</th><th>完整 Training k</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>{data.methods.map(m=><tr key={m.id}><th scope="row">{METHOD_NAMES[m.id]}{m.id===data.study.protocol.selectedMethod&&<small>Training CV 鎖定方法</small>}</th>{m.status === "ready" ? <><td>{m.mode === "auto" ? "auto → " : "固定 "}{m.finalK}</td>{m.cv.status === "ready" ? <ScoreCells mean={m.cv.mean} std={m.cv.std}/> : <td colSpan={3}>CV unavailable：{m.cv.reason}</td>}</> : <td colSpan={4}>unavailable：{m.reason}</td>}</tr>)}</tbody></table></div>
        <p className="fs-note">RMSE／MAE 單位：{data.study.dataset.unit}；R² 無單位。方法由 Training CV 的平均 RMSE 規則鎖定；切換圖表指標不會重新挑選方法，也不使用 Holdout 排名。比較多個方法後的勝出者 CV 仍有選取樂觀偏差。</p>
      </div>
    </section>
    <section className="panel fs-selection" aria-labelledby="study-selected-features-title" data-study-method={selected.id}><div className="panel-header"><div><div className="section-kicker">FINAL TRAINING SELECTION</div><h2 id="study-selected-features-title">{METHOD_NAMES[selected.id]} · 選中的特徵</h2><p>完整 Training 最終重 fit 的選欄；不是把某個 fold 的欄位套用到其他 fold。</p></div><span className="neutral-tag">{selected.status === "ready" ? `最終 k=${selected.finalK}` : "unavailable"}</span></div>
      <div className="fs-section-body"><p className="fs-body-text">{descriptions[selected.id]}</p>{selected.status === "ready" ? <>
        <div className="feature-chips" data-study-final-features>{selected.features.map(f=><span key={f}>{f}</span>)}</div>
        <div className="fs-point-summary"><strong>完整 Training 最終 k = {selected.finalK}</strong><span>{selected.mode === "auto" ? "auto 的最終 k 來自完整 Training 內部 CV，不能當作每個外層 fold 都使用此 k。" : "k=7 是預先指定的設定，沒有用 CV 或 Holdout 掃描其他 k。"}</span></div>
        <div className="fs-fold-selection"><p className="fs-fold-heading">各 Outer fold 的實際 k 與驗證分數 · 非時間順序</p><ol className="fs-fold-cards">{selected.folds.map(f=><li key={f.fold}><span>Fold {f.fold}</span><strong>{f.k}<small> 個特徵</small></strong><span>{metricName(metric)} {f.scores ? fmt(f.scores[metric]) : "unavailable"}{f.scores ? unit : ""}</span></li>)}</ol>{selected.folds.length<5&&<p className="missing-inline" role="status">部分 fold 成果 unavailable；沒有填入替代分數或特徵數。</p>}</div>
        <details className="fs-data-table"><summary>查看逐 fold 的特徵與分數</summary><div className="fs-table-scroll"><table className="metrics-table"><caption>每個外層 Training fold 內獨立選欄；各 fold 是不同隨機分組，不是時間曲線</caption><thead><tr><th>Fold</th><th>實際 k</th><th>RMSE</th><th>MAE</th><th>R²</th><th>特徵</th></tr></thead><tbody>{selected.folds.map(f=><tr key={f.fold}><th scope="row">{f.fold}</th><td>{f.k}</td>{f.scores ? <><td>{fmt(f.scores.rmse)}</td><td>{fmt(f.scores.mae)}</td><td>{fmt(f.scores.r2)}</td></> : <td colSpan={3}>unavailable</td>}<td>{f.features.join(", ")}</td></tr>)}</tbody></table></div></details>
        {selected.warnings.length>0&&<div className="fs-warning"><strong>實驗 Warnings</strong><ul>{selected.warnings.map(w=><li key={w}>{w}</li>)}</ul></div>}
      </> : <p className="missing-inline" role="status">方法成果 unavailable：{selected.reason}</p>}</div>
    </section>
  </>;
}
