"use client";
import { useState } from "react";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import type { EvaluationModelId, StudyEvaluationData } from "@/lib/study-evaluation-data";
import { studyPredictionRows, visibleStudyPredictions, type PredictionSort } from "@/lib/study-prediction-data";

const fmt = (n: number | null, digits = 6) => n === null ? "unavailable" : n.toLocaleString("en-US", { maximumFractionDigits: digits });
const label = (id: EvaluationModelId) => id === "baseline" ? "Baseline LR" : "MI + LR";
const tick = { fill: "var(--muted)", fontSize: 10 };
export function StudyPredictionBrowser({ data }: { data: StudyEvaluationData }) {
  const [modelId, setModelId] = useState<EvaluationModelId>("baseline");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<PredictionSort>("sample");
  const [position, setPosition] = useState(1);
  const model = data.models.find(m => m.id === modelId)!;
  const rows = studyPredictionRows(model), visible = visibleStudyPredictions(rows, query, sort);
  const current = visible.find(p => p.position === position) ?? visible[0];
  const currentIndex = visible.findIndex(p => p.position === current?.position);
  const color = modelId === "baseline" ? "var(--baseline)" : "var(--selected)";
  const unit = data.study.dataset.unit;
  function selectModel(id: EvaluationModelId) { setModelId(id); setQuery(""); setSort("sample"); setPosition(1); }
  return <section className="panel" data-study-prediction-controls={data.datasetId} data-study-prediction-model={modelId}>
    <div className="panel-header"><div><div className="section-kicker">SAMPLE EXPLORER · NOT TIME SERIES</div><h2>逐筆 Holdout 預測</h2></div><div className="segmented-control" role="group" aria-label="探索模型">{data.models.map(m => <button key={m.id} className={modelId === m.id ? "selected" : ""} aria-pressed={modelId === m.id} onClick={() => selectModel(m.id)}>{label(m.id)}</button>)}</div></div>
    <div className="px-section-body">
      <div className="study-prediction-filters"><label htmlFor="study-sample-search">搜尋樣本編號<input id="study-sample-search" type="search" inputMode="numeric" value={query} placeholder={`1–${data.study.protocol.holdoutRows}`} aria-describedby="study-sample-help" onChange={e => { setQuery(e.target.value); setPosition(1); }}/></label><label htmlFor="study-error-sort">排序<select id="study-error-sort" value={sort} onChange={e => setSort(e.target.value as PredictionSort)}><option value="sample">樣本編號</option><option value="error-desc">Absolute Error：大到小</option><option value="error-asc">Absolute Error：小到大</option></select></label><button className="study-sample-button" onClick={() => { setQuery(""); setSort("sample"); setPosition(1); }}>重設搜尋與排序</button></div>
      <p id="study-sample-help" className="fs-note">輸入完整整數編號精確搜尋；排序以絕對誤差判斷，相同誤差依樣本編號排列。散點圖 X 軸維持原始 Holdout 筆次，排序只影響下方表格與逐筆瀏覽。</p>
      <p role="status" className="fs-note" data-study-visible-count={visible.length}>{label(modelId)} · 顯示 {visible.length} / {rows.length} 筆</p>
      {model.predictions.status !== "ready" ? <p role="status" className="missing-inline">逐筆預測 unavailable：{model.predictions.reason}</p> : visible.length === 0 ? <p role="status" className="missing-inline">找不到樣本；請輸入範圍內的完整整數編號，或重設搜尋。</p> : <>
        <p className="ev-chart-caption">X 軸：Holdout 樣本編號（非時間）；Y 軸：{data.study.dataset.target}（{unit}）。深色圓點為 Actual，{modelId === "baseline" ? "綠色" : "紫色"}圓點為 Predicted。</p>
        <div className="ev-residual-chart" role="img" aria-label={`${label(modelId)} 實際與預測樣本索引散點圖`} data-study-sample-count={visible.length}><ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 650, height: 270 }}><ScatterChart margin={{ top: 12, right: 15, bottom: 15, left: 0 }} accessibilityLayer><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 4"/><XAxis type="number" dataKey="position" name="樣本編號" domain={[1, data.study.protocol.holdoutRows]} allowDecimals={false} tick={tick}/><YAxis type="number" dataKey="value" name={data.study.dataset.target} domain={["auto", "auto"]} tickFormatter={v => fmt(Number(v), 2)} tick={tick} width={70}/><Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text)" }}/><Scatter name="Actual" data={visible.map(p => ({ position: p.position, value: p.actual }))} fill="var(--text)" isAnimationActive={false}/><Scatter name={label(modelId)} data={visible.map(p => ({ position: p.position, value: p.predicted }))} fill={color} isAnimationActive={false}/></ScatterChart></ResponsiveContainer></div>
        <div className="study-sample-inspector" data-study-selected-sample={current.position}>
          <div className="study-prediction-filters"><label htmlFor="study-sample-detail">逐筆檢視<select id="study-sample-detail" value={current.position} onChange={e => setPosition(Number(e.target.value))}>{visible.map(p => <option key={p.position} value={p.position}>樣本 {p.position}</option>)}</select></label><button className="study-sample-button" disabled={currentIndex <= 0} onClick={() => setPosition(visible[currentIndex - 1].position)}>上一筆</button><button className="study-sample-button" disabled={currentIndex >= visible.length - 1} onClick={() => setPosition(visible[currentIndex + 1].position)}>下一筆</button></div>
          <h3>樣本 {current.position} · {label(modelId)}</h3>
          <dl className="study-sample-values">{[["Actual", current.actual], ["Predicted", current.predicted], ["Residual（實際 − 預測）", current.residual], ["Absolute Error", current.absoluteError], ["Lower 95% PI", current.lower], ["Upper 95% PI", current.upper]].map(([name, value]) => <div key={String(name)}><dt>{name}</dt><dd>{fmt(value as number | null)}</dd></div>)}</dl>
          <p className="fs-note">單位：{unit}。{current.residual > 0 ? "模型低估此筆實際值。" : current.residual < 0 ? "模型高估此筆實際值。" : "預測與實際值相同。"}{current.lower !== null && current.upper !== null ? current.actual >= current.lower && current.actual <= current.upper ? "實際值位於這筆保存的 95% PI 內。" : "實際值落在這筆保存的 95% PI 外。" : "這筆 PI unavailable。"}</p>
          {data.datasetId === "student-performance" && (current.predicted < 0 || current.predicted > 20 || (current.lower !== null && current.lower < 0) || (current.upper !== null && current.upper > 20)) && <p className="fs-warning">此筆預測或 PI 超出 G3 的 0–20 分範圍；這是線性模型的限制，保留原值，未任意裁切。</p>}
        </div>
        <div className="fs-table-scroll study-prediction-table"><table className="metrics-table" data-study-holdout-table><caption>{label(modelId)} · {visible.length}/{rows.length} 筆 · {unit} · 點選編號查看明細；表格可橫向捲動</caption><thead><tr><th>樣本編號</th><th>Actual</th><th>Predicted</th><th>Residual</th><th>Absolute Error</th><th>Lower 95% PI</th><th>Upper 95% PI</th></tr></thead><tbody>{visible.map(p => <tr key={p.position}><th scope="row"><button className="study-sample-button" aria-label={`檢視樣本 ${p.position}`} aria-pressed={current.position === p.position} onClick={() => setPosition(p.position)}>{p.position}</button></th><td>{fmt(p.actual)}</td><td>{fmt(p.predicted)}</td><td>{fmt(p.residual)}</td><td>{fmt(p.absoluteError)}</td><td>{fmt(p.lower)}</td><td>{fmt(p.upper)}</td></tr>)}</tbody></table></div>
      </>}
      {model.interval.status !== "ready" && <p className="missing-inline" role="status">95% PI unavailable：{model.interval.reason}</p>}
      <p className="fs-note">Residual = Actual − Predicted；Absolute Error = |Residual|。顯示值最多六位小數，所有計算保留快照原始精度；不是新的訓練結果。</p>
    </div>
  </section>;
}
