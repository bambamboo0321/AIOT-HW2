"use client";
import { useEffect, useRef, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { WINDOW_SIZES, type PredictionMeta, type PredictionModelId, type PredictionRow, type PredictionWindow, type WindowSize } from "@/lib/prediction-types";
import { explainR2, highestObservation } from "@/lib/result-interpretation";
import { parsePredictionWindow, timePosition } from "@/lib/prediction-window";
const format = (n: number, digits = 4) => n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const time = (s: string) => s.replace("T", " ").slice(0, 16);
const labels = { baseline: "Baseline LR", selected: "MI + LR" };
const covered = (r: PredictionRow, id: PredictionModelId) => r.actual >= r[id].lower && r.actual <= r[id].upper;
const color = (id: PredictionModelId) => id === "baseline" ? "var(--baseline)" : "var(--selected)";

export function PredictionExplorer({ meta, initial }: { meta: PredictionMeta; initial: PredictionWindow }) {
  const [window, setWindow] = useState(initial);
  const [models, setModels] = useState<"both" | PredictionModelId>("both");
  const [interval, setInterval] = useState<"none" | PredictionModelId>("selected");
  const [chartMode, setChartMode] = useState<"prediction" | "residual">("prediction");
  const [selected, setSelected] = useState(initial.rows[0].position);
  const [jumpTime, setJumpTime] = useState(initial.rows[0].timestamp.slice(0, 16));
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const request = useRef(0); const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const active = window.rows.find(r => r.position === selected) ?? window.rows[0];
  const visible = meta.models.filter(m => models === "both" || m.id === models);
  const lastStart = Math.max(0, meta.totalRows - window.size);
  const chartRows = window.rows.map(r => ({ ...r, baselinePrediction: r.baseline.prediction, selectedPrediction: r.selected.prediction,
    baselineResidual: r.actual - r.baseline.prediction, selectedResidual: r.actual - r.selected.prediction,
    band: interval === "none" ? undefined : [r[interval].lower, r[interval].upper] }));
  const peak = highestObservation(window.rows);
  const pageRows = window.rows.slice(page * 24, (page + 1) * 24);
  function choose(position: number) { setSelected(position); setPage(Math.floor((position - window.rows[0].position) / 24)); }
  async function load(start: number, size: WindowSize = window.size) {
    const token = ++request.current; controller.current?.abort(); const abort = new AbortController(); controller.current = abort;
    setLoading(true); setError(null);
    try {
      const response = await fetch(`/api/predictions?start=${start}&size=${size}`, { signal: abort.signal, cache: "no-store" });
      if (!response.ok) throw new Error("預測視窗暫時無法載入，請再試一次。");
      const next = parsePredictionWindow(await response.json());
      if (next.sourceDigest !== meta.sourceDigest || next.totalRows !== meta.totalRows) throw new Error("成果版本已改變，請重新整理頁面。");
      if (token !== request.current) return;
      const focus = next.rows.find(r => r.position === start + 1) ?? next.rows[0];
      setWindow(next); setSelected(focus.position); setJumpTime(focus.timestamp.slice(0, 16)); setPage(Math.floor((focus.position - next.rows[0].position) / 24));
    } catch (e) { if (token === request.current && !(e instanceof Error && e.name === "AbortError")) setError(e instanceof Error ? e.message : "無法載入預測視窗。"); }
    finally { if (token === request.current) setLoading(false); }
  }
  function jump(value: string) { try { void load(timePosition(value, meta)); } catch (e) { setError(e instanceof Error ? e.message : "時間無效"); } }
  function selectModels(next: typeof models) { setModels(next); if (next !== "both" && interval !== "none") setInterval(next); }
  return <>
    <section className="panel px-workbench" aria-labelledby="timeline-title" aria-busy={loading}>
      <div className="panel-header"><div><div className="section-kicker">EXPLORE SAVED HOLDOUT PREDICTIONS</div><h2 id="timeline-title">沿時間查看預測</h2><p>每個點都是十分鐘觀測，保留尖峰與原始區間。</p></div><span className="neutral-tag">{window.rows.length} 筆視窗</span></div>
      <div className="px-section-body">
        <div className="px-window-controls"><label htmlFor="prediction-size">視窗長度<select id="prediction-size" value={window.size} disabled={loading} onChange={e => { void load(window.start, Number(e.target.value) as WindowSize); }}>{WINDOW_SIZES.map(size => <option key={size} value={size}>{size / 6} 小時 · {size} 筆</option>)}</select></label><div className="px-window-nav" role="group" aria-label="預測視窗導航"><button disabled={loading || window.start === 0} onClick={() => { void load(0); }}>最前</button><button disabled={loading || window.start === 0} onClick={() => { void load(Math.max(0, window.start - window.size)); }}>上一段</button><button disabled={loading || window.start === lastStart} onClick={() => { void load(Math.min(lastStart, window.start + window.size)); }}>下一段</button><button disabled={loading || window.start === lastStart} onClick={() => { void load(lastStart); }}>最後</button></div><form className="px-time-jump" onSubmit={e => { e.preventDefault(); jump(String(new FormData(e.currentTarget).get("time") ?? "")); }}><label htmlFor="prediction-time">跳到時間<input id="prediction-time" name="time" type="datetime-local" step={600} min={meta.start.slice(0, 16)} max={meta.end.slice(0, 16)} key={`${window.start}-${window.size}`} defaultValue={jumpTime} required/></label><button type="submit" disabled={loading}>前往</button></form></div>
        <div className="px-window-status" role="status" aria-live="polite"><strong>{time(window.rows[0].timestamp)} → {time(window.rows.at(-1)!.timestamp)}</strong><span>Holdout 第 {window.rows[0].position.toLocaleString("en-US")}～{window.rows.at(-1)!.position.toLocaleString("en-US")} 筆／共 {meta.totalRows.toLocaleString("en-US")} 筆{loading ? " · 載入中…" : ""}</span></div>
        {error && <p className="px-error" role="alert">{error} 目前仍顯示上一個成功載入的視窗。</p>}
        <div className="px-chart-controls"><div className="segmented-control px-models" role="group" aria-label="時間圖模型">{(["both", "baseline", "selected"] as const).map(id => <button key={id} aria-pressed={models === id} className={models === id ? "selected" : ""} onClick={() => selectModels(id)}>{id === "both" ? "兩模型" : labels[id]}</button>)}</div><div className="segmented-control px-mode" role="group" aria-label="時間圖內容"><button aria-pressed={chartMode === "prediction"} className={chartMode === "prediction" ? "selected" : ""} onClick={() => setChartMode("prediction")}>預測與區間</button><button aria-pressed={chartMode === "residual"} className={chartMode === "residual" ? "selected" : ""} onClick={() => setChartMode("residual")}>殘差</button></div><label className="px-band-control" htmlFor="prediction-interval">區間陰影<select id="prediction-interval" value={interval} disabled={chartMode === "residual"} onChange={e => setInterval(e.target.value as typeof interval)}><option value="none">不顯示</option>{visible.map(m => <option key={m.id} value={m.id}>{m.label} · {(meta.nominalCoverage * 100).toFixed(0)}% PI</option>)}</select></label></div>
        <div className="chart-legend px-legend">{chartMode === "prediction" && <span><i className="legend-dot px-actual"/>實際值</span>}{visible.map(m => <span key={m.id}><i className={`legend-dot ${m.id}`}/>{m.label}</span>)}<span>{chartMode === "residual" ? "殘差 = 實際 − 預測（Wh）" : "耗能／區間（Wh）"}</span></div>
        <div className="px-chart" role="img" aria-label={`${chartMode === "prediction" ? "實際值、模型預測與所選 PI" : "兩模型殘差"}時間圖；每點均為保存的真實觀測，可在下方選取單筆查看`}><ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 750, height: 330 }}><ComposedChart data={chartRows} margin={{ top: 15, right: 20, bottom: 15, left: 0 }} accessibilityLayer onClick={state => { const row = window.rows.find(r => r.timestamp === state.activeLabel); if (row) choose(row.position); }}><CartesianGrid vertical={false} stroke="var(--chart-grid)" strokeDasharray="3 4"/><XAxis dataKey="timestamp" minTickGap={40} interval="preserveStartEnd" tickFormatter={s => String(s).slice(5, 10) + " " + String(s).slice(11, 16)} tick={{ fill: "var(--muted)", fontSize: 10 }}/><YAxis domain={["auto", "auto"]} width={55} tickFormatter={v => Number(v).toLocaleString("en-US", { maximumFractionDigits: 1 })} tick={{ fill: "var(--muted)", fontSize: 10 }}/><Tooltip content={({ active: show, label }) => { const row = window.rows.find(r => r.timestamp === label); return show && row ? <div className="px-tooltip"><strong>第 {row.position} 筆 · {time(row.timestamp)}</strong><p>實際值：{format(row.actual)} Wh</p>{visible.map(m => <div key={m.id}><p style={{ color: color(m.id) }}>{m.label}：{format(chartMode === "prediction" ? row[m.id].prediction : row.actual - row[m.id].prediction)} Wh</p>{chartMode === "prediction" && <p>殘差：{format(row.actual - row[m.id].prediction)} Wh</p>}</div>)}{chartMode === "prediction" && interval !== "none" && <p>{labels[interval]} PI：[{format(row[interval].lower)}, {format(row[interval].upper)}] Wh · {covered(row, interval) ? "涵蓋" : "未涵蓋"}</p>}</div> : null; }}/>
          {chartMode === "prediction" && interval !== "none" && <Area type="linear" dataKey="band" name="95% Prediction Interval" stroke="none" fill={color(interval)} fillOpacity={0.16} isAnimationActive={false}/>}
          {chartMode === "prediction" && <Line type="linear" dataKey="actual" name="實際值" stroke="var(--text)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false}/>}
          {visible.map(m => <Line key={m.id} type="linear" dataKey={chartMode === "prediction" ? `${m.id}Prediction` : `${m.id}Residual`} name={m.label} stroke={color(m.id)} strokeWidth={1.8} dot={false} activeDot={{ r: 4 }} isAnimationActive={false}/>)}
          {chartMode === "residual" && <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="4 4"/>}<ReferenceLine x={active.timestamp} stroke="var(--accent)" strokeDasharray="3 3"/>
        </ComposedChart></ResponsiveContainer></div>
        <p className="fs-note">預測估計的是同時間能耗，並非多步未來預報。時間按來源顯示，原始時區未註明。區間下界保留負值；單筆 PI 不是平均值 CI 或同時涵蓋整條曲線的保證。</p>
      </div>
    </section>

    <section className="panel px-interpretation" aria-labelledby="window-reading-title"><div className="panel-header"><div><div className="section-kicker">UNDERSTAND THE WINDOW</div><h2 id="window-reading-title">目前視窗怎麼看？</h2><p>隨視窗更新；使用兩模型的全部觀測，與上方顯示選項分開。</p></div><span className="neutral-tag">{window.rows.length} 筆 · 描述性</span></div><div className="conclusion-body">
      <p className="window-peak">此視窗實際最大值為 <strong>{format(peak.actual, 2)} Wh</strong>，出現在 {time(peak.timestamp)}（第 {peak.position.toLocaleString("en-US")} 筆）；下列預測皆對應這一筆。</p>
      <div className="window-reading-grid">{meta.models.map(m => { const stats = window.stats[m.id]; const residual = peak.actual - peak[m.id].prediction; return <article key={m.id}><h3><span className={`legend-dot ${m.id}`}/>{m.label}</h3><p>最大值當下預測 <strong>{format(peak[m.id].prediction, 2)} Wh</strong>，{residual > 0 ? "低估" : residual < 0 ? "高估" : "誤差為"} {format(Math.abs(residual), 2)} Wh；{covered(peak, m.id) ? "落在" : "未落在"}該筆 PI 內。</p><p>視窗 R²：<strong>{stats.r2 === null ? "無法定義" : format(stats.r2, 3)}</strong>。{explainR2(stats.r2)}</p><p>視窗 PI 涵蓋 {stats.covered}／{window.rows.length} 筆（{format(stats.coverage * 100, 2)}%），平均寬度 {format(stats.averageWidth, 2)} Wh；實際值高於上界 {stats.above} 筆、低於下界 {stats.below} 筆。</p></article>; })}</div>
      <p className="conclusion-scope">最大值的單筆誤差不能代表整個視窗；局部 R² 與覆蓋率也不能取代完整 Holdout。名目 {(meta.nominalCoverage * 100).toFixed(0)}% PI 不保證每一個短視窗都達到相同覆蓋率，亦不等於預測準確率。</p>
    </div></section>

    <section className="panel px-detail" aria-labelledby="observation-title"><div className="panel-header"><div><div className="section-kicker">ONE OBSERVATION, TWO MODELS</div><h2 id="observation-title">單筆預測明細</h2><p>點圖表、表格時間，或從選單選取觀測。</p></div><label htmlFor="prediction-row" className="px-row-picker">選取觀測<select id="prediction-row" value={active.position} onChange={e => choose(Number(e.target.value))}>{window.rows.map(r => <option key={r.position} value={r.position}>第 {r.position} 筆 · {time(r.timestamp)}</option>)}</select></label></div><div className="px-section-body"><div className="px-observation"><span>Holdout 第 {active.position.toLocaleString("en-US")} 筆 · {time(active.timestamp)}</span><strong>實際值 {format(active.actual)} Wh</strong></div><div className="fs-table-scroll"><table className="metrics-table px-detail-table"><caption>殘差為正代表低估；PI 以閉區間判定是否涵蓋</caption><thead><tr><th>模型</th><th>預測 (Wh)</th><th>殘差 (Wh)</th><th>絕對誤差 (Wh)</th><th>PI 下界 (Wh)</th><th>PI 上界 (Wh)</th><th>PI 判定</th></tr></thead><tbody>{meta.models.map(m => <tr key={m.id}><th scope="row">{m.label}</th><td>{format(active[m.id].prediction, 6)}</td><td>{format(active.actual - active[m.id].prediction, 6)}</td><td>{format(Math.abs(active.actual - active[m.id].prediction), 6)}</td><td>{format(active[m.id].lower, 6)}</td><td>{format(active[m.id].upper, 6)}</td><td><span className={covered(active, m.id) ? "px-covered" : "px-missed"}>{covered(active, m.id) ? "涵蓋" : active.actual > active[m.id].upper ? "高於上界" : "低於下界"}</span></td></tr>)}</tbody></table></div></div></section>

    <section className="panel px-window-metrics"><div className="panel-header"><div><div className="section-kicker">CURRENT WINDOW ONLY</div><h2>目前視窗的描述統計</h2><p>僅使用目前 {window.rows.length} 筆；切換時段會改變，不能取代完整 Holdout 成績。</p></div><span className="neutral-tag">探索性</span></div><div className="px-section-body"><div className="fs-table-scroll"><table className="metrics-table px-stat-table"><caption>目前視窗 · RMSE／MAE／寬度單位為 Wh；Coverage 為實測比例</caption><thead><tr><th>模型</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Coverage</th><th>平均 PI 寬度</th><th>區間外筆數</th></tr></thead><tbody>{meta.models.map(m => { const s = window.stats[m.id]; return <tr key={m.id}><th scope="row">{m.label}</th><td>{format(s.rmse)}</td><td>{format(s.mae)}</td><td>{s.r2 === null ? "無法定義" : format(s.r2)}</td><td>{format(s.coverage * 100)}%</td><td>{format(s.averageWidth)}</td><td>{s.above + s.below}／{window.rows.length}</td></tr>; })}</tbody></table></div><p className="fs-note">R² 在樣本不足或實際值為常數時顯示「無法定義」。預設前 {meta.defaultSize} 筆沿用 Phase 3C 預先固定的展示視窗；切換其他時段屬描述性探索，不據此選模或調整區間。最後一段為維持所選筆數，可能與前一段重疊。</p></div></section>

    <section className="panel px-records"><div className="panel-header"><div><div className="section-kicker">MEASURED VALUES, NOT SMOOTHED SAMPLES</div><h2>目前視窗逐筆數值</h2><p>以每頁 24 筆呈現；圖表使用整個視窗的所有觀測，沒有抽樣或平滑。</p></div></div><div className="px-section-body"><div className="fs-table-scroll"><table className="metrics-table px-record-table"><caption>時間與實際／預測值（Wh）；點時間查看完整區間</caption><thead><tr><th>Holdout 筆次</th><th>時間</th><th>實際值</th><th>Baseline 預測</th><th>MI 預測</th></tr></thead><tbody>{pageRows.map(r => <tr key={r.position} className={active.position === r.position ? "px-selected-row" : ""}><th scope="row">{r.position.toLocaleString("en-US")}</th><td><button aria-pressed={active.position === r.position} aria-label={`查看第 ${r.position} 筆`} onClick={() => choose(r.position)}>{time(r.timestamp)}</button></td><td>{format(r.actual)}</td><td>{format(r.baseline.prediction)}</td><td>{format(r.selected.prediction)}</td></tr>)}</tbody></table></div><div className="px-pagination"><button disabled={page === 0} onClick={() => setPage(page - 1)}>上一頁</button><span>第 {page * 24 + 1}～{Math.min((page + 1) * 24, window.rows.length)} 筆／視窗共 {window.rows.length} 筆</span><button disabled={(page + 1) * 24 >= window.rows.length} onClick={() => setPage(page + 1)}>下一頁</button></div></div></section>
  </>;
}
