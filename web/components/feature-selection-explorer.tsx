"use client";

import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";
import { METHOD_NAMES, type FeatureData, type MethodId, type Metric, type Scores } from "@/lib/feature-selection-types";

const descriptions: Record<MethodId, string> = {
  f_test: "以各特徵與目標的線性關聯評分，選前 k 個；相關不代表因果。",
  mutual_information: "以互資訊捕捉特徵與目標的依賴關係；分數不代表因果效果。",
  forward_sfs: "逐步加入特徵，內部時間 CV 評估候選；只呈現實際測試的稀疏 k grid。",
  rfe: "依標準化後 Linear Regression 係數逐步刪除特徵，保留指定 k。",
  lasso_cv: "時間 CV 選 alpha，再用非零係數決定 k；alpha 依平均 MSE 選擇。最終預測仍用獨立 Linear Regression。",
  rfecv: "透過內部時間 CV 自動選 k，並逐步刪除特徵；不同 fold 可能選到不同 k。",
};
const fmt = (n: number, digits = 4) => n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const label = (m: Metric) => m === "r2" ? "R²" : m.toUpperCase();
function ScoreCells({ scores }: { scores: Scores }) { return <><td>{fmt(scores.rmse)}</td><td>{fmt(scores.mae)}</td><td>{fmt(scores.r2)}</td></>; }
const tooltipStyle = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--text)" };

export function FeatureSelectionExplorer({ data }: { data: FeatureData }) {
  const [methodId, setMethodId] = useState<MethodId>(data.winner);
  const [metric, setMetric] = useState<Metric>("rmse");
  const [k, setK] = useState(data.methods.find(m => m.id === data.winner)!.finalK);
  const [evaluation, setEvaluation] = useState<"nested" | "selection">("nested");
  const method = data.methods.find(m => m.id === methodId)!;
  const chosen = method.curve.find(p => p.k === k);
  const meanOuterK = method.outerK.reduce((a, b) => a + b, 0) / method.outerK.length;
  const unit = metric === "r2" ? "" : " Wh";
  const rows = method.curve.map(p => ({ k: p.k, value: p.scores?.[metric] ?? null }));
  return <>
    <section className="panel fs-explorer" aria-labelledby="explorer-title">
      <div className="panel-header"><div><div className="section-kicker">EXPLORE THE EVIDENCE</div><h2 id="explorer-title">方法與特徵數量</h2><p>只瀏覽既有成果；操作不會啟動訓練。</p></div><span className="neutral-tag">Phase 3B</span></div>
      <div className="fs-methods" role="group" aria-label="特徵選擇方法">{data.methods.map(m => <button key={m.id} aria-pressed={methodId === m.id} onClick={() => { setMethodId(m.id); setK(m.finalK); }} className={methodId === m.id ? "active" : ""}>{METHOD_NAMES[m.id]}<small>{m.automatic ? "Auto k" : `${m.curve.length} measured k`}</small></button>)}</div>
      <div className="fs-explorer-body">
        <div className="fs-method-description"><h3>{METHOD_NAMES[method.id]}</h3><p>{descriptions[method.id]}</p></div>
        <div className="chart-controls"><span className="fs-chart-level">{method.automatic ? "Nested CV · 各 Outer fold 實際評分" : `固定 k · ${data.outerFolds}-fold 時間 CV 曲線`}</span><div className="segmented-control" role="group" aria-label="特徵選擇評估指標">{(["rmse", "mae", "r2"] as const).map(m => <button key={m} aria-pressed={metric === m} className={metric === m ? "selected" : ""} onClick={() => setMetric(m)}>{label(m)}</button>)}</div></div>
        {method.automatic && <div className="fs-fold-selection" aria-label="各 Outer fold 實際選中特徵數">
          <p className="fs-fold-heading">各 Outer fold 實際選中的特徵數</p>
          <ol className="fs-fold-cards">{method.outerResults.map(result => <li key={result.fold}><span>Fold {result.fold}</span><strong>{result.k}<small> 個特徵</small></strong><span className="fs-fold-score">{label(metric)} {fmt(result.scores[metric])}{unit}</span></li>)}</ol>
          <p className="fs-note">每個 fold 都有真實的選欄數與驗證分數；下方按時間順序連接五筆實測結果，虛線是 Outer 平均。</p>
        </div>}
        <p className="fs-axis-note">{label(metric)}{unit} · {metric === "r2" ? "越高越好，可能為負值" : "越低越好"}</p>
        <div className="fs-chart" role="img" aria-label={`${METHOD_NAMES[method.id]} ${label(metric)} ${method.automatic ? "各 Outer fold 的實際評分折線；每點對應該 fold 的整數 k" : "固定 k 實測曲線"}；完整數值見下方資料表`}>
          <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 650, height: 270 }}>
            {method.automatic ? <LineChart data={method.outerResults.map(result => ({ fold: result.fold, k: result.k, value: result.scores[metric] }))} margin={{ top: 20, right: 20, bottom: 25, left: 5 }} accessibilityLayer>
              <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 4"/>
              <XAxis type="number" dataKey="fold" domain={[1, data.outerFolds]} ticks={method.outerResults.map(r => r.fold)} tickFormatter={fold => `Fold ${fold}`} tick={{ fill: "var(--muted)", fontSize: 11 }} label={{ value: "Outer fold（時間順序）", position: "bottom", fill: "var(--muted)", fontSize: 11 }}/>
              <YAxis type="number" domain={["auto", "auto"]} tick={{ fill: "var(--muted)", fontSize: 11 }} width={65}/>
              <Tooltip content={({ active, label: foldLabel }) => {
                const result = method.outerResults.find(r => r.fold === Number(foldLabel));
                return active && result ? <div className="fs-auto-tooltip" style={tooltipStyle}><strong>{METHOD_NAMES[method.id]} · Fold {result.fold}</strong><p>實際選中特徵：{result.k} 個</p><p>{label(metric)}：{fmt(result.scores[metric], 6)}{unit}</p><p>Outer 平均：{fmt(method.nestedCV[metric], 6)}{unit}</p></div> : null;
              }}/>
              <ReferenceLine y={method.nestedCV[metric]} stroke="var(--muted)" strokeDasharray="5 5"/>
              <Line type="linear" dataKey="value" name={label(metric)} stroke="var(--selected)" strokeWidth={2} dot={{ r: 4, strokeWidth: 1 }} activeDot={{ r: 6 }} isAnimationActive={false}/>
            </LineChart> : <LineChart data={rows} margin={{ top: 15, right: 20, bottom: 25, left: 5 }} accessibilityLayer>
              <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 4"/>
              <XAxis type="number" dataKey="k" domain={[1, data.candidateFeatures]} ticks={method.curve.map(p => p.k)} minTickGap={12} tick={{ fill: "var(--muted)", fontSize: 11 }} label={{ value: "指定 k（特徵數）", position: "bottom", fill: "var(--muted)", fontSize: 11 }}/>
              <YAxis domain={["auto", "auto"]} width={65} tick={{ fill: "var(--muted)", fontSize: 11 }}/>
              <Tooltip contentStyle={tooltipStyle} labelFormatter={v => `k = ${v}`} formatter={v => [`${fmt(Number(v), 6)}${unit}`, label(metric)]}/>
              <Line type="linear" dataKey="value" stroke="var(--selected)" strokeWidth={2} dot={{ r: 3, strokeWidth: 1 }} activeDot={{ r: 5 }} connectNulls={false} isAnimationActive={false}/>
            </LineChart>}
          </ResponsiveContainer>
        </div>
        {method.automatic ? <div className="fs-point-summary" aria-live="polite"><strong>完整 Training 最佳 k = {method.finalK}</strong><span>Outer folds 平均 k = ({method.outerK.join(" + ")}) ÷ {method.outerK.length} = {fmt(meanOuterK, 2)}</span><span>{label(metric)} Outer 平均（虛線）：{fmt(method.nestedCV[metric], 6)}{unit}</span><p>平均 k 是統計摘要。圖中每個點則是該 fold 的實際評分；兩次選到相同 k 仍是不同時段的評估。完整 Training 的最終 k 另行標示。</p></div> : <div className="fs-k-control"><label htmlFor="measured-k">檢視實測 k</label><select id="measured-k" value={k} onChange={e => setK(Number(e.target.value))}>{method.curve.map(p => <option key={p.k} value={p.k}>k = {p.k}</option>)}</select><output aria-live="polite">{chosen?.scores ? `${label(metric)} = ${fmt(chosen.scores[metric], 6)}${unit}` : "此點 folds 不完整，無可比較分數"}</output></div>}
        <p className="fs-note">{method.automatic ? "此折線的橫軸是 fold，不是 k。各 fold 的訓練／驗證時段不同，因此不能將高低變化解讀為增加特徵的效果，也沒有補造 k=1～25 的固定 k 曲線。" : method.id === "forward_sfs" ? "圓點為實測結果，連線僅引導閱讀；未測的 k 沒有數值，也不能選取。" : "曲線是固定 k 的 Outer CV 平均。查看曲線後挑出的最小值，不能當作獨立泛化驗證。"}</p>
        <details className="fs-data-table"><summary>查看{method.automatic ? "每個 fold 的實際" : "全部實測 k"}數值</summary><div className="fs-table-scroll"><table className="metrics-table"><caption>{method.automatic ? "Nested CV 各 Outer fold 的實際 k、RMSE、MAE、R²；平均以各 fold 等權計算" : "預先指定 k 的時間 CV；不是 Holdout"}</caption><thead><tr><th>{method.automatic ? "Fold／實際 k" : "指定 k"}</th><th>RMSE (Wh)</th><th>MAE (Wh)</th><th>R²</th><th>{method.automatic ? "資料類型" : "有效 folds"}</th></tr></thead><tbody>{method.automatic ? <>{method.outerResults.map(result => <tr key={result.fold}><th>Fold {result.fold} · k = {result.k}</th><ScoreCells scores={result.scores}/><td>實測</td></tr>)}<tr><th>Outer 平均</th><ScoreCells scores={method.nestedCV}/><td>{data.outerFolds}/{data.outerFolds} folds</td></tr></> : method.curve.map(p => <tr key={p.k}><th>{p.k}</th>{p.scores ? <ScoreCells scores={p.scores}/> : <td colSpan={3}>無完整有效分數</td>}<td>{p.validFolds}/{data.outerFolds}</td></tr>)}</tbody></table></div></details>
      </div>
    </section>

    <div className="fs-detail-grid">
      <section className="panel fs-selection" aria-labelledby="selected-features-title"><div className="panel-header"><div><div className="section-kicker">FINAL TRAINING SELECTION</div><h2 id="selected-features-title">選中的特徵</h2><p>{METHOD_NAMES[method.id]} · 完整 Training 最終重 fit</p></div><span className="neutral-tag">k = {method.finalK}</span></div><div className="fs-section-body"><div className="feature-chips">{method.features.map((name, i) => <span key={name} title={method.transformedFeatures[i]}>{name}</span>)}</div><p className="fs-note">這份清單固定對應最終 k = {method.finalK}，與上方正在檢視的曲線 k 分開。CV 的每個 fold 會各自重新選欄，組合可能不同。</p><details><summary>查看轉換後欄位名稱</summary><p className="fs-transformed">{method.transformedFeatures.join(", ")}</p></details></div></section>
      <section className="panel"><div className="panel-header"><div><div className="section-kicker">HOW TO READ K</div><h2>同一方法，不同階段</h2><p>每個分數都要對照評估層級。</p></div></div><dl className="fs-k-facts"><div><dt>完整 Training 最終 k</dt><dd>{method.finalK}</dd></div><div><dt>{data.innerFolds}-fold 最終選模 CV 平均 k</dt><dd>{fmt(method.meanFinalK, 2)}</dd></div><div><dt>Nested CV 各 Outer fold k</dt><dd>{method.outerK.join(" / ")}</dd></div></dl><p className="fs-note fs-inset">k 計算前處理後、送入獨立 Linear Regression 的數值欄位；本資料為一對一 numeric 轉換。</p></section>
    </div>

    <section className="panel fs-comparison" aria-labelledby="method-comparison-title"><div className="panel-header"><div><div className="section-kicker">SIX METHODS · ONE EVALUATION LEVEL</div><h2 id="method-comparison-title">六種方法比較</h2><p>{evaluation === "nested" ? `${data.outerFolds} 個 Outer folds；固定 k 方法在 ${data.innerFolds}-fold Inner CV 挑 k，自動方法使用內部 CV。` : `完整 Training 上以 ${data.innerFolds}-fold CV 選最終設定；數值已參與選模。`}</p></div></div><div className="fs-section-body"><div className="segmented-control fs-evaluation-control" role="group" aria-label="方法比較評估層級"><button aria-pressed={evaluation === "nested"} className={evaluation === "nested" ? "selected" : ""} onClick={() => setEvaluation("nested")}>Nested CV</button><button aria-pressed={evaluation === "selection"} className={evaluation === "selection" ? "selected" : ""} onClick={() => setEvaluation("selection")}>最終選模 CV</button></div><div className="fs-table-scroll"><table className="metrics-table"><caption>{evaluation === "nested" ? "Nested Outer 評估（非 Holdout）" : "最終選模 CV（非獨立測試）"} · RMSE／MAE 越低越好，R² 越高越好</caption><thead><tr><th>方法</th><th>RMSE (Wh)</th><th>MAE (Wh)</th><th>R²</th><th>{evaluation === "nested" ? "Outer fold k" : "CV 平均 k"}</th></tr></thead><tbody>{data.methods.map(m => <tr key={m.id}><th>{METHOD_NAMES[m.id]}{m.id === data.winner && <small>正式選模結果</small>}</th><ScoreCells scores={evaluation === "nested" ? m.nestedCV : m.finalCV}/><td>{evaluation === "nested" ? m.outerK.join(" / ") : fmt(m.meanFinalK, 2)}</td></tr>)}</tbody></table></div><p className="fs-note">正式方法由最終選模 CV 的平均 RMSE 規則決定，並不隨此頁指標切換而更換。Outer 分數描述各方法的程序，不能視為「從六種方法挑冠軍」的無偏估計。</p></div></section>
  </>;
}
