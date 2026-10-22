"use client";
import { useCustomCompute } from "./custom-compute-provider";
import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { isAutoMethod, isNestedMethod } from "@/lib/custom-selection-policy";
import { CustomSelectionDetails } from "./custom-selection-details";
import { trainingSelectionError } from "@/lib/custom-training-choice";
import type { ComparisonMethod } from "@/lib/custom-comparison-result";
import { CustomComparisonPanel } from "./custom-comparison-panel";
import { CustomResultDownload } from "./custom-result-download";
import { CustomPredictionForm } from "./custom-prediction-form";
import type { PreprocessingPlan } from "@/lib/custom-dataset";
import type { ValidationReport } from "@/lib/custom-validation-result";
import { parseTrainingReport, SELECTION_LABELS, type SelectionMethod, type Scores, type TrainingReport } from "@/lib/custom-training-result";
const format = (n: number | null, digits = 4) => n === null ? "無法定義" : n.toLocaleString("en-US", { maximumFractionDigits: digits });
const axisFormat = (n: number) => Math.abs(n) >= 1_000_000 || (n !== 0 && Math.abs(n) < 0.01) ? n.toExponential(1) : format(n, 2);
type Validated = Extract<ValidationReport, { status: "valid" }>;
type Completed = Extract<TrainingReport, { status: "completed" }>;
export function CustomTrainingPanel({ file, config, validation }: { file: File; config: PreprocessingPlan; validation: Validated }) {
  const compute=useCustomCompute();
  const [selectionMethod, setSelectionMethod] = useState<SelectionMethod>("none");
  const [k, setK] = useState(String(Math.min(6,config.feature_columns.length)));
  const [appliedChoice,setAppliedChoice] = useState<string|null>(null);
  const [comparisonBusy,setComparisonBusy] = useState(false);
  const [holdoutViewed,setHoldoutViewed] = useState(false);
  const requestedK=(selectionMethod === "none" || isAutoMethod(selectionMethod)) ? null : Number(k);
  const validK=selectionMethod === "none" || isAutoMethod(selectionMethod) || (/^[1-9]\d*$/.test(k) && Number.isSafeInteger(requestedK) && requestedK! <= config.feature_columns.length);
  const selectionError = validK ? trainingSelectionError(selectionMethod,requestedK,config,validation) : `k 必須是 1 到 ${config.feature_columns.length} 之間的整數。`;
  const [consent, setConsent] = useState(false); const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<TrainingReport | null>(null); const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null); const generation = useRef(0);
  useEffect(() => () => { generation.current++; controller.current?.abort(); }, []);
  function cancel() { generation.current++; controller.current?.abort(); controller.current = null; setBusy(false); setError("已取消本次訓練；不會顯示未完成的成績。"); }
  function resetExperiment() { compute.clearModels(); generation.current++; controller.current?.abort(); controller.current=null; setBusy(false); setConsent(false); setResult(null); setError(null); setAppliedChoice(null); }
  function applyChoice(method:ComparisonMethod,nextK:number|null) {
    if(busy||comparisonBusy||trainingSelectionError(method,nextK,config,validation))return;
    resetExperiment();setSelectionMethod(method);if(nextK!==null)setK(String(nextK));
    setAppliedChoice(`已套用 ${SELECTION_LABELS[method]} · ${nextK===null?"自動選 k":`k = ${nextK}`}。這只帶入設定，尚未訓練；請確認後再開始。`);
    document.getElementById("custom-training-title")?.scrollIntoView({block:"start"});
  }
  async function train() {
    if (!consent || busy || comparisonBusy || !!selectionError) return;
    const token = ++generation.current; const abort = new AbortController(); controller.current = abort;
    setBusy(true); setResult(null); setError(null); setAppliedChoice(null); let expired = false;
    const timer = setTimeout(() => { expired = true; abort.abort(); }, 130_000);
    const body = new FormData(); body.append("csv", file); body.append("config", JSON.stringify(config));
    body.append("review", JSON.stringify({ schema_version: 1, artifact_type: "custom_dataset_review", dataset: { sha256: validation.dataset.sha256 }, preprocessing: config, review: { confirmed: true, training_started: false, scope: "configuration_only" } }));
    body.append("experiment", JSON.stringify({ schema_version: 1, method: "linear_regression", confirmed: true, dataset_sha256: validation.dataset.sha256, config_sha256: validation.config_sha256, selection: {method:selectionMethod,k:requestedK} }));
    try {
      const response = await compute.request("/api/custom-dataset/train", { method: "POST", body, signal: abort.signal, cache: "no-store" });
      const payload: unknown = await response.json().catch(() => { throw new Error("本機服務未回傳有效訓練結果。"); });
      if (![200,422].includes(response.status)) throw new Error(payload && typeof payload === "object" && "error" in payload && typeof payload.error === "string" ? payload.error : "本機訓練暫時無法完成，請稍後重試。");
      const report = parseTrainingReport(payload);
      if ((response.status === 200) !== (report.status === "completed")) throw new Error("訓練回應狀態不一致。");
      if (report.status === "completed" && (report.validation.dataset.sha256 !== validation.dataset.sha256 || report.validation.config_sha256 !== validation.config_sha256 || report.validation.dataset.rows !== validation.dataset.rows || report.target !== config.target_column || JSON.stringify(report.features) !== JSON.stringify(config.feature_columns) || report.selection.method !== selectionMethod || report.selection.requested_k !== requestedK)) throw new Error("訓練結果與目前資料或設定版本不一致，請重新驗證。");
      if (token === generation.current) { setResult(report); if (report.status === "completed") setHoldoutViewed(true); }
    } catch (e) {
      if (token === generation.current) setError(expired ? "訓練等待逾時，請縮小資料後重試。" : e instanceof Error && e.name !== "AbortError" ? e.message : "訓練已取消。");
    } finally { clearTimeout(timer); if (token === generation.current) { setBusy(false); controller.current = null; } }
  }
  return <><CustomComparisonPanel file={file} config={config} validation={validation} disabled={busy} holdoutViewed={holdoutViewed} onBusyChange={setComparisonBusy} onApply={applyChoice}/><section className="panel cd-training" aria-labelledby="custom-training-title" aria-busy={busy}><div className="panel-header"><div><div className="section-kicker">07 · EVALUATE YOUR CHOSEN SETTING</div><h2 id="custom-training-title">訓練與評估</h2><p>指定選欄方法與 k，比較選欄後、全部候選欄位及平均值基準。</p></div><span className="neutral-tag">{compute.mode==="browser"?"Browser 實驗":"Server 實驗"}</span></div><div className="cd-body">
    <div className="cd-training-plan"><strong>最終預測器：Linear Regression</strong><p>目標 {config.target_column} · {config.feature_columns.length} 個輸入 · {validation.split.training_rows.toLocaleString()} 筆 Training／{validation.split.holdout_rows.toLocaleString()} 筆 Holdout · {config.cv_splits} folds</p><p>每個 CV fold 分別學習前處理與選欄，最後只用完整 Training 重新 fit。方法由你指定；固定方法需指定 k，RFECV／LassoCV 則在 Training 的內部 3-fold CV 自動選 k／alpha。</p></div>
    <div className="cd-config-grid cd-selection-controls"><label htmlFor="dataset-selection-method">特徵選擇方法<select id="dataset-selection-method" value={selectionMethod} disabled={busy||comparisonBusy} onChange={e=>{resetExperiment();setSelectionMethod(e.target.value as SelectionMethod);}}><option value="none">不自動選欄 · 使用全部候選欄位</option><option value="f_test">F-test · 線性關聯</option><option value="mutual_information">Mutual Information · 統計依賴</option><option value="rfe">RFE · 遞迴消除</option><option value="forward_sfs">Forward SFS · 逐步加入，指定 k</option><option value="rfecv">RFECV · 自動選 k</option><option value="lasso_cv">LassoCV · 自動選 alpha 與 k</option></select></label>{selectionMethod!=="none"&&!isAutoMethod(selectionMethod)&&<label htmlFor="dataset-selection-k">保留特徵數 k（1～{config.feature_columns.length}）<input id="dataset-selection-k" type="number" min={1} max={config.feature_columns.length} step={1} value={k} disabled={busy||comparisonBusy} onChange={e=>{resetExperiment();setK(e.target.value);}}/></label>}</div>
    <p className="cd-confirmation">候選欄位是你在上方勾選的 {config.feature_columns.length} 欄。F-test 看線性關聯；MI 看統計依賴，但選完後仍用線性迴歸，不保證更準。k 等於候選數時保留全部，不執行排名。RFE 依標準化線性係數逐步移除欄位，與上方曲線使用相同選欄流程。SFS 每次加入一個特徵；RFECV／LassoCV 自動選 k。這四種方法需 StandardScaler，內層 CV 沿用隨機或時間切分及 gap。{compute.mode==="browser"?"Browser 訓練資料上限 2,000 筆、15 個候選特徵、3 MiB；仍須通過所選方法的原有工作量保護。上方比較超限不影響合法訓練。":"Server Mode：RFE／RFECV／LassoCV 限 5,000 筆與 25 欄，SFS 限 2,000 筆與 15 欄。"}</p>
    {isAutoMethod(selectionMethod)&&<p id="custom-auto-k-guide" className="cd-chart-guide">此方法會在每個外層 Training 內自行選 k；不同 CV fold 可能選出不同數量，最終模型也會在完整 Training 再選一次。你不需要輸入固定 k。LassoCV 若選出 0 欄，會停止該次正式訓練，不強制補回特徵或改用其他模型。</p>}
    {selectionError&&<p className="cd-error" role="alert">訓練限制：{selectionError}</p>}{appliedChoice&&<p id="comparison-applied-choice" className="cd-validation" role="status">{appliedChoice}</p>}
    {holdoutViewed&&<p className="cd-validation cd-cautions">你已查看這份資料的 Holdout；改方法或 k 再執行屬於重複使用測試集，不能把最低 Holdout 誤差當成獨立驗證。</p>}
    <label className="cd-training-consent"><input id="dataset-training-consent" type="checkbox" checked={consent} disabled={busy||comparisonBusy} onChange={e => setConsent(e.target.checked)}/><span>我確認使用目前資料、方法與設定開始訓練，並了解查看 Holdout 後再調整設定，會重複使用測試集。</span></label>
    <div className="cd-download-actions"><button id="dataset-train" className="cd-button cd-primary" disabled={!consent || busy || comparisonBusy || !!selectionError} onClick={() => { void train(); }}>{busy ? "訓練與評估中…" : "開始訓練與評估"}</button>{busy && <button id="dataset-train-cancel" className="cd-button" onClick={cancel}>取消訓練</button>}</div>
    <p className="cd-confirmation">{compute.mode==="browser"?"CSV 與已確認設定只在瀏覽器處理，不會上傳。每方法上限 60 秒、工作上限 90 秒；結果與最多 3 個暫存模型留在瀏覽器記憶體，模型約 30 分鐘到期。取消、逾時、換檔、改設定、離頁或切換模式會清除模型。":"按下訓練會將 CSV 與已確認設定送至運算服務，完成後清除上傳暫存檔；運算上限 120 秒。結果留在本頁記憶體，最終模型另在服務私有暫存約 30 分鐘，最多保留 3 個，可手動清除。"}</p>
    {busy && <p role="status" className="cd-confirmation">正在訓練與評估；流程包含逐 fold CV 與保留測試集評估…</p>}{error && <p className="cd-error" role="alert">{error}</p>}
    {result && <div className="cd-training-result" role="status" aria-live="polite">{result.status === "completed" ? <TrainingResults report={result} config={config} comparisonBusy={comparisonBusy}/> : <div className="cd-validation"><h3>{result.training_started ? "訓練未完成" : "尚未開始訓練"}</h3><ul>{result.errors.map((e,i)=><li key={i}>{e.message}</li>)}</ul><p>不顯示部分分數；請檢查設定後重新驗證。</p></div>}</div>}
  </div></section></>;
}
function ScoreCells({ scores }: { scores: Scores }) { return <><td>{format(scores.rmse)}</td><td>{format(scores.mae)}</td><td>{format(scores.r2)}</td></>; }
function TrainingResults({ report, config, comparisonBusy }: { report: Completed; config: PreprocessingPlan; comparisonBusy:boolean }) {
  const [metric, setMetric] = useState<keyof Scores>("rmse"); const [page, setPage] = useState(0);
  const [visibleSeries,setVisibleSeries] = useState({actual:true,prediction:true,baseline_prediction:true,training_mean:true});
  type SeriesKey=keyof typeof visibleSeries;
  const labels = { rmse: "RMSE", mae: "MAE", r2: "R²" }; const h=report.holdout;
  const hasSelection=report.selection.method!=="none";
  const modelLabel=hasSelection ? `${SELECTION_LABELS[report.selection.method]} + LR` : "全部候選欄位 + LR";
  const comparisonLabel=hasSelection ? "全部候選欄位 LR" : "Training 平均值";
  const modelNames: ("linear_regression"|"all_features"|"training_mean")[]=hasSelection ? ["linear_regression","all_features","training_mean"] : ["linear_regression","training_mean"];
  const nameOf=(name: typeof modelNames[number])=>name==="linear_regression"?modelLabel:name==="all_features"?"全部候選欄位 + LR":"Training 平均值基準";
  const chartSeries: {key:SeriesKey;label:string;color:string}[]=[
    ...(report.validation.split.strategy==="chronological"?[{key:"actual" as const,label:"實際值",color:"var(--text)"}]:[]),
    {key:"prediction",label:modelLabel,color:"var(--baseline)"},
    {key:"training_mean",label:"Training 平均值",color:"var(--selected)"},
    ...(hasSelection?[{key:"baseline_prediction" as const,label:"全部候選欄位 + LR",color:"var(--muted)"}]:[]),
  ];
  const allSeriesHidden=chartSeries.every(s=>!visibleSeries[s.key]);
  const lr=h.linear_regression[metric], mean=(hasSelection ? h.all_features : h.training_mean)[metric];
  const better=lr!==null&&mean!==null ? (metric === "r2" ? lr>mean : lr<mean) : null;
  const equal=lr!==null&&mean!==null&&lr===mean;
  const chronological=report.validation.split.strategy === "chronological";
  const rows=report.preview.slice(page*24,(page+1)*24);
  const extent=report.preview.flatMap(r=>[r.actual,r.prediction,r.baseline_prediction,r.training_mean]);const low=Math.min(...extent),high=Math.max(...extent);
  return <><div className="cd-python-passed"><h3>本次訓練與評估完成</h3><p>以下是這份上傳資料的真實結果。Holdout 共 {h.rows.toLocaleString()} 筆，全數納入評分；此測試集已查看。</p></div>
    <div className="cd-selection-summary"><h3>{SELECTION_LABELS[report.selection.method]} · {report.features.length} 個候選 → {report.selection.final_features.length} 個輸入</h3><p>完整 Training 的{report.selection.ranking_performed?"最終選欄":"使用欄位"}：{report.selection.final_features.join("、")}</p>{report.selection.details&&<CustomSelectionDetails details={report.selection.details}/>}<p>此清單屬於完整 Training；各 CV fold 自己選欄，結果可以不同，不是拿最終清單回頭評分。</p><details><summary>查看各 CV fold 選中的欄位</summary><div className="fs-table-scroll"><table className="metrics-table cd-fold-features"><caption>Selector 只 fit 當前 fold 的 Training；欄位依原候選順序列出</caption><thead><tr><th>Fold</th><th>k</th><th>實際欄位</th><th>alpha</th></tr></thead><tbody>{report.selection.fold_features.map(f=><tr key={f.fold}><th>Fold {f.fold}</th><td>{f.features.length}</td><td>{f.features.join("、")}</td><td>{f.details?.alpha??"—"}</td></tr>)}</tbody></table></div></details></div>
    {report.prediction_model&&<CustomPredictionForm key={report.prediction_model.id} model={report.prediction_model} ranges={report.prediction_input_ranges} paused={comparisonBusy}/>}
    <div className="chart-controls cd-training-controls"><span>保留測試集 · {metric === "r2" ? "越高越好，不是準確率" : "越低越好，使用目標欄位原單位"}</span><div className="segmented-control" role="group" aria-label="自訂資料評估指標">{(["rmse","mae","r2"] as const).map(m=><button key={m} aria-pressed={metric===m} className={metric===m?"selected":""} onClick={()=>setMetric(m)}>{labels[m]}</button>)}</div></div>
    <div className={`cd-stats cd-training-stats ${hasSelection?"cd-selection-comparison":""}`}><div><strong>{format(lr)}</strong><span>{modelLabel} · {labels[metric]}</span></div>{hasSelection&&<div><strong>{format(h.all_features[metric])}</strong><span>全部候選欄位 + LR · {labels[metric]}</span></div>}<div><strong>{format(h.training_mean[metric])}</strong><span>訓練平均值基準 · {labels[metric]}</span></div></div>
    <p className="cd-training-reading">{better===null ? "這個指標無法定義，請參考 RMSE／MAE。" : equal ? `此指標下，模型與${comparisonLabel}相同。` : better ? `此指標下，模型優於${comparisonLabel}。` : `此指標下，模型尚未勝過${comparisonLabel}。`} 平均值只由 Training 計算；R² 的比較均值則依各評估集定義。</p>
    <div className="fs-table-scroll"><table className="metrics-table cd-training-metrics"><caption>完整 Holdout · 原目標單位 · 不套用既有能源研究成績</caption><thead><tr><th>方法</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>{modelNames.map(name=><tr key={name}><th>{nameOf(name)}</th><ScoreCells scores={h[name]}/></tr>)}</tbody></table></div>
    <h3 className="cd-subheading">Training 區內 CV · 與 Holdout 分開</h3><div className="fs-table-scroll"><table className="metrics-table cd-cv-summary"><caption>每 fold 等權平均 ± 標準差（ddof=0）；{isNestedMethod(report.selection.method)?"外層分數：每個 fold 內另做 3-fold 選欄；不是信賴區間":"未進行內層選模；不是信賴區間"}</caption><thead><tr><th>方法</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>{modelNames.map(name=><tr key={name}><th>{nameOf(name)}</th>{(["rmse","mae","r2"] as const).map(m=><td key={m}>{format(report.cv[name][m].mean)}{report.cv[name][m].std!==null&&<> ± {format(report.cv[name][m].std)}</>}<small className="cd-defined-folds">{report.cv[name][m].defined_folds}/{report.cv.folds.length} folds 有定義</small></td>)}</tr>)}</tbody></table></div>
    <details className="cd-cv-details"><summary>查看各 fold 的實際成績</summary><div className="fs-table-scroll"><table className="metrics-table"><caption>僅 Training 區內 · 每個 fold 重新 fit 前處理與模型</caption><thead><tr><th>Fold</th><th>方法</th><th>訓練／驗證筆數</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>{report.cv.folds.flatMap(f=>modelNames.map(name=><tr key={`${f.fold}-${name}`}><th>Fold {f.fold}</th><td>{nameOf(name)}</td><td>{f.training_rows.toLocaleString()}／{f.validation_rows.toLocaleString()}</td><ScoreCells scores={f[name]}/></tr>))}</tbody></table></div></details>
    <h3 id="custom-holdout-chart-title" className="cd-subheading">{chronological ? "Holdout 前段的實際值與預測" : "Holdout 實際值與預測散佈"}</h3><p className="cd-confirmation">僅預覽前 {report.preview.length}／{h.rows.toLocaleString()} 筆；{chronological ? "依時間順序，沒有平滑或抽樣補點。" : "橫軸為實際值、縱軸為預測值；虛線代表兩者相等。筆序沒有時間軸意義。"} 指標仍使用全部測試資料，不顯示 PI。</p>
    <figure className="cd-holdout-figure" aria-labelledby="custom-holdout-chart-title" aria-describedby="custom-holdout-chart-guide">
    <div className="cd-axis-caption cd-y-caption"><strong>Y 軸：{chronological ? "實際值與預測值" : "預測值"}</strong><span>目標：{report.target} · 原始單位</span></div>
    <div id="custom-holdout-series" className="cd-training-chart" role="img" aria-label={`${report.target}；${chronological ? "X 軸為依時間排序的 Holdout 筆次，Y 軸為實際值與預測值" : "X 軸為實際值，Y 軸為預測值，虛線代表兩者相等"}；完整預覽數值見下表`}><ResponsiveContainer width="100%" height="100%">{chronological ? <LineChart data={report.preview} margin={{top:15,right:15,bottom:10,left:0}}><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3"/><XAxis dataKey="position" tick={{fill:"var(--muted)",fontSize:11}}/><YAxis includeHidden domain={["auto","auto"]} tickFormatter={axisFormat} tick={{fill:"var(--muted)",fontSize:11}} width={76}/><Tooltip contentStyle={{background:"var(--surface)",borderColor:"var(--border)"}} labelFormatter={n=>`Holdout 第 ${n} 筆`}/><Line type="linear" dataKey="actual" hide={!visibleSeries.actual} name="實際值" stroke="var(--text)" dot={false} isAnimationActive={false}/><Line type="linear" dataKey="prediction" hide={!visibleSeries.prediction} name={modelLabel} stroke="var(--baseline)" dot={false} isAnimationActive={false}/>{hasSelection&&<Line type="linear" dataKey="baseline_prediction" hide={!visibleSeries.baseline_prediction} name="全部候選欄位 + LR" stroke="var(--muted)" strokeDasharray="3 3" dot={false} isAnimationActive={false}/>}<Line type="linear" dataKey="training_mean" hide={!visibleSeries.training_mean} name="Training 平均值" stroke="var(--selected)" strokeDasharray="5 4" dot={false} isAnimationActive={false}/></LineChart> : <ScatterChart margin={{top:15,right:15,bottom:15,left:0}}><CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3"/><XAxis type="number" dataKey="actual" hide={!visibleSeries.actual} name="實際值" domain={[low,high]} tickFormatter={axisFormat} tickCount={5} minTickGap={24} interval="preserveStartEnd" tickMargin={8} tick={{fill:"var(--muted)",fontSize:11}}/><YAxis type="number" dataKey="prediction" hide={!visibleSeries.prediction} name="預測值" domain={[low,high]} width={76} tickFormatter={axisFormat} tickCount={5} tick={{fill:"var(--muted)",fontSize:11}}/><Tooltip contentStyle={{background:"var(--surface)",borderColor:"var(--border)"}}/><ReferenceLine segment={[{x:low,y:low},{x:high,y:high}]} stroke="var(--muted)" strokeDasharray="5 4"/><Scatter hide={!visibleSeries.prediction} data={report.preview} name={modelLabel} fill="var(--baseline)" isAnimationActive={false}/>{hasSelection&&<Scatter hide={!visibleSeries.baseline_prediction} data={report.preview.map(r=>({...r,prediction:r.baseline_prediction}))} name="全部候選欄位 + LR" fill="var(--muted)" isAnimationActive={false}/>}<Scatter hide={!visibleSeries.training_mean} data={report.preview.map(r=>({...r,prediction:r.training_mean}))} name="Training 平均值" fill="var(--selected)" isAnimationActive={false}/></ScatterChart>}</ResponsiveContainer></div>
    <div className="cd-chart-legend" role="group" aria-label="圖表資料顯示切換">{chartSeries.map(series=><button key={series.key} id={`custom-chart-toggle-${series.key}`} type="button" aria-pressed={visibleSeries[series.key]} aria-controls="custom-holdout-series" aria-label={`${visibleSeries[series.key]?"隱藏":"顯示"}${series.label}`} onClick={()=>setVisibleSeries(current=>({...current,[series.key]:!current[series.key]}))}><span className="cd-chart-swatch" style={{backgroundColor:series.color}} aria-hidden="true"/><span className="cd-chart-series-name">{series.label}</span>{!visibleSeries[series.key]&&<small>已隱藏</small>}</button>)}</div>
    <p className="cd-chart-toggle-guide">點上方名稱可隱藏，再點一次恢復；只改變圖表顯示，評估指標與下載資料不變。</p>
    {allSeriesHidden&&<p className="cd-chart-empty" role="status">目前已隱藏全部資料，點任一名稱即可恢復顯示。</p>}
    <figcaption className="cd-axis-caption cd-x-caption"><strong>X 軸：{chronological ? "Holdout 筆次（依時間排序）" : "實際值"}</strong>{!chronological&&<span>目標：{report.target} · 原始單位</span>}</figcaption>
    <div id="custom-holdout-chart-guide" className="cd-chart-guide"><strong>這張圖怎麼看？</strong>{chronological ? <p>每個位置對應一筆按時間排列的測試資料，並非等距的未來預報時間。模型線越貼近實際值線，該筆誤差越小；模型在線上方是高估，下方是低估。</p> : <><p>每個模型的點對應一筆測試資料。斜虛線表示「預測＝實際」；點越靠近虛線，該筆誤差越小。在虛線上方是高估，下方是低估。</p><p>兩軸都是目標「{report.target}」的數值，不是輸入特徵；輸入特徵共同用來計算預測值。</p></>}<p>「Training 平均值」對所有資料都猜同一個訓練目標平均值，所以呈水平排列；它是用來比較的基準。</p></div>
    </figure>
    <details className="cd-training-preview"><summary>查看預覽的實際值與預測</summary><div className="fs-table-scroll"><table className="metrics-table"><caption>CSV 資料列從 1 起算，不含標頭與跳過的空白行；每頁 24 筆</caption><thead><tr><th>Holdout 筆次</th><th>{chronological?"來源時間":"CSV 資料列"}</th><th>實際值</th><th>{modelLabel}</th>{hasSelection&&<th>全部欄位 LR</th>}<th>平均值基準</th></tr></thead><tbody>{rows.map(r=><tr key={r.position}><th>{r.position}</th><td>{r.timestamp??r.source_row}</td><td>{format(r.actual,6)}</td><td>{format(r.prediction,6)}</td>{hasSelection&&<td>{format(r.baseline_prediction,6)}</td>}<td>{format(r.training_mean,6)}</td></tr>)}</tbody></table></div><div className="px-pagination"><button disabled={page===0} onClick={()=>setPage(page-1)}>上一頁</button><span>第 {page*24+1}～{Math.min((page+1)*24,report.preview.length)} 筆</span><button disabled={(page+1)*24>=report.preview.length} onClick={()=>setPage(page+1)}>下一頁</button></div></details>
    {report.warnings.length>0&&<div className="cd-validation cd-cautions"><strong>資料與數值提醒</strong><ul>{report.warnings.map((w,i)=><li key={i}>{w}</li>)}</ul></div>}
    <details className="cd-provenance"><summary>訓練欄位、版本與解讀限制</summary><p>目標：{report.target} · 輸入：{report.features.join("、")}</p><p>CSV SHA-256：<code>{report.validation.dataset.sha256}</code></p><p>設定 SHA-256：<code>{report.validation.config_sha256}</code></p><ul>{report.limits.map((l,i)=><li key={i}>{l}</li>)}</ul></details>
    <CustomResultDownload report={report} config={config}/>
  </>;
}
