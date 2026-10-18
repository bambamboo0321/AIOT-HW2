import Link from "next/link";
import { Icon } from "@/components/icon";
import { PredictionExplorer } from "@/components/prediction-explorer";
import { loadPredictions } from "@/lib/load-predictions";
import { predictionWindow } from "@/lib/prediction-window";
export const metadata = { title: "Prediction Explorer" };
export default async function PredictionExplorerPage() {
  const snapshot = await loadPredictions();
  if (snapshot.status !== "ready") return <div className="page-container"><div className="eyebrow">PREDICTION EXPLORER</div><h1>Prediction explorer</h1><section className="data-error" role="alert"><h2>逐筆預測資料尚未就緒</h2><p>無法呈現已核實的保存預測。</p><ul>{snapshot.issues.map(issue => <li key={issue}>{issue}</li>)}</ul></section></div>;
  const { meta, rows } = snapshot.data; const initial = predictionWindow(rows, meta, meta.defaultStart, meta.defaultSize);
  return <div className="page-container px-page"><section className="research-header"><div><div className="eyebrow"><span className="eyebrow-line"/>FOLLOW EACH OBSERVATION THROUGH TIME</div><h1>Prediction explorer<span className="heading-dot">.</span></h1><p className="page-description">對照實際值、模型預測與單筆觀測區間。</p></div><div className="source-badge"><span className="status-dot"/>Saved predictions · time ordered</div></section><div className="research-context"><span className="context-icon"><Icon name="prediction" size={18}/></span><p><strong>{meta.totalRows.toLocaleString("en-US")} 筆固定 Holdout 預測</strong><span className="context-separator">/</span>只瀏覽已保存成果，操作不會重新訓練或預測。</p></div>
    <section className="px-global" aria-label="完整 Holdout 固定參考"><div><span>完整 Holdout</span><strong>{meta.totalRows.toLocaleString("en-US")}<small> 筆</small></strong><p>每 {meta.samplingSeconds / 60} 分鐘一筆</p></div>{meta.models.map(m => <div key={m.id}><span>{m.label} · 完整 Holdout RMSE</span><strong>{m.holdout.rmse.toFixed(4)}<small> Wh</small></strong><p>{m.featureCount} 個特徵 · PI 覆蓋率 {(m.coverage * 100).toFixed(4)}%</p></div>)}<div><span>名目 Prediction Interval</span><strong>{(meta.nominalCoverage * 100).toFixed(0)}<small>%</small></strong><p>不是實測保證或分類準確率</p></div></section>
    <PredictionExplorer meta={meta} initial={initial}/>
    <section className="panel px-context"><div className="panel-header"><div><div className="section-kicker">LOCAL PATTERNS, GLOBAL LIMITS</div><h2>時段探索與整體評估</h2></div></div><div className="px-section-body"><p>這份 Holdout 在 Phase 2 已被查看。較好看的局部時段不代表完整資料表現，也不據此重新選特徵、選 k 或校準 PI。完整指標、殘差分布與 OLS 區間假設請見模型評估頁。</p><Link href="/model-evaluation" className="ev-text-link">查看完整 Model Evaluation <Icon name="arrow" size={14}/></Link></div></section>
    <details className="provenance"><summary><Icon name="book" size={16}/>逐筆預測來源與完整性<span>SHA-256 verified</span></summary><div className="provenance-body"><p>逐筆資料來自 Phase 3C 已保存預測。核對完整 Holdout 的列數、時間順序、原始來源雜湊、回歸指標與區間計數；瀏覽器每次只取得最多 288 筆的視窗，沒有任意檔案讀取入口，也未載入原始特徵 Dataset、leverage 或模型。</p>{snapshot.sources.map(source => <div className="provenance-item" key={source.path}><code>{source.path}</code><code>{source.sha256}</code></div>)}</div></details>
  </div>;
}
