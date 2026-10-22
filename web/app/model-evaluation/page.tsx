import { datasetFromPageParams, researchHref, type ResearchPageProps } from "@/lib/dataset-routing";
import { StudyModelEvaluation } from "@/components/study-model-evaluation";
import Link from "next/link";
import { ResearchConclusion } from "@/components/research-conclusion";
import { Icon } from "@/components/icon";
import { EvaluationExplorer } from "@/components/evaluation-explorer";
import { loadEvaluation } from "@/lib/load-evaluation";
const format = (n: number, digits = 4) => n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const date = (s: string) => s.replace("T", " ").slice(0, 16);
export async function generateMetadata({ searchParams }: ResearchPageProps) {
  const dataset = datasetFromPageParams(await searchParams);
  return { title: `Model Evaluation · ${dataset.name}` };
}
export default async function ModelEvaluationPage({ searchParams }: ResearchPageProps) {
  const dataset = datasetFromPageParams(await searchParams);
  if (dataset.id !== "appliances-energy") return <StudyModelEvaluation id={dataset.id}/>;
  const snapshot = await loadEvaluation();
  if (snapshot.status !== "ready") return <div className="page-container"><div className="eyebrow">MODEL EVALUATION</div><h1>Model evaluation</h1><section className="data-error" role="alert"><h2>模型評估資料尚未就緒</h2><p>請確認正式成果與來源定義，沒有填入替代分數。</p><ul>{snapshot.issues.map(issue => <li key={issue}>{issue}</li>)}</ul></section></div>;
  const { data } = snapshot; const [baseline, selected] = data.models;
  const delta = selected.holdout.rmse - baseline.holdout.rmse;
  const reduction = (1 - selected.features.length / baseline.features.length) * 100;
  return <div className="page-container ev-page"><section className="research-header"><div><div className="eyebrow"><span className="eyebrow-line"/>MEASURE PERFORMANCE, UNDERSTAND LIMITS</div><h1>Model evaluation<span className="heading-dot">.</span></h1><p className="page-description">固定模型比較、誤差分布與預測區間。</p></div><div className="source-badge"><span className="status-dot"/>Saved predictions · verified</div></section>
    <div className="research-context"><span className="context-icon"><Icon name="evaluation" size={18}/></span><p><strong>Baseline LR / MI + LR</strong><span className="context-separator">/</span>讀取 Phase 3C 已保存成果，理解預測能力與不確定性。</p></div>
    <section className="ev-summary" aria-label="固定模型評估摘要"><div><span>完整 Holdout</span><strong>{data.holdoutRows.toLocaleString("en-US")}<small> 筆</small></strong><p>同一時間切分，保留原順序</p></div><div><span>模型複雜度</span><strong>{baseline.features.length} → {selected.features.length}</strong><p>MI 減少 {format(reduction, 0)}% 特徵</p></div><div><span>Holdout RMSE · MI − Baseline</span><strong>{delta > 0 ? "+" : ""}{format(delta)}<small> Wh</small></strong><p>{delta > 0 ? "MI 在這份 Holdout 的誤差較高" : delta < 0 ? "MI 在這份 Holdout 的誤差較低" : "兩模型 RMSE 相同"}</p></div><div><span>名目 Prediction Interval</span><strong>{format(data.nominalCoverage * 100, 0)}<small>%</small></strong><p>單筆觀測區間，非實測保證</p></div></section>
    <section className="ev-split" aria-label="評估時間切分"><div><span className="section-kicker">TRAINING · {data.trainingRows.toLocaleString("en-US")} ROWS</span><p>{date(data.trainingPeriod.start)} → {date(data.trainingPeriod.end)}</p></div><Icon name="arrow" size={18}/><div><span className="section-kicker">HOLDOUT · {data.holdoutRows.toLocaleString("en-US")} ROWS</span><p>{date(data.holdoutPeriod.start)} → {date(data.holdoutPeriod.end)}</p></div><small>依來源時間顯示，原始時區未註明。</small></section>
    <ResearchConclusion datasetId={dataset.id} detailed baseline={{ label: baseline.label, features: baseline.features.length, ...baseline.holdout, ...baseline.interval }} selected={{ label: selected.label, features: selected.features.length, ...selected.holdout, ...selected.interval }}/>
    <EvaluationExplorer data={data}/>
    <div className="ev-explain-grid"><section className="panel"><div className="panel-header"><div><div className="section-kicker">FEATURES VERSUS ERROR</div><h2>少特徵不保證更準</h2><p>完整 Training 最終模型使用的欄位。</p></div></div><div className="ev-section-body"><p className="ev-body-text">MI 保留 {selected.features.length} 個特徵。相較 {baseline.features.length} 欄 Baseline，Holdout RMSE 差 {delta > 0 ? "+" : ""}{format(delta)} Wh；這是此時段的描述性結果，不據此重新選 k 或替換模型。</p><div className="feature-chips">{selected.features.map(f => <span key={f}>{f}</span>)}</div><details className="fs-data-table"><summary>查看 Baseline 的全部特徵</summary><div className="feature-chips">{baseline.features.map(f => <span key={f}>{f}</span>)}</div></details></div></section><section className="panel"><div className="panel-header"><div><div className="section-kicker">KEEP EVALUATION LEVELS CLEAR</div><h2>CV 與固定 Holdout</h2><p>不同資料範圍與不同評估程序。</p></div></div><div className="ev-section-body"><p className="ev-body-text">Feature Selection 的 CV 評估包含各 fold 重新選欄、選 k；本頁評估完整 Training 建立後的兩個固定模型。既有研究沒有六方法的 Holdout 成績，不補造排名。</p><Link className="ev-text-link" href={researchHref("/feature-selection", dataset.id)}>查看 Feature Selection 與逐 fold 評分 <Icon name="arrow" size={14}/></Link></div></section></div>
    <section className="panel ev-limitations"><div className="panel-header"><div><div className="section-kicker">INTERPRETATION AND ASSUMPTIONS</div><h2>這些結果能說明什麼？</h2></div></div><div className="ev-section-body"><ul><li>Phase 2 曾查看 Holdout；它不是從未見過的獨立最終測試集，不能據此宣稱跨住宅或未來時段泛化。</li><li>模型使用當下感測器資訊估計同時間能耗，並非只使用過去資料的多步預報。</li><li>經典 OLS 區間依賴線性結構、誤差獨立與同方差等假設；時間相依、尖峰與漂移都可能影響解讀。</li><li>MI 特徵由同一 Training 選出；這份 plug-in PI 沒有涵蓋選欄不確定性，也不是同時覆蓋整條曲線的機率保證。</li></ul><details className="fs-data-table"><summary>查看保存的 PI 公式與彙整定義</summary><p className="ev-body-text">區間中心使用保存的預測；殘差變異、有效秩、自由度與 leverage 只使用既有 Training 成果。本頁沒有重新擬合或依 Holdout 更動區間。</p><code className="ev-formula">{data.formula}</code><p className="fs-note">殘差平均、線性插值分位數與分箱筆數由保存預測作描述性彙整。指標保留正式 JSON 的完整精度；彙整結果另核對正式指標與區間計數。寬度 SD 使用 ddof=0。</p></details></div></section>
    <details className="provenance"><summary><Icon name="book" size={16}/>資料來源與完整性<span>SHA-256 verified</span></summary><div className="provenance-body"><p>只讀既有 Phase 3C 成果；來源雜湊、兩模型的切分、完整預測筆數、指標與區間計數均經核對。送往瀏覽器的殘差資料只有 20 個分箱與描述統計，未公開完整逐筆 CSV、原始 Dataset 或模型。</p>{snapshot.sources.map(source => <div className="provenance-item" key={source.path}><code>{source.path}</code><code>{source.sha256}</code></div>)}</div></details>
  </div>;
}
