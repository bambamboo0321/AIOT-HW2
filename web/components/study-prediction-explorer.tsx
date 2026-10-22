import { ResearchLivePredictor } from "@/components/research-live-predictor";
import Link from "next/link";
import { Icon } from "./icon";
import { StudyPredictionBrowser } from "./study-prediction-browser";
import { loadStudyEvaluation } from "@/lib/load-study-evaluation";
import { DATASET_REGISTRY } from "@/lib/dataset-registry";
import { researchHref } from "@/lib/dataset-routing";
import type { StudyDatasetId } from "@/lib/study-overview-data";

export async function StudyPredictionExplorer({ id }: { id: StudyDatasetId }) {
  const registry = DATASET_REGISTRY[id], snapshot = await loadStudyEvaluation(id);
  if (snapshot.status !== "ready") return <div className="page-container px-page" data-dataset-id={id} data-prediction-state="unavailable">
    <div className="eyebrow">{registry.name}</div><h1>Prediction explorer · {registry.name}</h1>
    <ResearchLivePredictor datasetId={id}/><h2 className="rlp-holdout-heading">Holdout Explorer · 已保存的測試資料</h2><section className="data-error" role="alert"><h2>逐筆預測 unavailable</h2><p>快照缺少或未通過驗證，不使用其他資料集替代。</p><ul>{snapshot.issues.map(i => <li key={i}>{i}</li>)}</ul></section></div>;
  const { data } = snapshot, { dataset, protocol } = data.study;
  return <div className="page-container px-page study-prediction-page" data-dataset-id={id} data-prediction-state="ready">
    <section className="research-header"><div><div className="eyebrow"><span className="eyebrow-line"/>{dataset.name}</div><h1>Prediction explorer<span className="heading-dot">.</span></h1><p className="page-description">{dataset.name} · Target：{dataset.target} · {dataset.unit}</p></div><span className="source-badge"><span className="status-dot"/>Saved Holdout · v1</span></section>
    <ResearchLivePredictor datasetId={id}/><h2 className="rlp-holdout-heading">Holdout Explorer · 已保存的測試資料</h2><div className="research-context"><span className="context-icon"><Icon name="prediction" size={18}/></span><p><strong>{protocol.holdoutRows} 筆／模型</strong> · 與 Model Evaluation 共用同一份已驗證快照；只瀏覽，不重新訓練、推論或選模。</p></div>
    <section className="px-global" aria-label="樣本與區間定義"><div><span>Training／Holdout</span><strong>{protocol.trainRows} / {protocol.holdoutRows}</strong><p>隨機 80/20 · seed {protocol.seed}</p></div><div><span>樣本編號</span><strong>1–{protocol.holdoutRows}</strong><p>Holdout 筆次，非原始 CSV 列號</p></div><div><span>Residual 定義</span><strong>實際 − 預測</strong><p>正值為低估；負值為高估</p></div><div><span>名目單筆 PI</span><strong>95<small>%</small></strong><p>非平均值 CI，不保證實際覆蓋率</p></div></section>
    <section className="panel"><div className="panel-header"><div><div className="section-kicker">READ SAVED OBSERVATIONS</div><h2>使用前提與樣本意義</h2></div></div><div className="px-section-body">
      <p className="fs-warning">{id === "student-performance" ? "G3 為 0–20 分；必須已知 G1、G2 才適用，非學期開始前預測。線性迴歸不保證輸出落在 0–20，預測值與 95% PI 均不裁切。absences 的累計截止時點未明。" : "價格使用原始資料單位，幣別未核實。14 個候選欄位含 12 數值與 2 固定二元類別；輸入缺失僅在 Training fold 補值及縮放。預測與 PI 下界可能為負，保留原值，不代表有效的負價格。"}</p>
      <p className="fs-note">樣本索引沒有時間意義，不代表受試者或汽車識別碼。所有選欄、補值、縮放及 CV 僅使用 Training；固定 k=7；MI 方法由 Training CV 鎖定，不以此處 Holdout 調參。</p></div></section>
    <StudyPredictionBrowser key={id} data={data}/>
    <section className="panel"><div className="panel-header"><h2>局部誤差與整體評估</h2></div><div className="px-section-body"><p className="fs-note">搜尋或排序只調整顯示；完整模型指標與 PI 覆蓋率仍使用全部 Holdout。名目 95% PI 不保證實際 95% 覆蓋，也未校正選欄不確定性或異質變異。</p><Link className="ev-text-link" href={researchHref("/model-evaluation", id)}>查看相同資料集的完整 Model Evaluation →</Link></div></section>
    <details className="provenance"><summary><Icon name="book" size={16}/>共用逐筆來源與完整性<span>Phase 3D · SHA-256</span></summary><div className="provenance-body"><p>直接使用 Model Evaluation 的版本化快照與相同驗證 adapter，沒有建立第二套預測資料。來源路徑只供追溯，Production 不讀取 experiments/。缺失預測或 PI 顯示 unavailable，不由摘要反推。</p>{data.sources.map(s => <div className="provenance-item" key={s.path}><code>{s.path}</code><code>{s.sha256}</code></div>)}<div className="provenance-item"><code>Overview SHA-256</code><code>{data.overviewSha256}</code></div></div></details>
  </div>;
}
