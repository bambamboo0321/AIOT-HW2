import Link from "next/link";
import { Icon } from "./icon";
import { StudyFeatureExplorer } from "./study-feature-explorer";
import { DATASET_REGISTRY } from "@/lib/dataset-registry";
import { researchHref } from "@/lib/dataset-routing";
import { loadStudyFeatureSelection } from "@/lib/load-study-feature-selection";
import type { StudyDatasetId } from "@/lib/study-overview-data";
export async function StudyFeatureSelection({id}:{id:StudyDatasetId}) {
  const registry=DATASET_REGISTRY[id],snapshot=await loadStudyFeatureSelection(id);
  if(snapshot.status!=="ready")return <div className="page-container fs-page" data-dataset-id={id} data-feature-state="unavailable"><div className="eyebrow">{registry.name}</div><h1>{registry.name} · Feature selection</h1><section className="data-error" role="alert"><h2>特徵選擇資料 unavailable</h2><p>快照缺少或未通過驗證；不使用能源或其他資料集替代。</p><ul>{snapshot.issues.map(i=><li key={i}>{i}</li>)}</ul></section></div>;
  const {data}=snapshot,{dataset,protocol}=data.study;
  return <div className="page-container fs-page" data-dataset-id={id} data-feature-state="ready">
    <section className="research-header"><div><div className="eyebrow"><span className="eyebrow-line"/>{dataset.name}</div><h1>Feature selection<span className="heading-dot">.</span></h1><p className="page-description">{dataset.name} · {dataset.rows} 筆 · {dataset.candidates.length} 個候選特徵 · Target：{dataset.target}（{dataset.unit}）</p></div><span className="source-badge"><span className="status-dot"/>Verified research · v1</span></section>
    <div className="research-context"><span className="context-icon"><Icon name="features" size={18}/></span><p><strong>固定 k=7 與自動選 k 的六方法研究</strong>／所有最終預測器均為獨立 Linear Regression。{registry.note}</p></div>
    <section className="fs-protocol" aria-label="評估資料範圍"><div><strong>{protocol.trainRows}</strong><span>Training · 隨機 80/20，seed {protocol.seed}</span></div><div><strong>{protocol.outerFolds} / {protocol.innerFolds}</strong><span>Outer / Inner KFold · 非時間 CV</span></div><div><strong>{protocol.fixedK} / auto</strong><span>預先指定 k／內層自動選擇</span></div><div><strong>{protocol.holdoutRows}</strong><span>Holdout · 未參與選欄或調參</span></div></section>
    <StudyFeatureExplorer key={id} data={data}/>
    <section className="panel"><div className="panel-header"><div><div className="section-kicker">PROTOCOL AND LIMITS</div><h2>Training CV 與 Holdout 分開解讀</h2></div></div><div className="fs-section-body"><p className="fs-body-text">Training CV 在 80% Training 內執行，外層 5-fold 每次重新學習補值、StandardScaler、特徵選擇與 Linear Regression。SFS、LassoCV、RFECV 使用外層 Training 內的 3-fold；F-test、MI、RFE 的 k=7 預先固定，不另做內層 k 搜尋。MI 由平均 Training CV RMSE 規則在查看 Holdout 前鎖定，勝出者 CV 仍有選取樂觀偏差。</p>
      <p className="fs-note">最終完整 Training 重新選欄及 fit 後，Holdout 用來評估既定模型。此頁不以 Holdout 選方法、k 或 alpha；查看這些研究結果也不會重新訓練。</p>
      {id === "student-performance" ? <p className="fs-warning">模型使用 G1、G2：已知前兩次成績後估計 G3，非學期開始前預測。absences 的累計截止時點未明；類別欄位未納入這 15 個數值候選特徵。</p> : <p className="fs-warning">原始 205 筆中排除 4 筆 price 缺失。候選輸入含 12 個數值欄與固定二元編碼 fuel-type（gas=0、diesel=1）、aspiration（std=0、turbo=1）；輸入缺失僅在各 Training fold 補中位數。幣別未核實；201 筆的單次切分不代表市場未來表現。</p>}
      <Link className="ev-text-link" href={researchHref("/",id)}>查看此資料集固定 Holdout 結果與 PI 限制 →</Link></div></section>
    <details className="provenance"><summary><Icon name="book" size={16}/>資料定義與來源<span>Snapshot v1 · SHA-256</span></summary><div className="provenance-body"><p>以已驗證實驗匯出六方法的完整 Training 最終選欄、五折實際選欄與 Training CV 平均／標準差。不複製原始資料列、模型或 Holdout 預測。來源路徑僅供追溯，Production 不讀取 experiments/。</p><div className="provenance-item"><code>{dataset.rawFile}</code><code>SHA-256 {dataset.rawSha256}</code></div>{data.sources.map(s=><div key={s.path} className="provenance-item"><code>{s.path}</code><code>{s.sha256}</code></div>)}<div className="provenance-item"><code>連結 Overview SHA-256</code><code>{data.overviewSha256}</code></div><p>缺失成果明確顯示 unavailable；不推測其他 k，也不使用能源結果填補。</p></div></details>
  </div>;
}
