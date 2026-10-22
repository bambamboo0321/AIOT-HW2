import { DEFAULT_DATASET_ID, type DatasetId } from "@/lib/dataset-registry";
import { researchHref } from "@/lib/dataset-routing";
import Link from "next/link";
import { explainR2 } from "@/lib/result-interpretation";

type ConclusionModel = { label: string; features: number; rmse: number; r2: number; coverage: number; averageWidth: number };
const format = (value: number, digits = 3) => value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export function ResearchConclusion({ baseline, selected, detailed = false, datasetId = DEFAULT_DATASET_ID }: { datasetId?: DatasetId; baseline: ConclusionModel; selected: ConclusionModel; detailed?: boolean }) {
  const delta = selected.rmse - baseline.rmse;
  return <section id={detailed ? "research-conclusion" : undefined} className="panel research-conclusion" aria-label="研究結論與使用限制">
    <div className="panel-header"><div><div className="section-kicker">READ THE RESULTS WITH THEIR LIMITS</div><h2>研究結論與使用限制</h2><p>同時間耗能估計 · 目前成果尚不足以宣稱能可靠掌握短時間尖峰。</p></div><span className="neutral-tag">完整 Holdout</span></div>
    <div className="conclusion-body"><div className="conclusion-grid">
      <article><h3>先看誤差，再看曲線</h3><p>Baseline 的 R² 為 <strong>{format(baseline.r2)}</strong>，MI 為 <strong>{format(selected.r2)}</strong>。R² 不是準確率；完整指標與局部時段需一起閱讀。</p></article>
      <article><h3>精簡特徵的取捨</h3><p>MI 將 {baseline.features} 個特徵減為 {selected.features} 個；Holdout RMSE {delta > 0 ? "增加" : delta < 0 ? "降低" : "差為"} <strong>{format(Math.abs(delta))} Wh</strong>。這份資料的比較不代表所有特徵選擇方法的普遍結果。</p></article>
      <article><h3>95% 區間不等於 95% 準確</h3><p>Baseline／MI 實測覆蓋率為 {format(baseline.coverage * 100, 2)}%／{format(selected.coverage * 100, 2)}%，平均區間寬度為 {format(baseline.averageWidth, 1)}／{format(selected.averageWidth, 1)} Wh。覆蓋率需與寬度及局部漏失一起看。</p></article>
    </div>
    {detailed ? <div className="conclusion-details"><h3>如何判斷模型能力？</h3><p>R² = 1 代表這批資料的預測與實際值完全吻合；R² = 0 代表平方誤差等於以這批實際平均值作固定預測；負值代表誤差更高。這個平均值是事後比較基準，實際部署時無法預先知道測試期平均值。</p><p>Baseline：{explainR2(baseline.r2)} MI：{explainR2(selected.r2)}</p><h3>目前證據與尚待驗證的原因</h3><p>時間圖可檢查尖峰低估與低耗能時高估；每個時段的程度不同。環境特徵未充分反映電器啟停、線性結構不足或不同時段的資料分布改變，都是待驗證的解釋，現有比較尚未分辨各因素的影響。</p><p>研究成果適合用於比較固定模型與理解限制。若要評估實際應用，需先定義可接受誤差，再以適當基準、可在預測時取得的特徵，以及額外保留的未見時段進行驗證。未來預報還需明確定義提前多久預測。</p></div> : <Link className="ev-text-link" href={researchHref("/model-evaluation", datasetId, "research-conclusion")}>閱讀完整模型解讀與研究限制 →</Link>}
    <p className="conclusion-scope">這份 Holdout 曾於 Phase 2 查看；結論為既有固定成果的描述性比較，不代表跨住宅或未來時段的泛化保證。</p>
    </div>
  </section>;
}
