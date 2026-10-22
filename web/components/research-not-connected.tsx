import Link from "next/link";
import type { DatasetDefinition } from "@/lib/dataset-registry";
import { DEFAULT_DATASET_ID } from "@/lib/dataset-registry";
import { researchHref, type ResearchPath } from "@/lib/dataset-routing";
import { Icon } from "./icon";

export function ResearchNotConnected({ dataset, pageTitle, path }: { dataset: DatasetDefinition; pageTitle: string; path: ResearchPath }) {
  return <div className="page-container" data-dataset-id={dataset.id} data-research-state="not-connected">
    <section className="research-header"><div><div className="eyebrow"><span className="eyebrow-line"/>{dataset.name}</div>
      <h1>{pageTitle}<span className="heading-dot">.</span></h1><p className="page-description">{dataset.name} · Target：{dataset.target}</p></div>
      <span className="neutral-tag">尚未接入</span></section>
    <section className="panel dataset-pending" aria-labelledby="dataset-pending-title" role="status">
      <Icon name="dataset" size={30}/><h2 id="dataset-pending-title">研究結果尚未接入</h2>
      <p>此資料集的研究頁面將於後續階段串接。目前僅提供已確認的資料集基本資訊。</p>
      <dl><div><dt>有效資料筆數</dt><dd>{dataset.rows.toLocaleString("en-US")}</dd></div>
        <div><dt>候選特徵</dt><dd>{dataset.candidateFeatures}</dd></div>
        <div><dt>Target／單位</dt><dd>{dataset.target} · {dataset.targetUnit}</dd></div>
        <div><dt>評估協定</dt><dd>{dataset.protocol.label}</dd></div></dl>
      <p>{dataset.note}</p>
      <Link className="ev-text-link" href={researchHref(path, DEFAULT_DATASET_ID)}>查看既有能源研究 →</Link>
    </section>
  </div>;
}
