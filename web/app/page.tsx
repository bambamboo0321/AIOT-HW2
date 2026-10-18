import Link from "next/link";
import { ResearchConclusion } from "@/components/research-conclusion";
import { Icon, type IconName } from "@/components/icon";
import { PerformanceChart } from "@/components/performance-chart";
import { loadOverview } from "@/lib/load-overview";

const number = (value: number, digits = 0) => new Intl.NumberFormat("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
const percent = (value: number, digits = 4) => `${number(value * 100, digits)}%`;

const workflow: { name: string; subtitle: string; icon: IconName; developing?: boolean }[] = [
  { name: "Business Understanding", subtitle: "定義研究問題", icon: "book" },
  { name: "Data Understanding", subtitle: "理解資料與分布", icon: "dataset" },
  { name: "Data Preparation", subtitle: "在 fold 內前處理", icon: "features" },
  { name: "Modeling", subtitle: "線性迴歸與選特徵", icon: "energy" },
  { name: "Evaluation", subtitle: "區分驗證與泛化", icon: "evaluation" },
  { name: "Deployment", subtitle: "本機介面開發中", icon: "prediction", developing: true },
];

export default async function OverviewPage() {
  const snapshot = await loadOverview();
  if (snapshot.status !== "ready") return <div className="page-container"><div className="eyebrow">RESEARCH WORKSPACE</div><h1>Research overview</h1>
    <section className="data-error" role="alert"><Icon name="info" size={28}/><h2>研究資料尚未就緒</h2><p>無法顯示已核實的結果，因此沒有填入替代數值。</p><ul>{snapshot.issues.map(issue => <li key={issue}>{issue}</li>)}</ul></section>
  </div>;
  const { data } = snapshot;
  const { dataset, models, exploration } = data;
  const [baseline, selected] = models;
  const nominal = percent(data.nominalCoverage, 0);
  const improved = selected.rmse < baseline.rmse;
  return <div className="page-container">
    <section className="research-header" aria-labelledby="research-title">
      <div><div className="eyebrow"><span className="eyebrow-line"/>APPLIANCES ENERGY PREDICTION LAB</div>
        <h1 id="research-title">Research overview<span className="heading-dot">.</span></h1>
        <p className="page-description">Machine Learning <span>·</span> Feature Selection <span>·</span> Model Evaluation</p>
      </div><div className="source-badge"><span className="status-dot"/>Artifact-backed research</div>
    </section>
    <div className="research-context"><span className="context-icon"><Icon name="book" size={17}/></span>
      <p><strong>Multiple Linear Regression &amp; Feature Selection</strong><span className="context-separator">/</span>以 CRISP-DM 六階段，研究家庭環境資訊與家電耗能之間的關係。</p>
      <a href="https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction" target="_blank" rel="noreferrer noopener" className="context-source">UCI Dataset <Icon name="arrow" size={15}/></a>
    </div>

    <section className="dataset-grid" aria-label="Dataset overview">
      <StatCard label="Records" value={number(dataset.records)} icon="dataset" note={`${number(dataset.samplingSeconds / 60)}-minute observations`} tag="DATASET"/>
      <StatCard label="Raw input features" value={number(dataset.rawFeatures)} icon="features" note="Original dataset inputs"/>
      <StatCard label="Modeling features" value={number(dataset.modelingFeatures)} icon="evaluation" note="Baseline numeric features"/>
      <StatCard label="Prediction target" value={dataset.target} icon="energy" note="Appliance energy consumption" tag={dataset.targetUnit} compact/>
    </section>

    <ResearchConclusion baseline={baseline} selected={selected}/>

    <div className="overview-grid">
      <section className="panel performance-panel" aria-labelledby="performance-title">
        <div className="panel-header"><div><div className="section-kicker">MODEL EVALUATION</div><h2 id="performance-title">Holdout performance</h2><p>Chronological split <span className="middot">·</span> {number(data.holdoutRows)} observations</p></div><span className="neutral-tag">Fixed models</span></div>
        <PerformanceChart points={models.map(model => ({ label: model.id === "baseline" ? "Baseline" : "MI + LR", rmse: model.rmse, mae: model.mae, r2: model.r2 }))} unit={data.errorUnit}/>
        <div className="metrics-table-wrap"><table className="metrics-table"><caption className="sr-only">兩個固定模型在相同 chronological Holdout 的評估結果；R² 無單位。</caption>
          <thead><tr><th scope="col">Model</th><th scope="col">Features</th><th scope="col">RMSE <small>({data.errorUnit})</small></th><th scope="col">MAE <small>({data.errorUnit})</small></th><th scope="col">R²</th></tr></thead>
          <tbody>{models.map(model => <tr key={model.id}><th scope="row"><span className={`legend-dot ${model.id}`}/><span>{model.id === "baseline" ? "Baseline LR" : "MI + LR"}<small>{model.label}</small></span></th>
            <td>{model.features}</td><td>{number(model.rmse, 6)}</td><td>{number(model.mae, 6)}</td><td>{number(model.r2, 6)}</td></tr>)}</tbody>
        </table></div>
        <div className="panel-footnote"><Icon name="info" size={15}/><span>此 Holdout 已於 Phase 2 查看；此處為固定模型的描述性比較。</span></div>
      </section>

      <section className="panel interval-panel" aria-labelledby="interval-title"><div className="panel-header"><div><div className="section-kicker">UNCERTAINTY ESTIMATION</div><h2 id="interval-title">Prediction intervals</h2><p>Single-observation prediction interval</p></div><span className="accent-tag">{nominal} PI</span></div>
        <div className="interval-models">{models.map(model => <div className="interval-model" key={model.id}>
          <div className="interval-model-title"><span><i className={`legend-dot ${model.id}`}/>{model.id === "baseline" ? "Baseline Linear Regression" : "MI + Linear Regression"}</span><span className="tiny-label">HOLDOUT</span></div>
          <div className="coverage-row"><span>Empirical coverage</span><strong>{percent(model.coverage)}</strong></div>
          <progress className={`coverage-progress ${model.id}`} value={model.coverage} max={1} aria-label={`${model.label} empirical coverage`}>{percent(model.coverage)}</progress>
          <div className="width-row"><span>Average interval width</span><strong>{number(model.averageWidth, 3)} <small>{data.widthUnit}</small></strong></div>
        </div>)}</div>
        <div className="interval-note"><Icon name="info" size={17}/><p><strong>Coverage ≠ prediction accuracy</strong>較高覆蓋率可能來自較寬區間。名目 {nominal} 不是泛化保證，也不是平均值的 Confidence Interval。</p></div>
      </section>
    </div>

    <section className="panel workflow-panel" aria-labelledby="workflow-title"><div className="panel-header"><div><div className="section-kicker">RESEARCH METHODOLOGY</div><h2 id="workflow-title">From question to evidence</h2></div><span className="neutral-tag">CRISP-DM</span></div>
      <ol className="workflow-grid">{workflow.map((stage, i) => <li key={stage.name} className={stage.developing ? "developing" : ""}>
        <div className="workflow-top"><span className="workflow-icon"><Icon name={stage.icon} size={18}/></span><span className="step-number">0{i + 1}</span></div>
        <h3>{stage.name}</h3><p>{stage.subtitle}</p><span className={`workflow-state ${stage.developing ? "pending" : ""}`}>{stage.developing ? <><span className="pending-dot"/>In development</> : <><Icon name="check" size={12}/>Research artifacts</>}</span>
      </li>)}</ol><div className="workflow-disclaimer">Deployment 目前僅有本機 Python 流程與網站開發介面；尚未建立正式服務。</div>
    </section>

    <section className="findings-section" aria-labelledby="findings-title"><div className="section-heading"><div><div className="section-kicker">WHAT THE DATA TELLS US</div><h2 id="findings-title">Research findings</h2></div><span className="muted-label">Results, with their limits.</span></div>
      <div className="findings-grid"><article className="finding-card"><span className="finding-index">01 <span>FEATURE REDUCTION</span></span>
        <h3>Fewer features.<br/>The same need for evidence.</h3>
        <div className="feature-reduction"><strong>{baseline.features}</strong><Icon name="arrow" size={25}/><strong className="selected-text">{selected.features}</strong><span>features</span></div>
        <p>MI 將輸入由 {baseline.features} 欄精簡為 {selected.features} 欄，但 Holdout RMSE {improved ? "降低" : "並未改善"}。特徵較少不保證預測較準。</p>
        <div className="finding-note"><Icon name="info" size={15}/>原始資料特徵數與選取後 k 分開解讀。</div>
      </article>
      <article className="finding-card"><span className="finding-index">02 <span>EXPLORATORY SEARCH · PHASE 3D</span></span>
        <h3>A better CV score.<br/>An open research question.</h3>
        {exploration.available ? <><div className="exploration-score"><strong>{number(exploration.bestRmse, 6)}</strong><span>{data.errorUnit}<small>Exploratory fixed-subset CV RMSE</small></span></div>
          <p>{number(exploration.sampleCount)} 組抽樣中找到較低 CV 誤差的固定組合；固定 MI 參考為 {number(exploration.fixedMiRmse, 6)} {data.errorUnit}。{exploration.folds}-fold Training CV，gap={exploration.gap}。</p>
          <details className="exploration-details"><summary>查看探索性組合與評估限制</summary><div className="feature-chips">{exploration.bestFeatures.map(name => <span key={name}>{name}</span>)}</div><p>這是抽樣中的較低 CV 分數，不是最終模型。固定 MI 組合有回顧性資料重用偏差，搜尋最低值也有選取偏差；兩者不代表獨立泛化驗證。</p></details>
        </> : <p className="missing-inline" role="status">{exploration.reason}</p>}
        <div className="finding-note warning"><Icon name="info" size={15}/>Random Search ≠ independent generalization validation.</div>
      </article></div>
    </section>

    <details className="provenance"><summary><Icon name="check" size={15}/>{snapshot.sources.length} source artifacts<span>資料來源與追溯</span></summary><div className="provenance-body"><p>Dataset：{dataset.name}。保留成果快照，未重新訓練；網站不公開原始 CSV 或模型檔。</p>{snapshot.sources.map(source => <div className="provenance-item" key={source.path}><code>{source.path}</code><span>SHA-256 {source.sha256.slice(0, 16)}…</span></div>)}<p>四個研究頁面與自訂 CSV 檢查已可瀏覽；自訂資料模型訓練尚未串接。<Link href="/feature-selection">查看 Feature Selection 成果 →</Link></p></div></details>
  </div>;
}

function StatCard({ label, value, note, icon, tag, compact = false }: { label: string; value: string; note: string; icon: IconName; tag?: string; compact?: boolean }) {
  return <article className="stat-card"><div className="stat-top"><span>{label}</span><span className="stat-icon"><Icon name={icon} size={18}/></span></div><div className="stat-main"><strong className={compact ? "compact" : ""}>{value}</strong>{tag && <span className="stat-tag">{tag}</span>}</div><p>{note}</p></article>;
}
