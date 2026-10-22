import { Icon, type IconName } from "./icon";
import { PerformanceChart } from "./performance-chart";
import { loadStudyOverview } from "@/lib/load-study-overview";
import { DATASET_REGISTRY } from "@/lib/dataset-registry";
import type { StudyDatasetId, StudyModel } from "@/lib/study-overview-data";

const number = (n: number, digits = 0) => new Intl.NumberFormat("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(n);
const percent = (n: number) => `${number(n * 100, 2)}%`;
const descriptions = {
  "automobile-price": {
    question: "以車輛尺寸、引擎、油耗與兩個二元類別欄位，研究汽車價格與多元線性迴歸的關係。",
    source: "https://archive.ics.uci.edu/dataset/10/automobile",
    preparation: "原始 205 筆中排除 4 筆 price 缺失；保留輸入缺失，僅在各 Training fold 以中位數補值及 StandardScaler 縮放。",
    encoding: "12 個數值欄位與 2 個固定二元類別：fuel-type（gas=0、diesel=1）、aspiration（std=0、turbo=1）。類別直接使用既定語意編碼，沒有從 Holdout 學習字典，也沒有 One-Hot 展開。",
    limitation: "這是指定 14 欄與單次隨機 Holdout 的研究，不代表完整汽車市場。價格沿用來源單位，幣別尚未核實。樣本少，車輛價格可能存在非線性與異質變異。",
    timing: "固定規格的車價估計；沒有時間序列預報或市場未來價格驗證。",
  },
  "student-performance": {
    question: "以 15 個數值輸入建立葡萄牙語課程 G3 成績估計，檢查選欄後的預測能力與限制。",
    source: "https://archive.ics.uci.edu/dataset/320/student+performance",
    preparation: "649 筆均保留，包含 G3=0 的紀錄；目前選定欄位無缺失。各 Training fold 仍使用中位數補值流程與 StandardScaler，未對 Holdout fit。",
    encoding: "使用 15 個數值候選特徵；教育程度、讀書時間等級別保留原資料數值編碼。類別欄位未加入本次 Baseline 或 MI 模型。",
    limitation: "模型使用 G1、G2，屬於已知前兩次成績後的預測，非學期開始前預測。absences 的累計截止時點未明，不能宣稱所有輸入在更早時點已知。",
    timing: "G1＝第一期成績、G2＝第二期成績；G3 是最終目標。這是條件式成績估計，不是因果推論。",
  },
} as const;
const stages: [string, IconName, string][] = [
  ["Business Understanding", "book", "定義目標與使用時點"],
  ["Data Understanding", "dataset", "核對原始檔版本與欄位"],
  ["Data Preparation", "features", "僅在 Training fold 補值與縮放"],
  ["Modeling", "energy", "多元線性迴歸與六種選欄"],
  ["Evaluation", "evaluation", "Training CV 與獨立 Holdout 分開"],
  ["Deployment", "prediction", "研究結果快照展示，非模型部署"],
];
export async function StudyOverview({ id }: { id: StudyDatasetId }) {
  const registry = DATASET_REGISTRY[id], content = descriptions[id];
  const snapshot = await loadStudyOverview(id);
  if (snapshot.status !== "ready") return <div className="page-container" data-dataset-id={id} data-overview-state="unavailable">
    <div className="eyebrow">{registry.name}</div><h1>{registry.name} overview</h1>
    <section className="data-error" role="alert"><h2>研究資料尚未就緒</h2><p>此資料集快照缺少或未通過完整性驗證；沒有使用其他資料集替代。</p><ul>{snapshot.issues.map(issue => <li key={issue}>{issue}</li>)}</ul></section>
  </div>;
  const { dataset, protocol, models, sources } = snapshot.data;
  const ready = models.filter((m): m is StudyModel & {status:"ready"} => m.status === "ready");
  const baseline = models[0], selected = models[1];
  const metricDigits = id === "automobile-price" ? 2 : 4;
  return <div className="page-container" data-dataset-id={id} data-overview-state="ready">
    <section className="research-header"><div><div className="eyebrow"><span className="eyebrow-line"/>{dataset.name}</div>
      <h1>{dataset.name} overview<span className="heading-dot">.</span></h1><p className="page-description">Multiple Linear Regression · Target：{dataset.target} · {dataset.unit}</p></div>
      <span className="source-badge"><span className="status-dot"/>Verified snapshot · v1</span></section>
    <div className="research-context"><span className="context-icon"><Icon name="book" size={17}/></span><p>{content.question}</p>
      <a className="context-source" href={content.source} target="_blank" rel="noreferrer noopener">UCI 資料定義 <Icon name="arrow" size={15}/></a></div>
    <section className="dataset-grid" aria-label={`${dataset.name} dataset overview`}>
      <Stat label="有效資料" value={number(dataset.rows)} note={`原始 ${number(dataset.rawRows)} 筆；Target 缺失排除 ${dataset.removedTargets} 筆`} icon="dataset"/>
      <Stat label="候選特徵" value={number(dataset.candidates.length)} note={id === "automobile-price" ? "12 數值 + 2 二元類別" : "15 個數值輸入（包含 G1、G2）"} icon="features"/>
      <Stat label="Training / Holdout" value={`${number(protocol.trainRows)} / ${number(protocol.holdoutRows)}`} note={`隨機 80/20 · seed ${protocol.seed}`} icon="evaluation"/>
      <Stat label="Prediction target" value={dataset.target} note={dataset.unit} icon="prediction"/>
    </section>
    <section className="research-context" aria-label="研究摘要與使用限制"><span className="context-icon"><Icon name="info" size={19}/></span><p><strong>研究摘要</strong> {content.timing} {content.limitation}</p></section>
    <div className="overview-grid">
      <section className="panel performance-panel" aria-labelledby="study-performance-title"><div className="panel-header"><div><div className="section-kicker">MODEL EVALUATION</div><h2 id="study-performance-title">Holdout performance</h2><p>固定隨機切分 · {protocol.holdoutRows} 筆 · {dataset.unit}</p></div><span className="neutral-tag">Baseline / MI</span></div>
        {ready.length === 2 ? <PerformanceChart key={id} points={ready.map(m => ({label:m.label,...m.holdout}))} unit={dataset.unit}/> : <p className="panel-footnote" role="status">部分模型成果缺失，暫不繪製比較圖。</p>}
        <div className="metrics-table-wrap"><table className="metrics-table"><caption className="sr-only">{dataset.name} 獨立 Holdout 指標；R² 無單位且不是準確率。</caption><thead><tr><th>Model</th><th>k</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead><tbody>
          {models.map(m => <tr key={m.id}><th scope="row">{m.label}</th>{m.status === "ready" ? <><td>{m.k}</td><td>{number(m.holdout.mae,metricDigits)}</td><td>{number(m.holdout.rmse,metricDigits)}</td><td>{number(m.holdout.r2,4)}</td></> : <td colSpan={4} role="status">{m.reason}</td>}</tr>)}
        </tbody></table></div>
        <div className="panel-footnote"><Icon name="info" size={15}/><span>MAE／RMSE 使用 {dataset.unit}；R² 不是準確率。MI 由 Training CV 選定，Holdout 未參與選欄或調參。</span></div>
      </section>
      <section className="panel interval-panel" aria-labelledby="study-interval-title"><div className="panel-header"><div><div className="section-kicker">UNCERTAINTY ESTIMATION</div><h2 id="study-interval-title">95% Prediction Interval</h2><p>單筆觀測的名目區間；不是平均值 CI</p></div></div>
        <div className="interval-models">{models.map(m => <div className="interval-model" key={m.id}><div className="interval-model-title"><span><i className={`legend-dot ${m.id === "baseline" ? "baseline" : "selected"}`}/>{m.label}</span></div>
          {m.status === "ready" && m.interval.status === "ready" ? <><div className="coverage-row"><span>Holdout 實際覆蓋率</span><strong>{percent(m.interval.coverage)}</strong></div>
            <progress className={`coverage-progress ${m.id === "baseline" ? "baseline" : "selected"}`} value={m.interval.coverage} max={1} aria-label={`${m.label} empirical coverage`}/>
            <div className="width-row"><span>{m.interval.covered} / {m.interval.total} 筆覆蓋</span><strong>{number(m.interval.averageWidth,metricDigits)} <small>{dataset.unit} · 平均寬度</small></strong></div>
          </> : <p className="missing-inline" role="status">{m.status === "unavailable" ? m.reason : m.interval.status === "unavailable" ? m.interval.reason : "PI 尚未提供"}</p>}</div>)}</div>
        <div className="interval-note"><Icon name="info" size={17}/><p>Training design SVD／殘差估計的 OLS PI，未用 Holdout 校準。名目 95% 不保證未來覆蓋率；未校正選欄後不確定性或異質變異，較高覆蓋率也不代表較準。</p></div>
      </section>
    </div>
    <section className="findings-section" aria-labelledby="study-findings-title"><div className="section-heading"><div><div className="section-kicker">EVIDENCE AND LIMITS</div><h2 id="study-findings-title">研究發現與資料處理</h2></div></div>
      <div className="findings-grid"><article className="finding-card"><span className="finding-index">01 <span>TRAINING CV VS HOLDOUT</span></span><h3>選欄結果與泛化表現分開解讀</h3>
        {baseline.status === "ready" && selected.status === "ready" ? <><div className="feature-reduction"><strong>{baseline.k}</strong><Icon name="arrow" size={25}/><strong className="selected-text">{selected.k}</strong><span>features</span></div>
          <p>MI 的平均 Training CV RMSE 為 {number(selected.cv.rmse,metricDigits)}，Baseline 為 {number(baseline.cv.rmse,metricDigits)}；獨立 Holdout RMSE {selected.holdout.rmse < baseline.holdout.rmse ? "降低" : "增加"}，R² {selected.holdout.r2 > baseline.holdout.r2 ? "提高" : "下降"}，MAE {selected.holdout.mae < baseline.holdout.mae ? "降低" : "增加"}。</p>
          <details className="exploration-details"><summary>查看 MI 最終特徵與 CV 協定</summary><div className="feature-chips">{selected.features.map(f => <span key={f}>{f}</span>)}</div><p>外層 {protocol.outerFolds}-fold Training CV；SFS、LassoCV、RFECV 內層 {protocol.innerFolds}-fold。固定 k={protocol.fixedK} 預先指定，未掃描 k。MI 本身以固定 k 選欄，不做內層調參。方法取最低平均 Training CV RMSE 後鎖定；勝出者 CV 仍有選取樂觀偏差。</p></details>
        </> : <p role="status">完整比較所需模型成果尚未提供。</p>}
      </article><article className="finding-card"><span className="finding-index">02 <span>DATA PREPARATION</span></span><h3>資料定義與前處理</h3><p>{content.preparation}</p><p>{content.encoding}</p>
        <details className="exploration-details"><summary>查看全部候選特徵</summary><div className="feature-chips">{dataset.candidates.map(f => <span key={f}>{f}</span>)}</div></details>
        <div className="finding-note warning"><Icon name="info" size={15}/>此頁只展示既有實驗結果，沒有重新訓練或依 Holdout 改選模型。</div>
      </article></div>
    </section>
    <section className="panel workflow-panel" aria-labelledby="study-workflow-title"><div className="panel-header"><div><div className="section-kicker">RESEARCH METHODOLOGY</div><h2 id="study-workflow-title">CRISP-DM 六階段</h2></div></div><ol className="workflow-grid">{stages.map(([name,icon,note],i) => <li key={name}><div className="workflow-top"><span className="workflow-icon"><Icon name={icon} size={18}/></span><span className="step-number">0{i+1}</span></div><h3>{name}</h3><p>{note}</p><span className="workflow-state">{i === 5 ? "Overview only" : "Research artifacts"}</span></li>)}</ol>
      <div className="workflow-disclaimer">此資料集 Overview、Feature Selection、Model Evaluation 與 Prediction Explorer 已接入；Research Live Predictor 已於 Prediction Explorer 接入。Custom Dataset 維持獨立。</div></section>
    <details className="provenance"><summary><Icon name="check" size={15}/>Snapshot v1<span>資料來源與追溯</span></summary><div className="provenance-body"><p>只使用提供的本機檔案；未宣稱與 UCI 下載檔逐位元相同。正式快照只保存彙總數值與欄位名稱，不包含原始 CSV、逐筆預測或模型。</p><div className="provenance-item"><code>{dataset.rawFile}</code><span>SHA-256 {dataset.rawSha256}</span></div>{sources.map(s => <div key={s.path} className="provenance-item"><code>{s.path}</code><span>SHA-256 {s.sha256}</span></div>)}</div></details>
  </div>;
}
function Stat({label,value,note,icon}:{label:string;value:string;note:string;icon:IconName}) {
  return <article className="stat-card"><div className="stat-top"><span>{label}</span><span className="stat-icon"><Icon name={icon} size={18}/></span></div><div className="stat-main"><strong className="compact">{value}</strong></div><p>{note}</p></article>;
}
