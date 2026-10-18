"use client";
import { useEffect, useRef, useState } from "react";
import { parseValidationReport, type ValidationReport } from "@/lib/custom-validation-result";
import { CustomTrainingPanel } from "@/components/custom-training-panel";
import { Icon } from "@/components/icon";
import { CSV_LIMITS, emptyDraft, validatePlan, type DatasetUpload, type WizardDraft } from "@/lib/custom-dataset";
const kindLabels = { numeric: "數值", categorical: "類別", datetime: "日期", text: "文字" };
const number = (n: number, digits = 0) => n.toLocaleString("en-US", { maximumFractionDigits: digits });
function download(value: unknown, name: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2) + "\n"], { type: "application/json;charset=utf-8" }));
  const a = document.createElement("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function CustomDatasetWizard() {
  const [dataset, setDataset] = useState<DatasetUpload | null>(null);
  const [draft, setDraft] = useState(emptyDraft);
  const [busy, setBusy] = useState(false); const [error, setError] = useState<string | null>(null);
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const backendController = useRef<AbortController | null>(null); const backendGeneration = useRef(0);
  const [backendBusy, setBackendBusy] = useState(false);
  const [backendResult, setBackendResult] = useState<{ report: ValidationReport } | { error: string } | null>(null);
  const [reviewed, setReviewed] = useState<string | null>(null);
  const worker = useRef<Worker | null>(null); const timeout = useRef<ReturnType<typeof setTimeout> | null>(null); const generation = useRef(0);
  function stop() { generation.current++; worker.current?.terminate(); worker.current = null; if (timeout.current) clearTimeout(timeout.current); timeout.current = null; }
  useEffect(() => () => { backendController.current?.abort(); worker.current?.terminate(); if (timeout.current) clearTimeout(timeout.current); }, []);
  function invalidateBackend() { backendGeneration.current++; backendController.current?.abort(); backendController.current = null; setBackendBusy(false); setBackendResult(null); }
  function change(next: Partial<WizardDraft>) { invalidateBackend(); setDraft(d => ({ ...d, ...next })); setReviewed(null); }
  function clear() { invalidateBackend(); setSourceFile(null); stop(); setBusy(false); setDataset(null); setDraft(emptyDraft()); setReviewed(null); setError(null); }
  function open(files: FileList | null) {
    clear();
    if (!files?.length) return;
    if (files.length !== 1) { setError("請一次選擇一個 CSV。"); return; }
    const file = files[0];
    if (!file.name.toLowerCase().endsWith(".csv")) { setError("請選擇 .csv 檔案。"); return; }
    if (!file.size || file.size > CSV_LIMITS.bytes) { setError("CSV 必須為非空檔案，且不超過 25 MiB。"); return; }
    setSourceFile(file);
    setBusy(true); const token = generation.current;
    try {
      const next = new Worker(new URL("../workers/csv-profile.worker.ts", import.meta.url), { type: "module" }); worker.current = next;
      timeout.current = setTimeout(() => { if (generation.current === token) { stop(); setBusy(false); setError("資料檢查超過 30 秒，請縮小 CSV 後重試。"); } }, 30_000);
      next.onmessage = (event: MessageEvent<{ ok: true; data: DatasetUpload } | { ok: false; error: string }>) => {
        if (generation.current !== token) return;
        stop(); setBusy(false);
        if (event.data.ok) setDataset(event.data.data); else { setSourceFile(null); setError(event.data.error); }
      };
      next.onerror = () => { if (generation.current === token) { stop(); setBusy(false); setError("資料檢查工具無法啟動，請重新整理後再試。"); } };
      next.postMessage({ file });
    } catch { stop(); setBusy(false); setError("此瀏覽器無法啟動資料檢查，請使用支援 Web Worker 的瀏覽器。"); }
  }
  const plan = dataset ? validatePlan(dataset, draft) : null;
  const reviewKey = dataset && plan ? dataset.sha256 + JSON.stringify(plan.config) : null;
  const ready = !!plan && plan.errors.length === 0;
  const confirmed = ready && reviewed === reviewKey;
  async function validateWithPython() {
    if (!confirmed || !dataset || !plan || !sourceFile) return;
    invalidateBackend(); const token = backendGeneration.current; const controller = new AbortController(); backendController.current = controller;
    setBackendBusy(true); let expired = false;
    const timer = setTimeout(() => { expired = true; controller.abort(); }, 40_000);
    const body = new FormData(); body.append("csv", sourceFile);
    body.append("config", JSON.stringify(plan.config));
    body.append("review", JSON.stringify({ schema_version: 1, artifact_type: "custom_dataset_review", dataset: { sha256: dataset.sha256 }, preprocessing: plan.config, review: { confirmed: true, training_started: false, scope: "configuration_only" } }));
    try {
      const response = await fetch("/api/custom-dataset/validate", { method: "POST", body, signal: controller.signal, cache: "no-store" });
      const payload: unknown = await response.json().catch(() => { throw new Error("本機服務未回傳有效結果，請重新整理後重試。"); });
      if (response.status !== 200 && response.status !== 422) {
        const message = payload && typeof payload === "object" && "error" in payload && typeof payload.error === "string" ? payload.error : "本機驗證服務暫時無法完成，請稍後重試。";
        throw new Error(message);
      }
      const report = parseValidationReport(payload);
      if ((response.status === 200) !== (report.status === "valid")) throw new Error("驗證回應狀態不一致，請重新驗證。");
      if (report.status === "valid" && (report.dataset.sha256 !== dataset.sha256 || report.dataset.rows !== dataset.rows || report.dataset.columns !== dataset.columns.length || !report.review_checked)) throw new Error("Python 與瀏覽器的資料版本或筆數不一致，請重新選檔檢查。");
      if (token === backendGeneration.current) setBackendResult({ report });
    } catch (e) {
      if (token === backendGeneration.current) setBackendResult({ error: expired ? "驗證等待逾時，請稍後重試或縮小資料。" : e instanceof Error && e.name !== "AbortError" ? e.message : "驗證已取消。" });
    } finally { clearTimeout(timer); if (token === backendGeneration.current) { setBackendBusy(false); backendController.current = null; } }
  }
  const available = dataset?.columns.filter(c => c.kind === "numeric" && !c.infinite && !c.invalidNumeric && c.unique > 1 && c.name !== draft.target) ?? [];
  const dates = dataset?.columns.filter(c => c.dates) ?? [];
  return <>
    <section className="panel cd-upload" aria-labelledby="upload-title"><div className="panel-header"><div><div className="section-kicker">01 · CHOOSE YOUR DATA</div><h2 id="upload-title">選擇 CSV，先理解資料</h2><p>選檔後先在瀏覽器檢查；只有按下驗證或訓練按鈕才會傳送至本機服務。</p></div><span className="neutral-tag">UTF-8 · CSV</span></div>
      <div className="cd-body"><div className={`cd-dropzone ${busy ? "cd-busy" : ""}`} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); open(e.dataTransfer.files); }}>
        <Icon name="dataset" size={30}/><h3>拖曳一個 CSV 到這裡</h3><p>逗號分隔，首列為欄名；支援引號、換行內容與 UTF-8 BOM。</p><label className="cd-file-button">選擇 CSV 檔案<input id="dataset-file" type="file" accept=".csv,text/csv" aria-label="選擇 CSV 檔案" onChange={e => { open(e.currentTarget.files); e.currentTarget.value = ""; }}/></label>
        <small>上限 25 MiB、50,000 筆、200 欄、100 萬個儲存格。</small></div>
        {busy && <div className="cd-status" role="status">正在檢查 CSV…<button onClick={clear}>取消檢查</button></div>}
        {error && <p className="cd-error" role="alert">{error}</p>}
        {!dataset && !busy && !error && <p className="fs-note">先選檔案，再由你指定目標與特徵。完成設定後先送至本機 Python 驗證，再另外確認開始訓練。</p>}
      </div>
    </section>
    {dataset && plan && <>
      <section className="panel cd-profile" aria-labelledby="profile-title"><div className="panel-header"><div><div className="section-kicker">02 · REVIEW DATA QUALITY</div><h2 id="profile-title">資料品質與欄位</h2><p className="cd-filename">{dataset.fileName} · {number(dataset.bytes / 1024, 1)} KiB</p></div><button className="cd-button" onClick={clear}>清除資料</button></div><div className="cd-body">
        <div className="cd-stats"><div><strong>{number(dataset.rows)}</strong><span>資料筆數</span></div><div><strong>{dataset.columns.length}</strong><span>原始欄位（含目標）</span></div><div><strong>{number(dataset.columns.reduce((s, c) => s + c.missing, 0))}</strong><span>缺失儲存格</span></div><div><strong>{number(dataset.duplicateRows)}</strong><span>完全重複紀錄</span></div></div>
        <p className="fs-note">型態為瀏覽器的保守推測。空字串及常見 NA／NaN／null 記號視為缺失；純空白字串保留為內容。前置零與超出安全範圍的整數保留文字語意。IQR 候選僅作描述，未刪除或改寫任何資料。</p>
        <div className="fs-table-scroll"><table className="metrics-table cd-profile-table"><caption>完整 CSV 的描述性統計；不作為訓練用的補值、縮放或裁切門檻</caption><thead><tr><th>欄位</th><th>型態</th><th>缺失</th><th>不同值</th><th>最小／最大</th><th>中位數</th><th>IQR 候選</th><th>需留意</th></tr></thead><tbody>{dataset.columns.map(c => <tr key={c.name}><th scope="row">{c.name}</th><td>{kindLabels[c.kind]}</td><td>{c.missing}（{number(c.missing / dataset.rows * 100, 1)}%）</td><td>{number(c.unique)}</td><td>{c.numeric ? `${number(c.numeric.min, 3)}／${number(c.numeric.max, 3)}` : "—"}</td><td>{c.numeric ? number(c.numeric.median, 3) : "—"}</td><td>{c.numeric?.outliers ?? "—"}</td><td>{c.warnings.join("；") || "—"}</td></tr>)}</tbody></table></div>
        <details className="cd-preview"><summary>查看前 5 筆原始內容預覽</summary><p className="fs-note">每格最多顯示 200 字元，未轉換資料；以下內容僅留在本瀏覽器。</p><div className="fs-table-scroll"><table className="metrics-table"><caption>原始 CSV 前 {dataset.preview.length} 筆</caption><thead><tr>{dataset.columns.map(c => <th key={c.name}>{c.name}</th>)}</tr></thead><tbody>{dataset.preview.map((row, i) => <tr key={i}>{row.map((value, j) => <td key={j}>{value === "" ? "（空）" : value}</td>)}</tr>)}</tbody></table></div></details>
      </div></section>
      <section className="panel cd-configuration" aria-labelledby="configuration-title"><div className="panel-header"><div><div className="section-kicker">03 · DEFINE THE QUESTION</div><h2 id="configuration-title">目標、特徵與前處理</h2><p>目前支援數值迴歸設定；型態不合的欄位會列出原因，保留由你判斷。</p></div></div><div className="cd-body">
        <div className="cd-config-grid"><label htmlFor="dataset-target">要預測的目標欄位<select id="dataset-target" value={draft.target} onChange={e => { const target = e.target.value; change({ target, features: draft.features.filter(f => f !== target) }); }}><option value="">請自行選擇目標</option>{dataset.columns.map(c => <option key={c.name} value={c.name} disabled={!c.targetEligible}>{c.name}{!c.targetEligible ? "（不適用：需非恆定有限數值）" : ""}</option>)}</select></label>
          <label htmlFor="dataset-missing-target">目標缺失時<select id="dataset-missing-target" value={draft.missingTarget} onChange={e => change({ missingTarget: e.target.value as WizardDraft["missingTarget"] })}><option value="error">拒絕，先修正資料</option><option value="drop">在切分前排除目標缺失列</option></select></label></div>
        <fieldset className="cd-features"><legend>輸入特徵 · 已選 {draft.features.length} 個原始欄位</legend><div className="cd-feature-actions"><button className="cd-button" disabled={!draft.target} onClick={() => change({ features: available.filter(c => !c.possibleId).map(c => c.name) })}>選取可用數值欄（排除疑似 ID）</button><button className="cd-button" onClick={() => change({ features: [] })}>全部取消</button></div><div className="cd-feature-grid">{dataset.columns.map(c => { const eligible = !!draft.target && available.some(v => v.name === c.name); return <label key={c.name} className={eligible ? "" : "cd-unavailable"}><input type="checkbox" checked={draft.features.includes(c.name)} disabled={!eligible} onChange={e => change({ features: e.target.checked ? [...draft.features, c.name] : draft.features.filter(f => f !== c.name) })}/><span>{c.name}<small>{c.name === draft.target ? "預測目標" : c.kind !== "numeric" ? `${kindLabels[c.kind]} · 不作本階段輸入` : c.infinite || c.invalidNumeric ? "含無效數值" : c.unique <= 1 ? "常數／全缺失" : c.possibleId ? "疑似 ID，請確認" : "數值"}</small></span></label>; })}</div></fieldset>
        <div className="cd-config-grid"><label htmlFor="dataset-missing-numeric">數值特徵缺失<select id="dataset-missing-numeric" value={draft.missingNumeric} onChange={e => change({ missingNumeric: e.target.value as WizardDraft["missingNumeric"] })}><option value="median">訓練 fold 中位數補值</option><option value="mean">訓練 fold 平均值補值</option><option value="none">不補值（遇缺失拒絕）</option></select></label><label htmlFor="dataset-scaling">數值縮放<select id="dataset-scaling" value={draft.scaling} onChange={e => change({ scaling: e.target.value as WizardDraft["scaling"] })}><option value="standard">StandardScaler · 僅訓練 fold 學習</option><option value="none">不縮放</option></select></label><label htmlFor="dataset-outliers">輸入極端值<select id="dataset-outliers" value={draft.outliers} onChange={e => change({ outliers: e.target.value as WizardDraft["outliers"] })}><option value="keep">保留原值</option><option value="clip">以訓練 fold 的 1.5 IQR 裁切</option></select></label><label htmlFor="dataset-split">資料切分<select id="dataset-split" value={draft.split} onChange={e => change({ split: e.target.value as WizardDraft["split"], gap: 0 })}><option value="random">隨機切分 · seed 42</option><option value="chronological">依時間前後切分</option></select></label>
          {draft.split === "chronological" && <label htmlFor="dataset-time-column">排序時間欄位<select id="dataset-time-column" value={draft.splitColumn} onChange={e => change({ splitColumn: e.target.value })}><option value="">請選擇 ISO 日期欄</option>{dates.map(c => <option key={c.name} value={c.name}>{c.name}{c.missing || c.dates!.unique !== dataset.rows ? "（含缺失或重複）" : ""}</option>)}</select></label>}
          <label htmlFor="dataset-test-size">保留測試資料比例<select id="dataset-test-size" value={draft.testSize} onChange={e => change({ testSize: Number(e.target.value) })}>{[0.2, 0.25, 0.3].map(v => <option key={v} value={v}>{v * 100}%</option>)}</select></label><label htmlFor="dataset-cv">訓練區內 CV folds<input id="dataset-cv" type="number" min={2} max={10} step={1} value={Number.isNaN(draft.cvSplits) ? "" : draft.cvSplits} onChange={e => change({ cvSplits: e.currentTarget.valueAsNumber })}/></label>{draft.split === "chronological" && <label htmlFor="dataset-gap">CV gap（觀測列）<input id="dataset-gap" type="number" min={0} max={50_000} step={1} value={Number.isNaN(draft.gap) ? "" : draft.gap} onChange={e => change({ gap: e.currentTarget.valueAsNumber })}/></label>}</div>
        <p className="fs-note">上述是待執行的規則，尚未補值、縮放、裁切或切分。正式流程必須在每個訓練 fold 內重新學習規則；類別編碼與日期衍生尚未串接。原始欄位數與未來選特徵的 k 分開計算。</p>
      </div></section>
      <section className="panel cd-review" aria-labelledby="review-title"><div className="panel-header"><div><div className="section-kicker">04 · REVIEW BEFORE ANY EXPERIMENT</div><h2 id="review-title">確認與下載設定</h2><p>下載設定供後續流程使用；Python 驗證通過後，可另外確認並開始本機訓練。</p></div><span className="neutral-tag">設定預覽</span></div><div className="cd-body">
        <div className="cd-review-summary"><p>目標：<strong>{draft.target || "尚未選擇"}</strong> · 輸入：<strong>{draft.features.length} 欄</strong></p><p>{plan.usableRows} 筆可用於規劃；預計訓練 {Number.isFinite(plan.trainRows) ? plan.trainRows : "—"} 筆／測試 {Number.isFinite(plan.testRows) ? plan.testRows : "—"} 筆。只計入指定的目標缺失排除；正式流程仍須逐 fold 驗證型態、變異與可用筆數。</p></div>
        {plan.errors.length > 0 && <div className="cd-validation" role="status"><strong>請先處理以下項目</strong><ul>{plan.errors.map(message => <li key={message}>{message}</li>)}</ul></div>}
        {plan.warnings.length > 0 && <div className="cd-validation cd-cautions"><strong>確認資料意義與評估限制</strong><ul>{plan.warnings.map(message => <li key={message}>{message}</li>)}</ul></div>}
        <details className="cd-json"><summary>查看完整前處理設定 JSON</summary><pre>{JSON.stringify(plan.config, null, 2)}</pre></details>
        <div className="cd-download-actions"><button className="cd-button cd-primary" id="dataset-confirm" disabled={!ready || confirmed} onClick={() => setReviewed(reviewKey)}>{confirmed ? "已確認目前設定" : "我已檢查，確認這份設定"}</button><button className="cd-button" id="dataset-download" disabled={!confirmed} onClick={() => download(plan.config, "custom-dataset-config.json")}>下載前處理設定</button><button className="cd-button" id="dataset-profile-download" disabled={!confirmed} onClick={() => { const { preview: omitted, ...summary } = dataset; void omitted; download({ schema_version: 1, artifact_type: "custom_dataset_review", dataset: summary, preprocessing: plan.config, review: { confirmed: true, training_started: false, scope: "configuration_only" } }, "custom-dataset-review.json"); }}>下載檢查摘要</button></div>
        <p className="cd-confirmation" role="status">{confirmed ? "此資料版本與設定已由你確認。修改任何設定後，需重新確認才能下載。" : "尚未確認；下載按鈕會在設定有效且經你確認後啟用。"}</p>
        <details className="cd-provenance"><summary>資料版本與檢查範圍</summary><p>CSV SHA-256：<code>{dataset.sha256}</code></p><p>摘要包含欄位統計與設定，不包含原始資料列或預覽內容。缺失記號：空字串、#N/A、#N/A N/A、#NA、-1.#IND、-1.#QNAN、-NaN、-nan、1.#IND、1.#QNAN、&lt;NA&gt;、N/A、NA、NULL、NaN、None、n/a、nan、null。</p><p>日期只辨識無時區的 ISO 年月日與時分秒，不猜日／月順序；正式 Python 流程仍會重新驗證。此摘要僅記錄設定，不代表已完成訓練；實際成績見下方訓練結果，不會改寫原始 CSV 或既有研究成果。</p></details>
      </div></section>
      <section className="panel cd-python" aria-labelledby="python-validation-title" aria-busy={backendBusy}><div className="panel-header"><div><div className="section-kicker">05 · CHECK WITH PYTHON</div><h2 id="python-validation-title">本機 Python 驗證</h2><p>確認設定後，由 Python 實際檢查資料與每個 CV fold；不會訓練模型。</p></div><span className="neutral-tag">訓練前檢查</span></div><div className="cd-body">
        <p>按下驗證會將目前 CSV、設定與版本確認送到本機網站服務。資料只供這次檢查，結束後自動清除伺服器暫存檔；不保存上傳資料或模型。</p>
        <div className="cd-download-actions"><button id="dataset-python-validate" className="cd-button cd-primary" disabled={!confirmed || backendBusy} onClick={() => { void validateWithPython(); }}>{backendBusy ? "Python 驗證中…" : "傳送至本機 Python 驗證"}</button>{backendBusy && <button id="dataset-python-cancel" className="cd-button" onClick={() => { invalidateBackend(); setBackendResult({ error: "已取消本次 Python 驗證。" }); }}>取消驗證</button>}</div>
        {!confirmed && <p className="fs-note">請先完成上方的設定確認。更改資料或設定後，舊驗證結果會清除。</p>}
        {backendBusy && <p className="cd-confirmation" role="status">正在檢查資料、版本與時間／CV 切分，通常需要幾秒鐘…</p>}
        {backendResult && <div className="cd-python-result" role="status" aria-live="polite">{"error" in backendResult ? <p className="cd-error">{backendResult.error}</p> : <PythonResult report={backendResult.report}/>}</div>}
      </div></section>
      {confirmed && sourceFile && backendResult && "report" in backendResult && backendResult.report.status === "valid" && <CustomTrainingPanel key={backendResult.report.config_sha256} file={sourceFile} config={plan.config} validation={backendResult.report}/>}
    </>}
  </>;
}

function PythonResult({ report }: { report: ValidationReport }) {
  const stages: Record<string, string> = { config: "設定", csv: "CSV 格式", data: "欄位資料", review: "資料版本與確認", split: "資料切分與 CV" };
  if (report.status === "invalid") return <div className="cd-validation"><h3>Python 驗證未通過</h3><ul>{report.errors.map((issue, i) => <li key={i}><strong>{stages[issue.stage]}：</strong>{issue.message}</li>)}</ul><p>請依提示調整資料或設定，重新確認後再驗證；尚未啟動訓練。</p></div>;
  return <><div className="cd-python-passed"><h3>Python 訓練前驗證通過</h3><p>CSV 與已確認設定版本一致。此結果代表資料可進入後續流程，不代表模型準確度或訓練已完成。</p></div>
    <div className="cd-stats cd-python-counts"><div><strong>{number(report.split.training_rows)}</strong><span>Python 訓練區筆數</span></div><div><strong>{number(report.split.holdout_rows)}</strong><span>Python 保留測試筆數</span></div><div><strong>{report.split.target_missing_dropped}</strong><span>目標缺失排除筆數</span></div><div><strong>{report.split.cv_folds.length}</strong><span>CV folds</span></div></div>
    <div className="fs-table-scroll"><table className="metrics-table cd-fold-table"><caption>Python 實際建立的索引筆數 · {report.split.strategy === "chronological" ? "時間切分" : "隨機切分"}，未執行前處理或訓練</caption><thead><tr><th>Fold</th><th>訓練筆數</th><th>驗證筆數</th></tr></thead><tbody>{report.split.cv_folds.map(f => <tr key={f.fold}><th scope="row">Fold {f.fold}</th><td>{number(f.training_rows)}</td><td>{number(f.validation_rows)}</td></tr>)}</tbody></table></div>
    {report.warnings.length > 0 && <div className="cd-validation cd-cautions"><strong>仍需留意</strong><ul>{report.warnings.map((message, i) => <li key={i}>{message}</li>)}</ul></div>}
    <details className="cd-provenance"><summary>驗證版本與範圍</summary><p>CSV SHA-256：<code>{report.dataset.sha256}</code></p><p>設定 SHA-256：<code>{report.config_sha256}</code></p><ul>{report.limits.map((message, i) => <li key={i}>{message}</li>)}</ul></details></>;
}
