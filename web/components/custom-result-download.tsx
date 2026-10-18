"use client";
import { useState } from "react";
import type { PreprocessingPlan } from "@/lib/custom-dataset";
import type { TrainingReport } from "@/lib/custom-training-result";
import { downloadArtifact } from "@/lib/browser-download";
import { evaluationJson, evaluationCsv, previewCsv, type DownloadArtifact } from "@/lib/custom-result-export";
export function CustomResultDownload({report,config}:{report:Extract<TrainingReport,{status:"completed"}>;config:PreprocessingPlan}) {
  const [notice,setNotice]=useState<string|null>(null);const [error,setError]=useState(false);
  function download(build:()=>DownloadArtifact){try{downloadArtifact(build());setError(false);setNotice("已產生下載檔案，請查看瀏覽器的下載項目。");}catch{setError(true);setNotice("無法產生下載檔案，請檢查目前結果或重試。");}}
  return <section className="cd-result-download" aria-labelledby="custom-download-title"><div className="section-kicker">SAVE YOUR RESULTS</div><h3 id="custom-download-title">下載本次結果</h3><p>保存目前這次實驗，不會重新訓練。評估指標使用全部 Holdout；預覽 CSV 只包含畫面保留的前 {report.preview.length}／{report.holdout.rows.toLocaleString()} 筆，最多 144 筆。</p><div className="cd-download-actions"><button id="custom-evaluation-json" className="cd-button" onClick={()=>download(()=>evaluationJson(report,config))}>下載評估報告 JSON</button><button id="custom-metrics-csv" className="cd-button" onClick={()=>download(()=>evaluationCsv(report))}>下載評估指標 CSV</button><button id="custom-preview-csv" className="cd-button" onClick={()=>download(()=>previewCsv(report))}>下載預覽 CSV（{report.preview.length}／{report.holdout.rows.toLocaleString()} 筆）</button></div><p className="cd-confirmation">報告包含前處理設定、選中特徵、每 fold 與完整 Holdout 指標；不含原始 CSV、模型或模型代碼。預覽檔另含實際目標與預測值。CSV 文字欄位若可能被試算表當成公式，會加上單引號作為純文字；數字不變。</p>{notice&&<p className={error?"cd-error":"cd-confirmation"} role={error?"alert":"status"}>{notice}</p>}</section>;
}
