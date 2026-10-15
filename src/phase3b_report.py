"""由正式結果 CSV 產生圖表，保持各 CV 評估層級分離。"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_selection import METHODS
from .phase2_common import ROOT, configure_matplotlib
from .phase3b_runner import OUTPUT, atomic_json

LABELS = {"f_test": "F-test", "mutual_information": "Mutual Information",
          "forward_sfs": "Forward SFS", "rfe": "RFE", "lasso_cv": "LassoCV", "rfecv": "RFECV"}


def export_csv(records, path):
    fields = ["level", "method", "requested_k", "selected_k", "selected_feature_names",
              "selected_original_features", "selected_feature_indices", "rmse", "mae", "r2",
              "outer_fold", "inner_fold", "runtime_seconds", "status", "train_range",
              "validation_range", "random_state", "preprocessing_configuration", "checkpoint_id"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            selection = record["selection"]
            original = [name for name, mapped in selection.get("original_feature_mapping", {}).items() if mapped]
            row = {key: record.get(key) for key in fields}
            row.update({"selected_feature_names": selection.get("selected_feature_names", []),
                        "selected_original_features": original,
                        "selected_feature_indices": selection.get("selected_feature_indices", []),
                        "preprocessing_configuration": "experiment_config.json#preprocessing"})
            for key in ("selected_feature_names", "selected_original_features", "selected_feature_indices", "train_range", "validation_range"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)


def aggregate(frame, level, method, requested_k=None):
    selected = frame[(frame.level == level) & (frame.method == method)]
    if requested_k is not None:
        selected = selected[selected.requested_k == requested_k]
    expected = 5 if level in {"outer_evaluation", "outer_k_curve"} else 3
    usable = len(selected) == expected and (selected.status == "ok").all()
    return {"method": method, "level": level, "requested_k": requested_k,
            "folds": len(selected), "valid_folds": int((selected.status == "ok").sum()),
            "selected_k": [float(k) if pd.notna(k) else None for k in selected.selected_k],
            **{metric: float(selected[metric].mean()) if usable else None for metric in ("rmse", "mae", "r2")}}


def generate_outputs(payload):
    export_csv(payload["records"], OUTPUT / "feature_selection_results.csv")
    frame = pd.read_csv(OUTPUT / "feature_selection_results.csv")
    nested = [aggregate(frame, "outer_evaluation", method) for method in METHODS]
    runtime = {method: {"worker_seconds": sum(c.get("runtime_seconds", 0.) for c in payload["cases"] if c["case"]["method"] == method),
                        "wall_seconds_including_startup": sum(c["wall_seconds"] for c in payload["cases"] if c["case"]["method"] == method),
                        "csv_fit_seconds": float(frame[frame.method == method].drop_duplicates(["checkpoint_id", "requested_k"]).runtime_seconds.sum()),
                        "cases": sum(c["case"]["method"] == method for c in payload["cases"]),
                        "timeouts": sum(c["status"] == "timeout" and c["case"]["method"] == method for c in payload["cases"])} for method in METHODS}
    atomic_json(OUTPUT / "runtime_comparison.json", {"level": "all executed formal training-only stages, shared path costs counted once",
                                                    "pilot_excluded": True, "methods": runtime})
    best = {"selection_level": "final_selection_cv (3 folds on Training only)",
            "winner": payload["winner"], "by_method": payload["finalists"],
            "full_training_refits": payload["final_refits"], "nested_outer_assessment": nested,
            "model_path": "reports/phase3/results/models/final_linear_regression.joblib",
            "predictor_class": "sklearn.linear_model.LinearRegression",
            "holdout_used": False, "warning": "Method choice uses Training CV; outer scores describe each method, not an unbiased estimate of choosing the best of six."}
    atomic_json(OUTPUT / "best_features.json", best)
    plot_manifest = make_plots(frame, payload, runtime)
    atomic_json(OUTPUT / "plot_data.json", plot_manifest)
    write_report(payload, nested, runtime)


def make_plots(frame, payload, runtime):
    configure_matplotlib()
    import matplotlib.pyplot as plt
    figures = ROOT / "reports/phase3/figures"
    figures.mkdir(parents=True, exist_ok=True)
    manifest = {"csv_sha256": hashlib.sha256((OUTPUT / "feature_selection_results.csv").read_bytes()).hexdigest(),
                "figures": {}, "aggregation": "arithmetic mean across declared folds, eligible only when all folds valid"}
    colors = dict(zip(METHODS, plt.get_cmap("tab10").colors[:6]))

    def save(fig, name, data):
        fig.tight_layout()
        fig.savefig(figures / name, dpi=150, metadata={"Software": "AIOT-HW2 Phase 3B"})
        plt.close(fig)
        manifest["figures"][name] = data

    # Per-k curves are descriptive outer-fold comparisons, never used to choose k.
    for metric in ("rmse", "r2", "mae"):
        fig, ax = plt.subplots(figsize=(10, 6))
        data = []
        for method in METHODS:
            if method in {"lasso_cv", "rfecv"}:
                point = aggregate(frame, "outer_evaluation", method)
                data.append(point)
                if point[metric] is not None:
                    ax.scatter(np.mean(point["selected_k"]), point[metric], color=colors[method],
                               marker="D", s=80, label=LABELS[method] + " (auto k, mean)")
            else:
                points = [aggregate(frame, "outer_k_curve", method, k) for k in payload["config"]["grids"][method]]
                data.extend(points)
                ax.plot([p["requested_k"] for p in points],
                        [p[metric] if p[metric] is not None else np.nan for p in points],
                        linestyle="none" if method == "forward_sfs" else "-", marker="o", markersize=4,
                        color=colors[method], label=LABELS[method])
        reference = aggregate(frame, "outer_k_curve", "f_test", 25)
        if reference[metric] is not None:
            ax.axhline(reference[metric], color="black", linestyle="--", label="All 25 features: same outer CV")
        ax.set(title=f"{metric.upper()} vs feature count — descriptive outer CV (5 folds)",
               xlabel="Feature count (auto methods: mean selected k)", ylabel=metric.upper())
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        save(fig, f"{metric}_vs_feature_count.png", {"points": data, "baseline": reference,
                                                     "sfs_interpolation": False})

    fig, ax = plt.subplots(figsize=(10, 5))
    k_data = []
    for i, method in enumerate(METHODS):
        rows = frame[(frame.level == "outer_evaluation") & (frame.method == method)]
        values = rows.selected_k.tolist()
        k_data.append({"method": method, "selected_k": values, "level": "outer_evaluation"})
        ax.scatter(np.repeat(i, len(values)), values, marker="o", color=colors[method], label=LABELS[method])
    ax.axhline(25, color="black", linestyle="--", label="All-feature baseline (k=25)")
    ax.set(xticks=range(6), xticklabels=[LABELS[m] for m in METHODS], ylabel="Selected feature count",
           title="Inner-selected k in each outer fold (5 folds)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=7, ncol=3)
    save(fig, "best_k_comparison.png", {"points": k_data})

    config = json.loads((OUTPUT / "experiment_config.json").read_text())
    names = config["preprocessing"]["feature_columns"]
    final = frame[frame.level == "final_refit_training"]
    matrix = np.zeros((6, 25))
    selections = []
    for i, method in enumerate(METHODS):
        rows = final[final.method == method]
        if len(rows):
            indices = json.loads(rows.iloc[0].selected_feature_indices)
            matrix[i, indices] = 1
            selections.append({"method": method, "indices": indices})
    fig, ax = plt.subplots(figsize=(13, 5))
    from matplotlib.patches import Patch
    ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set(xticks=range(25), xticklabels=names, yticks=range(6), yticklabels=[LABELS[m] for m in METHODS],
           title="Selected features after full-Training refit (no validation scores)",
           xlabel="Original feature", ylabel="Method")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(handles=[Patch(color=plt.get_cmap("Blues")(1.), label="Selected")], loc="upper right")
    save(fig, "selected_features_comparison.png", {"level": "final_refit_training", "selections": selections})

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([LABELS[m] for m in METHODS], [runtime[m]["csv_fit_seconds"] for m in METHODS],
           color=[colors[m] for m in METHODS], label="CSV fit time (shared selection counted once)")
    ax.set(title="Total fit runtime across formal training-only stages (pilot excluded)", ylabel="Seconds", xlabel="Method")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8)
    save(fig, "runtime_comparison.png", {"level": "all formal stages", "methods": runtime,
                                         "source": "CSV runtime_seconds, deduplicated checkpoint_id/requested_k"})

    nested = [aggregate(frame, "outer_evaluation", m) for m in METHODS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ("rmse", "mae", "r2")):
        ax.bar([LABELS[m] for m in METHODS], [p[metric] if p[metric] is not None else np.nan for p in nested],
               color=[colors[m] for m in METHODS], label="Mean outer score; k chosen in inner CV")
        reference = aggregate(frame, "outer_k_curve", "f_test", 25)
        if reference[metric] is not None:
            ax.axhline(reference[metric], color="black", linestyle="--", label="All 25 features: same outer CV")
        ax.set(title=f"Nested CV: {metric.upper()}", ylabel=metric.upper(), xlabel="Method")
        ax.tick_params(axis="x", rotation=70)
        ax.legend(fontsize=6)
    save(fig, "nested_cv_performance_comparison.png", {"level": "outer_evaluation", "points": nested})
    return manifest


def write_report(payload, nested, runtime):
    winner = payload["winner"]
    lines = ["# Phase 3B — 正式特徵選擇實驗", "",
             "本階段所有建模、選模與 Benchmark 只使用既有 chronological Training 的 15,788 筆，不讀取 3,947 筆 Holdout Target。Phase 1 回歸驗證另唯讀檢查全檔資料品質，其結果不參與選模。最終 Predictor 維持 sklearn LinearRegression。", "",
             "## 方法與實驗設計", "",
             "原始 28 候選輸入排除 date、rv1、rv2，保留 25 個 numeric Features。F-test／MI／RFE 使用 k=1..25；SFS 使用預先固定的 [1,3,5,8,10,13,16,20,25]；LassoCV／RFECV 自動決定 k。k=25 明確全選，不冒充 SFS 搜尋。", "",
             "Outer TimeSeriesSplit=5；各 Outer Training 內以 Inner TimeSeriesSplit=3 選 k。gap=6。SFS 與自動方法另在當前可見的 Training 內使用 3-fold selector CV。所有方法共用相同切分；中位數補值、StandardScaler 與 Selector 只在當前 Training fit。沿用 Phase 2 前處理及 Phase 3A fold-local estimator。", "",
             "各 k 的 Outer Curve 是描述性比較；正式 Outer 評估只取事先由 Inner CV 決定的 k。另在完整 Training 上進行 3-fold final_selection_cv，決定最終方法及 k；最小平均 RMSE 為主，差距 ≤1e-9 時依較少平均 Features、較短 fit time、宣告方法順序決定。規則在正式分數產生前已寫入設定。選六種方法的程序本身未再經獨立最外層評估，不能將最佳方法的 Outer 成績宣稱為整個選模程序的無偏估計。", "",
             "## Training CV 選模結果", "",
             "以下是 final_selection_cv，與後面的 Nested Outer 表格層級不同。", "",
             "| 方法 | 要求 k／auto | 平均實選 k | RMSE | MAE | R² |", "|---|---:|---:|---:|---:|---:|"]
    for row in payload["finalists"]:
        lines.append(f"| {LABELS[row['method']]} | {row['requested_k'] or 'auto'} | {row['mean_selected_k']:.3f} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['r2']:.6f} |")
    baseline_rows = [r for r in payload["records"] if r["level"] == "final_selection_cv" and r["method"] == "f_test" and r["requested_k"] == 25]
    lines += ["", "同一 final_selection_cv 的全 25 Features 參考：" +
              "、".join(f"{m.upper()}={np.mean([r[m] for r in baseline_rows]):.6f}" for m in ("rmse", "mae", "r2")) + "。", "",
              "## Nested Outer 評估", "", "| 方法 | 各 Fold 實選 k | RMSE | MAE | R² |", "|---|---|---:|---:|---:|"]
    for row in nested:
        fmt = lambda v: f"{v:.6f}" if v is not None else "未完成／invalid"
        lines.append(f"| {LABELS[row['method']]} | {row['selected_k']} | {fmt(row['rmse'])} | {fmt(row['mae'])} | {fmt(row['r2'])} |")
    lines += ["", "## 最終模型與 Selected Features", "",
              f"依 Training CV 選定 **{LABELS[winner['method']]}**；要求 k={winner['requested_k'] or 'auto'}。各方法均在完整 Training 重新 fit Selector 及獨立 LinearRegression；最終模型位於 `results/models/final_linear_regression.joblib`，包含可重用前處理與 Selector。保存後已做載入／預測一致性檢查。", ""]
    for record in payload["final_refits"]:
        names = [n for n, values in record["selection"].get("original_feature_mapping", {}).items() if values]
        lines.append(f"- {LABELS[record['method']]}：實選 k={record['selected_k']}；{', '.join(names)}。")
    lines += ["", "完整 Training 的 refit 分數標記為 `final_refit_training`，僅為訓練內分數，未用來選模。", "",
              "## 計算時間、SFS 與續跑", "",
              "SFS 先在完整 15,788 筆上執行一次完整預定 grid 的成本 pilot，不保存或採用 pilot 的選擇／分數。正式 gate 估算涵蓋 24 個 partition，使用 full-n 實測成本乘 2 安全係數。詳見 `results/sfs_budget.json`。只在同一 Training Partition 共用確定性的 filter scores、RFE elimination path 與 sklearn SFS greedy prefixes；測試對照逐 k 原始 Selector。未依分數刪除任何 k，也不跨 Fold 分享規則。", "",
              "每個 method/partition 由獨立 subprocess 執行，120 秒硬期限，以 SIGKILL 終止 process group 並 reap；SFS 總啟動預算 3,600 秒。結果即時原子寫入 `.venv/phase3b_checkpoints/`，含 code/config/environment/data/range fingerprint；再次執行跳過所有 terminal cases，含 timeout，不自動重試。checkpoint 是本機暫存，不納入版本追蹤；正式 JSON 保存各 case 識別與時序。", "",
              "CSV 每列 runtime_seconds 將同一 grid 的共享 fit 耗時平均分攤至各 k，再加上各自 LR fit 耗時；不代表逐 k 獨立搜尋的成本。Runtime 圖由 CSV 去除重複的 Outer 評估列後加總，Ledger 另保存含評分與程序啟動的實測耗時。各方法 grid 工作量不同，圖中呈現的是這份正式實驗的總計算成本。", ""]
    for method in METHODS:
        lines.append(f"- {LABELS[method]}：Worker {runtime[method]['worker_seconds']:.3f} 秒，含程序啟動 {runtime[method]['wall_seconds_including_startup']:.3f} 秒；Cases {runtime[method]['cases']}，Timeout {runtime[method]['timeouts']}。")
    warning_cases = [c for c in payload["cases"] if c.get("warnings")]
    convergence = sum(w["category"] == "ConvergenceWarning" for c in payload["cases"] for w in c.get("warnings", []))
    invalid = [r for r in payload["records"] if r["status"] == "invalid_for_LR"]
    lines += ["", "## LassoCV 收斂與失敗紀錄", "",
              f"StandardScaler 必用；alpha grid=[100,10,1,0.1,0.01,0.001]，tol=1e-5，support 門檻=1e-8。針對 Phase 3A 的 7 個 ConvergenceWarning，正式執行前將 max_iter 由 10,000 提升至 100,000，alpha/tol 未變，僅以 Training 診斷；正式共 {convergence} 個 ConvergenceWarning、{len(warning_cases)} 個帶警告 Case。警告全文保存於 JSON。LassoCV 內部遵循 sklearn 的平均 MSE 決定 alpha；SFS／RFECV 內部使用 neg-RMSE。各方法之間及固定 k 的正式比較一律使用相同的 RMSE、MAE、R²。", "",
              f"k=0 的 {len(invalid)} 筆記錄保留 selected_k=0、status=invalid_for_LR 與 null LR 指標；未補成 1，未計算不存在的 LR 分數。任何失敗 Fold 都使該候選不可參與最佳值比較。LassoCV 在各內層 Fold 分別 fit scaler 的 blocks 上決定 alpha；其混合尺度的內建 refit 不用於選擇或預測，support 另以當前完整 Training 的 scaler+Lasso 重算，最終預測仍用 LR。", "",
              "## Leakage、Baseline 與 Holdout 限制", "",
              "資料 loader 僅 read_csv(nrows=15788)；整檔 SHA-256 僅檢查位元組完整性。Case range 必須位於 [0,15788)，Validation 與 Training 保持 gap=6。所有補值、縮放、特徵選擇都在目前 Training Partition 學習，Inner 選 k 不讀取 Outer 分數。固定同一 random_state=42、單一 BLAS thread。", "",
              "圖中的 Baseline 是相同 Training Outer CV 的 k=25 LR，不是 Phase 2 Holdout 分數。Phase 2 chronological Holdout RMSE 86.732939、MAE 52.014105、R² 0.092352 僅作既有歷史背景，無法與不同切分的本階段 CV 數值直接排名；原始 Baseline Metrics 與模型保持位元組不變。Phase 2 已查看 Holdout，不能將其稱為未查看的最終測試集。", "",
              "## 輸出、驗證與 Phase 3C 準備", "",
              "`results/feature_selection_results.csv/.json` 分開保存 inner_k_cv、outer_k_curve、outer_evaluation、final_selection_cv、final_refit_training；另存 Inner 選擇結果、實選 Features、Runtime 與設定／環境。七張圖位於 `figures/`；各圖資料與 CSV SHA-256 在 `results/plot_data.json`。SFS 僅畫實測 k 的點，缺失與無效結果不插值。", "",
              "本次驗證詳見 `phase3b_verification.json`，與 Phase 3A 的 `verification.json` 分開。Phase 1 既有 rv1/rv2 重複警告須保留，不能誤報為全項 PASS。", "",
              "完成驗收後可規劃 Phase 3C 的時間序列校準與區間覆蓋率評估；目前未製作 Prediction Interval。Holdout 已查看、時間相依性與模型選擇後評估偏差仍須處理，不能直接宣稱區間有效。尚未 Commit／Push，等待人工驗收。", ""]
    (ROOT / "reports/phase3/phase3b_report.md").write_text("\n".join(lines))
