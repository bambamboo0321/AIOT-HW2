"""Phase 3D：Training-only 六欄固定組合探索；不修改歷史成果或保存新模型。"""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

from .feature_selection import SelectionSpec, fit_preprocessor, fit_selected_regression
from .phase2_common import ROOT, configure_matplotlib
from .preprocessing_config import PreprocessingConfig
from .selection_runner import load_energy_training, regression_metrics
from .validate_data import sha256_file

OUTPUT = ROOT / "reports/phase3/phase3d"
METHODS = ("mutual_information", "f_test", "forward_sfs", "rfe")
K = 6
MAX_COMBINATIONS = 1000
TIE_TOLERANCE = 1e-9


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_json(name, payload):
    output_path(name).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n", encoding="utf-8")


def output_path(name):
    if Path(name).name != name or not name or name in {".", ".."}:
        raise ValueError("Phase 3D 輸出只能是固定目錄下的檔名。")
    if OUTPUT.is_symlink():
        raise ValueError("Phase 3D 輸出目錄不得為 symlink。")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    if path.is_symlink():
        raise ValueError("Phase 3D 輸出檔不得為 symlink。")
    return path


def check_protected():
    manifest = read("reports/phase3/phase3d/integrity_baseline.json")
    if (subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip() != manifest["branch"]
            or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip() != manifest["head"]
            or subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT)):
        raise ValueError("Branch／HEAD／空 staging 契約不符，停止。")
    changed = [name for name, digest in manifest["files"].items()
               if not (ROOT / name).is_file() or sha256_file(ROOT / name) != digest]
    if changed:
        raise ValueError(f"既有檔案完整性失敗：{changed}")
    return {"checked_files": len(manifest["files"]), "changed_files": changed,
            "staging_empty": True, "dataset_sha256": manifest["files"]["data/raw/KAG_energydata_complete.csv"]}


def load_conditions():
    experiment = read("reports/phase3/results/experiment_config.json")
    partition, config = load_energy_training()
    if asdict(config) != experiment["preprocessing"]:
        raise ValueError("無法一致還原 Phase 3B preprocessing。")
    if (len(config.feature_columns) != 25 or experiment["training_rows"] != len(partition.X)
            or config.random_state != experiment["random_state"]
            or config.cv_gap != experiment["gap"]
            or experiment["inner_splits"] != experiment["selector_splits"]
            or config.cv_splits != experiment["outer_splits"]
            or config.numerical_scaling != "standard" or config.missing_numeric != "median"
            or config.outlier_strategy != "keep" or config.split_strategy != "chronological"):
        raise ValueError("Phase 3B 固定比較條件不符；不得猜測或改用其他設定。")
    return partition, config, experiment


def sample_combinations(candidates, count, seed, k=K):
    names = list(candidates)
    if (len(set(names)) != len(names) or type(count) is not int or not 1 <= count <= MAX_COMBINATIONS
            or not 1 <= k <= len(names) or count > math.comb(len(names), k)):
        raise ValueError("樣本數超出 1,000 上限／組合空間，或候選與 k 無效。")
    rng = np.random.default_rng(seed)
    seen, sampled = set(), []
    while len(sampled) < count:
        indices = tuple(sorted(map(int, rng.choice(len(names), k, replace=False))))
        if indices not in seen:
            seen.add(indices)
            sampled.append(indices)
    return sampled


@dataclass
class PreparedCV:
    train: np.ndarray
    validation: np.ndarray
    training_y: np.ndarray
    validation_y: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    processor: object


def prepare_cv(partition, config, folds, gap):
    partition.validate(config)
    cache = []
    for train, valid in TimeSeriesSplit(n_splits=folds, gap=gap).split(partition.X):
        if set(partition.X.index[train]) & set(partition.forbidden_indices) or set(partition.X.index[valid]) & set(partition.forbidden_indices):
            raise ValueError("Holdout 不得進入 Training CV。")
        processor, training = fit_preprocessor(partition.X.iloc[train], partition.y.iloc[train], config)
        validation = np.asarray(processor.transform(partition.X.iloc[valid]), dtype=float)
        cache.append(PreparedCV(training, validation, partition.y.iloc[train].to_numpy(),
                                partition.y.iloc[valid].to_numpy(), train, valid, processor))
    return cache


def score_subset(cache, indices):
    indices = tuple(indices)
    if len(indices) != K or len(set(indices)) != K or any(type(i) is not int or not 0 <= i < cache[0].train.shape[1] for i in indices):
        raise ValueError("每個組合必須有六個合法且不重複的候選索引。")
    start = perf_counter()
    metrics = []
    for fold in cache:
        model = LinearRegression().fit(fold.train[:, indices], fold.training_y)
        metrics.append(regression_metrics(fold.validation_y, model.predict(fold.validation[:, indices])))
    return {"rmse_wh": float(np.mean([m["rmse"] for m in metrics])),
            "mae_wh": float(np.mean([m["mae"] for m in metrics])),
            "r2": float(np.mean([m["r2"] for m in metrics])),
            "fold_rmse_wh": [m["rmse"] for m in metrics],
            "fold_mae_wh": [m["mae"] for m in metrics],
            "fold_r2": [m["r2"] for m in metrics],
            "runtime_seconds": perf_counter() - start}


def selector_cv(partition, config, experiment, payload, cache):
    """B 類評估：每個 validation 的支援集合只能由自己的 training 產生。"""
    methods = []
    for method in METHODS:
        folds = []
        for number, fold in enumerate(cache, 1):
            train, valid = fold.train_indices, fold.validation_indices
            expected_train, expected_valid = [int(train[0]), int(train[-1]) + 1], [int(valid[0]), int(valid[-1]) + 1]
            if method != "forward_sfs":
                old = [r for r in payload["records"] if r["method"] == method and r["level"] == "final_selection_cv"
                       and r["requested_k"] == K and r["inner_fold"] == number]
                if (len(old) != 1 or old[0]["train_range"] != expected_train
                        or old[0]["validation_range"] != expected_valid or old[0]["status"] != "ok"):
                    raise ValueError("既有 k=6 紀錄不是相同 final_selection_cv。")
                row = old[0]
                indices = row["selection"]["selected_feature_indices"]
                metric = {key: row[key] for key in ("rmse", "mae", "r2")}
                # 只重算同一支援集合的 LR 分數，核對保存紀錄，不重新選擇它。
                check = score_subset([fold], indices)
                for key, value in metric.items():
                    np.testing.assert_allclose(value, check[key if key == "r2" else key + "_wh"], rtol=1e-12, atol=1e-10)
                source = "existing_phase3b_final_selection_cv_rechecked"
                seconds = row["runtime_seconds"]
            else:
                start = perf_counter()
                model = fit_selected_regression(partition.X.iloc[train], partition.y.iloc[train], config,
                    SelectionSpec(method, K, inner_splits=experiment["selector_splits"],
                                  gap=experiment["gap"], random_state=experiment["random_state"]))
                metric = regression_metrics(partition.y.iloc[valid], model.predict(partition.X.iloc[valid]))
                indices = model.selection.indices.tolist()
                seconds = perf_counter() - start
                source = "new_phase3d_fold_local_sfs_k6"
            folds.append({"fold": number, "features": [config.feature_columns[i] for i in indices],
                          "indices": indices, "train_range": expected_train, "validation_range": expected_valid,
                          **metric, "source": source, "runtime_seconds": seconds})
        methods.append({"method": method, "evaluation": "B_selector_inside_each_training_fold",
                        "k": K, "source": folds[0]["source"], "folds": folds,
                        **{key: float(np.mean([f[key] for f in folds])) for key in ("rmse", "mae", "r2")}})
    return methods


CSV_FIELDS = ("sample_id", "rank", "feature_indices", "features", "rmse_wh", "mae_wh", "r2",
              "fold_rmse_wh", "fold_mae_wh", "fold_r2", "runtime_seconds")


def write_csv(name, rows):
    with output_path(name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], list)
                             else row[key] for key in CSV_FIELDS})


def make_plots(summary, correlation):
    configure_matplotlib()
    import matplotlib.pyplot as plt
    sources = {}
    scores = summary["random_search"]["rmse_values_in_sample_order"]
    mi = summary["fixed_combinations"]["mutual_information"]["rmse_wh"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores, bins=30, color="#68a3bb", edgecolor="white")
    ax.axvline(mi, color="#bf4b45", label=f"Fixed MI reference: {mi:.3f} Wh")
    ax.set(xlabel="Mean Training CV RMSE (Wh)", ylabel="Number of random combinations",
           title="Phase 3D: 1,000 random fixed six-feature subsets (new, exploratory)")
    ax.legend()
    fig.tight_layout(); fig.savefig(output_path("rmse_distribution.png"), dpi=150); plt.close(fig)
    sources["rmse_distribution.png"] = {"rmse_wh": scores, "fixed_mi_rmse_wh": mi}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fixed = summary["fixed_combinations"]
    labels = ["Fixed MI", "Fixed F-test", "Fixed RFE", "Random best"]
    values = [fixed[m]["rmse_wh"] for m in ("mutual_information", "f_test", "rfe")] + [summary["random_search"]["best"]["rmse_wh"]]
    axes[0].bar(labels, values, color=["#bf4b45", "#68a3bb", "#68a3bb", "#d6a34a"])
    axes[0].set(title="A: fixed subsets (new, retrospective / selected)", ylabel="Mean Training CV RMSE (Wh)")
    procedures = summary["selector_cv"]
    axes[1].bar(["MI", "F-test", "SFS", "RFE"], [r["rmse"] for r in procedures],
                color=["#888888", "#888888", "#68a3bb", "#888888"])
    axes[1].set(title="B: fold-local selection (saved MI/F/RFE; new SFS)", ylabel="Mean Training CV RMSE (Wh)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Phase 3D: separate evaluation definitions; no Holdout scores")
    fig.tight_layout(); fig.savefig(output_path("method_comparison.png"), dpi=150); plt.close(fig)
    sources["method_comparison.png"] = {"fixed_labels": labels, "fixed_rmse_wh": values,
                                        "selector_methods": [r["method"] for r in procedures],
                                        "selector_rmse_wh": [r["rmse"] for r in procedures]}
    names = list(dict.fromkeys(fixed["mutual_information"]["features"] + summary["random_search"]["best"]["features"]))
    matrix = np.array([[correlation["matrix"][a][b] for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), names, rotation=55, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Training-only Pearson correlation: fixed MI and random-best union")
    fig.colorbar(im, ax=ax, label="Pearson r (dimensionless)")
    fig.tight_layout(); fig.savefig(output_path("feature_correlation.png"), dpi=150); plt.close(fig)
    sources["feature_correlation.png"] = {"features": names, "matrix": matrix.tolist(),
                                           "scope": "training only; descriptive after search"}
    write_json("plot_data.json", {"source": "comparison_summary.json / training_correlation.json",
                                 "figures": sources})


def run():
    start = perf_counter()
    check_protected()
    audit = read("reports/phase3/phase3d/audit_details.json")
    if audit["status"] != "PASS" or audit["major_problem_found"]:
        raise ValueError("先完成唯讀 Audit，重大問題未解除前不得搜尋。")
    partition, config, experiment = load_conditions()
    payload = read("reports/phase3/results/feature_selection_results.json")
    best = read("reports/phase3/results/best_features.json")
    cache = prepare_cv(partition, config, experiment["inner_splits"], experiment["gap"])
    conditions = {"source": "reports/phase3/results/experiment_config.json",
                  "comparison_level": "final_selection_cv", "cv_splits": experiment["inner_splits"],
                  "selector_splits": experiment["selector_splits"], "gap": experiment["gap"],
                  "random_seed": experiment["random_state"], "preprocessing": experiment["preprocessing"],
                  "training_rows": len(partition.X), "holdout_used": False,
                  "scoring": "arithmetic mean of fold RMSE; RMSE = sqrt(sklearn MSE), Wh",
                  "model": "sklearn.linear_model.LinearRegression", "k": K,
                  "sample_count": MAX_COMBINATIONS, "max_combinations": MAX_COMBINATIONS,
                  "sampling": "numpy default_rng.choice without replacement within subset; reject repeated subsets",
                  "folds": [{"fold": i, "train_range": [int(f.train_indices[0]), int(f.train_indices[-1])+1],
                             "validation_range": [int(f.validation_indices[0]), int(f.validation_indices[-1])+1]}
                            for i, f in enumerate(cache, 1)]}
    if conditions["folds"] != audit["folds"]:
        raise ValueError("CV 範圍與搜尋前 Audit 不一致。")
    write_json("experiment_config.json", conditions)
    procedures = selector_cv(partition, config, experiment, payload, cache)
    write_json("method_cv_results.json", {"evaluation": "B_selector_inside_each_training_fold", "methods": procedures})
    print("Four k=6 selection procedures verified; sampling 1,000 unique fixed subsets.", flush=True)
    combinations = sample_combinations(config.feature_columns, MAX_COMBINATIONS, experiment["random_state"])
    score_cache, rows = {}, []
    search_start = perf_counter()
    for sample, indices in enumerate(combinations, 1):
        score = score_subset(cache, indices)
        score_cache[indices] = score
        rows.append({"sample_id": sample, "feature_indices": list(indices),
                     "features": [config.feature_columns[i] for i in indices], **score})
        if sample == 20 or sample % 100 == 0:
            elapsed = perf_counter() - search_start
            print(f"Random subsets {sample}/1000; elapsed={elapsed:.2f}s; remaining estimate={elapsed/sample*(1000-sample):.2f}s", flush=True)
    search_seconds = perf_counter() - search_start
    fixed = {}
    for method in ("mutual_information", "f_test", "rfe"):
        refit = next(r for r in best["full_training_refits"] if r["method"] == method)
        indices = tuple(refit["selection"]["selected_feature_indices"])
        if len(indices) != K:
            raise ValueError("既有完整 Training 組合不是六欄，不能猜測。")
        score = score_cache[indices] if indices in score_cache else score_subset(cache, indices)
        fixed[method] = {"features": [config.feature_columns[i] for i in indices],
                         "feature_indices": list(indices), **score,
                         "evaluation": "A_fixed_subset_retrospective_cv",
                         "selection_source": "existing full-training selection; not unbiased selector assessment"}
    ranked = sorted(rows, key=lambda row: (row["rmse_wh"], row["feature_indices"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    write_csv("random_search_results.csv", ranked)
    write_csv("top10_combinations.csv", ranked[:10])
    write_csv("worst10_combinations.csv", list(reversed(ranked[-10:])))
    scores = np.array([r["rmse_wh"] for r in rows])
    mi = fixed["mutual_information"]
    better = int(np.count_nonzero(scores < mi["rmse_wh"] - TIE_TOLERANCE))
    ties = int(np.count_nonzero(np.abs(scores - mi["rmse_wh"]) <= TIE_TOLERANCE))
    mi_in_sample = tuple(mi["feature_indices"]) in score_cache
    corr = partition.X[config.feature_columns].corr(method="pearson")
    threshold = read("reports/phase2/eda_statistics.json")["high_correlation_threshold_absolute_r"]
    pairs = [{"a": a, "b": b, "r": float(corr.loc[a,b])}
             for i, a in enumerate(mi["features"]) for b in mi["features"][i+1:]
             if abs(corr.loc[a,b]) >= threshold]
    correlation = {"scope": "full Training descriptive only; no Holdout", "method": "Pearson",
                   "high_correlation_threshold_absolute_r": threshold, "mi_high_correlation_pairs": pairs,
                   "matrix": json.loads(corr.to_json(double_precision=15))}
    write_json("training_correlation.json", correlation)
    total = math.comb(len(config.feature_columns), K)
    summary = {"phase": "3D", "status": "complete", "units": {"rmse_wh": "Wh", "mae_wh": "Wh", "r2": "dimensionless", "runtime_seconds": "s"},
               "conditions": conditions, "selector_cv": procedures, "fixed_combinations": fixed,
               "fixed_sfs_k6": {"status": "not_available", "reason": "Phase 3B full-training SFS was k=10; no full-training SFS k=6 is invented; use B fold-local k=6 instead"},
               "random_search": {"sample_count": len(rows), "unique_count": len(score_cache),
                   "total_possible_combinations": total, "space_fraction": len(rows)/total,
                   "rmse_values_in_sample_order": scores.tolist(),
                   "distribution": {"min": float(scores.min()), "mean": float(scores.mean()), "std_ddof0": float(scores.std()),
                                    **{label: float(np.quantile(scores,q)) for label,q in (("p05",.05),("p25",.25),("median",.5),("p75",.75),("p95",.95))},
                                    "max": float(scores.max())},
                   "best": ranked[0], "top10": ranked[:10], "worst10": list(reversed(ranked[-10:])),
                   "mi_reference": {"in_random_sample": mi_in_sample, "random_better_than_mi": better,
                       "random_tied_with_mi": ties, "tie_tolerance_wh": TIE_TOLERANCE,
                       "competition_rank_among_random_and_mi": better+1,
                       "rank_population": len(rows)+(0 if mi_in_sample else 1),
                       "extra_reference_not_counted_in_1000": not mi_in_sample,
                       "common_features_with_best": [n for n in mi["features"] if n in ranked[0]["features"]]},
                   "elapsed_seconds": search_seconds},
               "correlation": {k:v for k,v in correlation.items() if k!="matrix"},
               "exhaustive_search": {"executed": False, "recommended_now": False,
                    "estimated_search_seconds_same_cached_cv": total*search_seconds/len(rows),
                    "estimate_limit": "linear projection of measured 1000-subset runtime; excludes startup and selector costs; not a benchmark guarantee"},
               "selection_bias": "A fixed method subsets reused full Training targets; random minimum reuses 1000 CV assessments. Neither is unbiased generalization error; do not rank them against B as equivalent estimators.",
               "sources": {name:sha256_file(ROOT/name) for name in (
                   "reports/phase3/results/experiment_config.json", "reports/phase3/results/best_features.json",
                   "reports/phase3/results/feature_selection_results.json", "reports/phase3/results/feature_selection_results.csv",
                   "src/phase3d_search.py")},
               "total_seconds_before_plots": perf_counter()-start}
    write_json("comparison_summary.json", summary)
    make_plots(summary, correlation)
    write_summary(summary)
    integrity = check_protected()
    integrity["outputs"] = {p.name:sha256_file(p) for p in OUTPUT.iterdir() if p.is_file() and p.name!="integrity_verification.json"}
    write_json("integrity_verification.json", integrity)
    print(json.dumps({"status":"complete", "best":ranked[0], "mi_fixed_rmse_wh":mi["rmse_wh"],
                      "mi_reference":summary["random_search"]["mi_reference"], "protected":integrity["checked_files"]}, ensure_ascii=False), flush=True)


def write_summary(s):
    random, fixed = s["random_search"], s["fixed_combinations"]
    mi, best = fixed["mutual_information"], random["best"]
    reference = random["mi_reference"]
    lines = ["# Phase 3D：六特徵組合探索與驗證", "",
             "本階段只用 Training；沿用 Phase 3B final_selection_cv：3 folds、gap=6、seed=42、median／StandardScaler、LinearRegression。RMSE 是各 fold RMSE 算術平均，單位 Wh。", "",
             "## B：包含選特徵流程的 CV（各 fold 內重選）", "",
             "| 方法 k=6 | Mean CV RMSE Wh | Mean CV MAE Wh | Mean CV R² | 來源 |", "| --- | ---: | ---: | ---: | --- |"]
    lines += [f"| {r['method']} | {r['rmse']:.6f} | {r['mae']:.6f} | {r['r2']:.6f} | {r['source']} |" for r in s["selector_cv"]]
    lines += ["", "MI／F-test／RFE 讀取原有同範圍 k=6 結果，另用該 fold 保存支援集合重算 LR，確認分數一致。SFS k=6 不在舊 grid，新增三個 training fold 內選擇及內層時間 CV；沒有先用完整 Training 選 SFS 再宣稱無偏。四方法的 validation 僅用於各 fold 最終評分；事後在方法間擇優仍有偏差。", "",
              "## A：固定組合的回顧性 CV（與 B 分開）", "",
              "| 固定六欄參考 | Mean CV RMSE Wh |", "| --- | ---: |"]
    lines += [f"| {m} | {r['rmse_wh']:.6f} |" for m,r in fixed.items()]
    lines += ["", "上述固定方法組合來自完整 Training 的既有支援集合，其產生曾看到回顧性 CV validation 的 Target；分數有資料重用／選取偏差，不是無偏的 selector 評估。Random 固定組合使用同一 CV，但挑最低者亦有搜尋偏差。原 SFS 沒有固定 k=6 全 Training 組合，因此 A 不捏造 SFS 結果，SFS 比較使用 B。", "",
              "## 1,000 組 Random Combination Search", "",
              f"- 不重複六欄組合 {random['unique_count']} 組；共 {random['total_possible_combinations']:,} 種可能，抽樣比例 {random['space_fraction']:.4%}。",
              f"- 僅隨機搜尋耗時 {random['elapsed_seconds']:.3f} 秒；含準備與方法比較、未含繪圖 {s['total_seconds_before_plots']:.3f} 秒。",
              f"- RMSE 分布：min {random['distribution']['min']:.6f}、median {random['distribution']['median']:.6f}、mean {random['distribution']['mean']:.6f}、max {random['distribution']['max']:.6f} Wh；SD(ddof=0) {random['distribution']['std_ddof0']:.6f} Wh。完整分位數見 JSON。",
              f"- 固定 MI RMSE {mi['rmse_wh']:.6f} Wh；優於 MI {reference['random_better_than_mi']} 組，與 MI 差距 ≤1e-9 Wh 的 {reference['random_tied_with_mi']} 組。MI competition rank={reference['competition_rank_among_random_and_mi']}／{reference['rank_population']}。",
              f"- MI 是否原就在 1,000 組內：{reference['in_random_sample']}；若不在，額外參考評估不計入 1,000 組。",
              f"- 最佳抽樣組合：{', '.join(best['features'])}；RMSE {best['rmse_wh']:.6f} Wh。",
              f"- 與 MI 共同 {len(reference['common_features_with_best'])} 欄：{', '.join(reference['common_features_with_best']) or '無'}。",
              "", "## 相關性、可信度與結論", "",
              f"Training-only Pearson 高相關門檻沿用 Phase 2 |r|≥{s['correlation']['high_correlation_threshold_absolute_r']}；MI 六欄內符合門檻的配對如下：", ""]
    lines += [f"- {p['a']}／{p['b']}：r={p['r']:.6f}。" for p in s["correlation"]["mi_high_correlation_pairs"]] or ["- 無符合門檻配對。"]
    lines += ["", "MI 單變量關聯高不代表特徵彼此不冗餘，也不等於對線性模型最佳。本結果能指出探索性差異，不能證明 MI 普遍不適合此資料，更不能證明 Random 最佳組合具有更好的真正泛化。", "",
              "原正式流程的唯讀查核及既有測試未發現直接 Holdout／fold 洩漏或指標計算錯誤；A 類固定方法組合存在明確的回顧性資料重用，已與 fold-local B 類分開。詳見 verification_report.md。未讀入 Holdout Target 作搜尋、未重新產生 Holdout 評估，未保存／取代任何模型。", "",
              f"目前不建議 Exhaustive Search：共有 {random['total_possible_combinations']:,} 組，以本次快取 CV 搜尋時間線性外推約 {s['exhaustive_search']['estimated_search_seconds_same_cached_cv']:.1f} 秒（僅成本估計，非保證），但增加搜尋會擴大選取偏差，不能解決時間相依及泛化證據不足。先釐清評估目標／預先指定探索範圍較有價值；若將來另獲授權，需用未參與搜尋的外層時間驗證評估整個搜尋程序。此次沒有執行 Exhaustive Search。", "",
              "## 輸出與重現", "",
              "`python -B -m src.phase3d_search --run` 僅重跑 Phase 3D，會更新本階段目錄；不重跑 Phase 3B。首次執行前必須已有通過的 audit_details.json、verification_report.md 與起始 integrity_baseline.json。執行上限固定 1,000 組。耗時欄位具有執行環境差異；固定 seed、支援集合與分數可重現。", "",
              "全部樣本／Top 10／Worst 10 CSV 採 UTF-8、LF。CSV 保存 fold 指標與 feature indices；comparison_summary.json 保存分布、排名、參考組合及來源 digest；plot_data.json 保存三張圖的實際來源數據。完整性與測試結果另見 integrity_verification.json、test_verification.json。", ""]
    output_path("phase3d_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Audit 通過後最多抽樣 1,000 組")
    args = parser.parse_args()
    if not args.run:
        parser.error("需要 --run；無 Exhaustive Search 選項。")
    with threadpool_limits(limits=1):
        run()


if __name__ == "__main__":
    main()
