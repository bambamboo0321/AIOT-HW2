"""有限時間、subprocess 隔離的 Phase 3A runtime benchmark；不排名預測分數。"""

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from time import monotonic
import warnings

from .feature_selection import METHODS
from .phase2_common import ROOT
from .selection_runner import PHASE3_DIR, load_energy_training, run_nested_cv, training_digest


@dataclass(frozen=True)
class BenchmarkCase:
    method: str
    k: int | None
    samples: int = 480
    outer_splits: int = 2
    inner_splits: int = 2
    gap: int = 6
    pilot: bool = False


def environment():
    return {"python": platform.python_version(), "platform": platform.system(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(),
            **{package: importlib.metadata.version(package)
               for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")},
            "thread_limit": 1, "random_state": 42}


def run_bounded(command, timeout_seconds, *, env=None):
    """到時直接終止獨立 process group，包含可能的子程序；不是事後判斷。"""
    if not 0 < timeout_seconds <= 120:
        raise ValueError("單一 Case timeout 必須 >0 且 <=120 秒。")
    started = monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, cwd=ROOT, env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return {"status": "ok" if process.returncode == 0 else "error", "returncode": process.returncode,
                "wall_seconds": monotonic() - started, "stdout": stdout, "stderr": stderr,
                "timeout": False, "child_reaped": process.poll() is not None}
    except subprocess.TimeoutExpired:
        # macOS/POSIX repository environment；不建立額外 worker pool。
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return {"status": "timeout", "returncode": process.returncode,
                "wall_seconds": monotonic() - started, "stdout": stdout, "stderr": stderr,
                "timeout": True, "child_reaped": process.poll() is not None}


def benchmark_case(case):
    from threadpoolctl import threadpool_limits
    partition, config = load_energy_training()
    partition = partition.prefix(case.samples)
    started = monotonic()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        result = run_nested_cv(partition, config, [case.method], () if case.k is None else (case.k,),
                               outer_splits=case.outer_splits, inner_splits=case.inner_splits,
                               gap=case.gap)
    method = result["methods"][case.method]
    elapsed = monotonic() - started
    folds = method["folds"]
    return {"case": asdict(case), "status": "ok", "timeout": False,
            "samples": len(partition.X), "features": len(config.feature_columns),
            "runtime_seconds": elapsed, "training_index_sha256": training_digest(partition),
            "last_training_row_index": int(partition.X.index[-1]),
            "holdout_used": False, "fit_accounting": method["fit_accounting"],
            "outer_cv_folds": case.outer_splits,
            "selector_cv_folds": case.inner_splits if case.method in {"forward_sfs", "lasso_cv", "rfecv"} else 0,
            "outer_training_rows": [f["train_rows"] for f in result["outer_fold_manifest"]],
            "timing": {key: sum(f["timing"][key] for f in folds) for key in folds[0]["timing"]},
            "selection_by_outer_fold": [f["selection"] for f in folds],
            "scored_outer_folds": method["scored_outer_folds"],
            "quality_status": "complete" if not caught and method["zero_feature_folds"] == 0 else "warnings_or_zero_features",
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
            "predictive_scores_published": False}


def default_cases():
    # 先跑一次 SFS pilot；若超時，不再嘗試其他 SFS cases。
    cases = [BenchmarkCase("forward_sfs", 3, samples=240, pilot=True)]
    cases += [BenchmarkCase(method, k) for method in ("f_test", "mutual_information", "rfe") for k in (3, 5, 10)]
    cases += [BenchmarkCase("forward_sfs", 3), BenchmarkCase("forward_sfs", 5),
              BenchmarkCase("lasso_cv", None), BenchmarkCase("rfecv", None)]
    return cases


def run_benchmark(cases=None, *, case_timeout=120., total_timeout=600.):
    if not 0 < case_timeout <= 120 or not 0 < total_timeout <= 600:
        raise ValueError("Phase 3A 上限：每 Case 120 秒，整體 600 秒。")
    cases = default_cases() if cases is None else list(cases)
    if len(cases) > 14:
        raise ValueError("Phase 3A 不得擴大超過預定 14 個小型 cases。")
    if any(c.samples > 1024 or c.outer_splits != 2 or c.inner_splits != 2 for c in cases):
        raise ValueError("Benchmark 只允許 <=1024 個 Training prefix rows、2 Outer／2 Inner folds。")
    started = monotonic()
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LOKY_MAX_CPU_COUNT"] = "1"
    records, timed_out_methods = [], set()
    for case in cases:
        remaining = total_timeout - (monotonic() - started)
        if remaining <= 0:
            records.append({"case": asdict(case), "status": "not_run_total_budget", "timeout": False})
            continue
        if case.method in timed_out_methods:
            records.append({"case": asdict(case), "status": "not_run_after_method_timeout", "timeout": False})
            continue
        command = [sys.executable, "-B", "-m", "src.selection_benchmark", "--worker",
                   "--method", case.method, "--samples", str(case.samples), "--gap", str(case.gap)]
        if case.k is not None:
            command += ["--k", str(case.k)]
        bounded = run_bounded(command, min(case_timeout, remaining), env=env)
        if bounded["status"] == "ok":
            try:
                record = json.loads(bounded["stdout"])
                record["case"] = asdict(case)
            except (ValueError, TypeError):
                record = {"case": asdict(case), "status": "invalid_worker_output", "timeout": False}
        else:
            record = {"case": asdict(case), "status": bounded["status"], "timeout": bounded["timeout"],
                      "returncode": bounded["returncode"], "child_reaped": bounded["child_reaped"]}
            if bounded["status"] == "error":
                # 不把含私人 Python 路徑的 traceback 寫入可提交報告。
                record["error_summary"] = bounded["stderr"].splitlines()[-1].replace(str(ROOT), "<repository>") if bounded["stderr"] else "worker failed"
            if bounded["timeout"]:
                timed_out_methods.add(case.method)
        record["wall_seconds_including_startup"] = bounded["wall_seconds"]
        records.append(record)
        print(f"{case.method} k={case.k} n={case.samples}: {record['status']} ({bounded['wall_seconds']:.3f}s)", flush=True)
    return {"purpose": "runtime_only_not_predictive_model_ranking", "environment": environment(),
            "limits": {"case_seconds": case_timeout, "total_seconds": total_timeout, "maximum_samples": 1024},
            "elapsed_seconds": monotonic() - started, "holdout_used": False,
            "sampling": "chronological prefix of Phase 2 Training only; no random resplit",
            "results": records}


def estimate_costs(benchmark):
    """明列 fit/path 計數；小樣本時間只作線性 n 外推，不是實測完整實驗。"""
    p, outer, inner, n = 25, 5, 3, 15788
    from sklearn.model_selection import TimeSeriesSplit
    formal_folds = list(TimeSeriesSplit(n_splits=outer, gap=6).split(range(n)))
    mean_training_rows = np_mean([len(train) for train, _ in formal_folds])
    mean_validation_rows = np_mean([len(valid) for _, valid in formal_folds])
    # 比較所有固定 k 的獨立曲線；k=p 直接全選，不執行搜尋。
    counts = {
        "f_test": {"selector_units": outer * (p - 1), "downstream_fits": outer * p},
        "mutual_information": {"selector_units": outer * (p - 1), "downstream_fits": outer * p},
        "forward_sfs": {"selector_units": outer * inner * sum(sum(range(p-k+1, p+1)) for k in range(1, p)),
                        "downstream_fits": outer * p},
        "rfe": {"selector_units": outer * sum(p-k+1 for k in range(1, p)), "downstream_fits": outer * p},
        "lasso_cv": {"selector_units": outer, "downstream_fits": outer, "cv_paths": outer * inner,
                     "alpha_path_points": outer * inner * 6, "discarded_stack_refits": outer, "support_refits": outer},
        "rfecv": {"selector_units_range": [outer * (inner*p+2), outer * (inner*p+p+1)],
                  "downstream_fits": outer},
    }
    output = {"status": "projection_not_full_experiment", "n_training": n, "p": p, "outer": outer, "inner": inner,
              "mean_formal_outer_training_rows": mean_training_rows,
              "mean_formal_outer_validation_rows": mean_validation_rows,
              "assumptions": ["各 k 獨立執行，未假設跨 k 快取。k=p 無搜尋。",
                              "依已測的 selector unit／path 與樣本數作線性 n 外推；未實測完整資料。",
                              "未建模 cache、BLAS、MI 的 n log n、Lasso convergence 或不同子集寬度影響。",
                              "0.5x..4x 只為敏感度示例，不是統計信賴區間。"], "methods": {}}
    for method, workload in counts.items():
        cases = [r for r in benchmark["results"] if r["case"]["method"] == method and r["status"] == "ok" and not r["case"]["pilot"]]
        if not cases:
            output["methods"][method] = {"workload": workload, "estimated_seconds": None, "reason": "no successful calibration"}
            continue
        selector_sum, downstream_sum, prep_sum, evaluation_sum = 0., 0., 0., 0.
        for r in cases:
            n_ratio = mean_training_rows / float(np_mean(r["outer_training_rows"]))
            n_valid_ratio = mean_validation_rows / (r["samples"] // (r["outer_cv_folds"] + 1))
            account = r["fit_accounting"]
            if method in {"f_test", "mutual_information"}:
                units = account.get("filter_score_calls", 0)
            elif method in {"forward_sfs", "rfe", "rfecv"}:
                units = account.get("selector_linear_regression_fits", 0)
            else:
                units = account.get("lasso_cv_fits", 0)
            formal_units = workload.get("selector_units", sum(workload.get("selector_units_range", [])) / 2)
            # LassoCV calibration 使用 2 paths；正式設計為 3，保守依 paths 比率外推。
            inner_ratio = inner / r["selector_cv_folds"] if method == "lasso_cv" else 1.
            selector_sum += r["timing"]["selector_seconds"] / max(1, units) * formal_units * n_ratio * inner_ratio
            downstream_sum += r["timing"]["predictor_fit_seconds"] / max(1, account.get("downstream_linear_regression_fits", 0)) * workload["downstream_fits"] * n_ratio
            prep_sum += r["timing"]["outer_preprocessing_seconds"] / r["outer_cv_folds"] * workload["downstream_fits"] * n_ratio
            evaluation_sum += r["timing"]["prediction_and_metrics_seconds"] / max(1, r["scored_outer_folds"]) * workload["downstream_fits"] * n_valid_ratio
        selector, downstream, prep, evaluation = (value / len(cases) for value in (selector_sum, downstream_sum, prep_sum, evaluation_sum))
        nominal = selector + downstream + prep + evaluation
        output["methods"][method] = {"workload": workload, "selector_seconds": selector,
                                      "downstream_fit_seconds": downstream, "outer_preprocessing_seconds": prep,
                                      "downstream_evaluation_seconds": evaluation,
                                      "estimated_seconds": nominal, "sensitivity_seconds": [nominal*.5, nominal*4],
                                      "calibration_warning_count": sum(len(r["warnings"]) for r in cases),
                                      "calibration_zero_feature_folds": sum(sum(s["selected_k"] == 0 for s in r["selection_by_outer_fold"]) for r in cases),
                                      "calibration_cases": len(cases)}
    output["nested_k_tuning_workload"] = {
        "filter_selector_calls_upper_bound_per_method": outer * (inner*(p-1)+1),
        "rfe_selector_lr_fits_upper_bound": outer * inner * sum(p-k+1 for k in range(1,p)) + outer*p,
        "sfs_selector_lr_fits_upper_bound": outer * inner * inner * sum(sum(range(p-k+1,p+1)) for k in range(1,p)) + outer * inner * sum(range(2,p+1)),
        "fixed_method_downstream_fits": outer * (inner*p+1),
        "explanation": "正式 inner k tuning 每個候選都重新 fit 完整流程；SFS 因自身 CV 再增加一層。成本不可等同各 k 描述曲線。"}
    return output


def np_mean(values):
    return sum(values) / len(values)


def write_outputs(result):
    folder = PHASE3_DIR / "benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    costs = estimate_costs(result)
    for row in result["results"]:
        if row["status"] == "ok":
            row["quality_status"] = "complete" if not row["warnings"] and all(s["selected_k"] > 0 for s in row["selection_by_outer_fold"]) else "warnings_or_zero_features"
    for name, payload in (("benchmark_results.json", result), ("cost_estimates.json", costs)):
        (folder / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    rows = []
    for r in result["results"]:
        case = r["case"]
        core_time = f"{r['runtime_seconds']:.4f}" if "runtime_seconds" in r else "—"
        wall_time = f"{r['wall_seconds_including_startup']:.4f}" if "wall_seconds_including_startup" in r else "—"
        rows.append(f"| {case['method']} | {case['k'] if case['k'] is not None else 'auto'} | {case['samples']} | {r['status']} | {core_time} | {wall_time} |")
    cost_rows = [f"| {m} | {v['estimated_seconds']:.2f} | {v.get('selector_seconds', 0):.2f} | {v.get('downstream_fit_seconds', 0):.2f} | {v.get('downstream_evaluation_seconds', 0):.2f} |" if v['estimated_seconds'] is not None else f"| {m} | 無法估算 | — | — | — |" for m, v in costs['methods'].items()]
    lasso = next((row for row in result["results"] if row["case"]["method"] == "lasso_cv" and row["status"] == "ok"), None)
    lasso_note = (f"LassoCV 的小樣本 Outer folds 實際選出 {[s['selected_k'] for s in lasso['selection_by_outer_fold']]} 個特徵；"
                  f"記錄了 {len(lasso['warnings'])} 個警告。" if lasso else "LassoCV 缺少成功的計時案例，不外推其收斂成本。")
    summary = f"""# Phase 3A：有限時間 Benchmark

僅計算成本，不比較模型預測分數；Holdout 未參與。資料為 Phase 2 Training 的前 240／480 列，保持原始時間順序與全部 25 個 Features。random_state=42，單執行緒。每 Case 上限 120 秒，整體上限 600 秒，subprocess 到時直接終止並回收。

每 Case 使用相同的 2 個 Outer folds；SFS／RFECV／LassoCV 內部使用 2 個時間 folds（gap=6）。固定 k 只驗證指定 k，不在此次 benchmark 尋找最佳 k。

| Method | k | Samples | Status | 核心實測秒 | 含啟動實測秒 |
| --- | ---: | ---: | --- | ---: | ---: |
{chr(10).join(rows)}

總實測牆鐘時間：{result['elapsed_seconds']:.3f} 秒。完整環境、警告、選中特徵 mapping、fit/path 計數與分項計時見 benchmark_results.json。零特徵 fold 不 fit Predictor；不補成 k=1。

## 完整設計成本外推

以下是基於小樣本的計算投影，**不是完整實驗實測**。固定方法為 k=1..25 各 k 獨立曲線、5 Outer／3 Selector CV folds；自動方法只跑一次 5 Outer／3 Inner nested 流程。表中總秒數另含外層前處理，不含 I/O、process 啟動、最終完整 Training 重訓或內層 k 調參。Downstream fit 包含訓練資料 transform 與 LR.fit，evaluation 包含 validation transform／predict／metrics。

| Method | 名目總秒數 | Selector 秒數 | Downstream LR fit 秒數 | Evaluation 秒數 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(cost_rows)}

計數／假設／敏感度區間見 cost_estimates.json。SFS 曲線需要 78,000 次候選 LR fits；若把 k 選擇也完整巢狀化，候選 fits 上界增至 238,860（另加 Downstream fits），不可只計算 125 次最終回歸。小資料下 Python 前處理開銷占比高，不能將此線性外推當成實際完工時間。

建議 Phase 3B 人工決定是否採較疏的 SFS k=[1,3,5,8,10,13,16,20,25]，或核准計算預算再執行全曲線。本階段不改變正式 k=1..25 設計，也不執行正式曲線。

{lasso_note} 零特徵 fold 跳過 Predictor，沒有強迫補成 k=1。警告均保留原訊息供核對，沒有因此擴大重試、挑選方法或調整 alpha grid。若存在收斂警告，其時間外推只是目前迭代上限下的成本，不能視為收斂解的成本保證。
"""
    (folder / "benchmark_summary.md").write_text(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--k", type=int)
    parser.add_argument("--samples", type=int, default=480)
    parser.add_argument("--gap", type=int, default=6)
    args = parser.parse_args()
    if args.worker:
        if args.method is None or not 12 <= args.samples <= 1024:
            parser.error("worker 需要 method 與 12..1024 samples")
        print(json.dumps(benchmark_case(BenchmarkCase(args.method, args.k, args.samples, gap=args.gap)), ensure_ascii=False, allow_nan=False))
    else:
        result = run_benchmark()
        write_outputs(result)


if __name__ == "__main__":
    main()
