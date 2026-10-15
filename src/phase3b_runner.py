"""Bounded、可續跑的 Training-only Phase 3B runner。

執行：python -B -m src.phase3b_runner --pilot；通過 gate 後 --run。
每個 subprocess 僅處理一個 method／training partition，不讀 Holdout Target。
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import warnings

import joblib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

from .feature_selection import METHODS, SelectionSpec, count_fits, fit_selected_regression
from .phase2_common import ROOT, EXPECTED_SHA256
from .phase3b_selection import FULL_GRID, SFS_GRID, fit_group
from .selection_benchmark import environment, run_bounded
from .selection_runner import load_energy_training, regression_metrics

OUTPUT = ROOT / "reports/phase3/results"
CACHE = ROOT / ".venv/phase3b_checkpoints"
CONFIG = {
    "phase": "3B", "schema_version": 1,
    "dataset_sha256": EXPECTED_SHA256, "training_rows": 15788,
    "holdout_rows": 3947, "holdout_target_read": False,
    "outer_splits": 5, "inner_splits": 3, "selector_splits": 3,
    "gap": 6, "random_state": 42, "thread_limit": 1,
    "grids": {m: list(SFS_GRID if m == "forward_sfs" else FULL_GRID)
              if m not in {"lasso_cv", "rfecv"} else [None] for m in METHODS},
    "lasso": {"alphas": [100., 10., 1., .1, .01, .001], "max_iter": 100000,
              "tol": 1e-5, "coefficient_tolerance": 1e-8,
              "reason": "Phase 3A had 7 ConvergenceWarnings at max_iter=10000; increase iteration budget before formal scores, unchanged alpha grid/tol; Training only"},
    "selection_rule": "minimum mean inner-CV RMSE; absolute tie tolerance 1e-9; fewer features, lower measured fit time, declared method order",
    "invalid_policy": "selected_k=0 => invalid_for_LR, null metrics; any failed fold makes candidate ineligible; retain all records",
    "case_timeout_seconds": 120, "sfs_gate_seconds": 3600,
    "pilot_projection": "24 full-grid training partitions at measured full-Training cost, multiplied by 2 safety factor; includes curves, nested tuning, final tuning and refit",
    "checkpoint_policy": "content-addressed by code/config/data/ranges; skip every terminal case including timeouts; no automatic retry",
    "path_reuse": "only identical training partition; deterministic filter scores/RFE elimination/SFS forward prefixes; sklearn algorithms unchanged",
}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint():
    sources = ["src/phase3b_runner.py", "src/phase3b_selection.py", "src/feature_selection.py",
               "src/preprocessing.py", "src/preprocessing_config.py", "src/selection_runner.py"]
    payload = {"config": CONFIG, "environment": environment(), "sources": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def thread_environment():
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        env[key] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def compact_selection(metadata):
    return {key: value for key, value in metadata.items()
            if key not in {"fit_indices", "inner_cv_manifest", "available_feature_names"}}


def execute_case(case):
    partition, config = load_energy_training()
    train = np.arange(*case["train"])
    valid = np.arange(*case["validation"]) if case.get("validation") else None
    if len(train) < 40 or train[-1] >= len(partition.X):
        raise ValueError("training range outside allowed Training partition")
    if valid is not None and (valid[-1] >= len(partition.X) or train[-1] + 6 >= valid[0]):
        raise ValueError("validation range overlaps training/gap or enters holdout")
    X, y = partition.X.iloc[train], partition.y.iloc[train]
    method = case["method"]
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1), count_fits() as counts:
        warnings.simplefilter("always")
        if case.get("final_refit"):
            k = case["grid"][0]
            model = fit_selected_regression(X, y, config, SelectionSpec(
                method, k, max_iter=CONFIG["lasso"]["max_iter"], tolerance=CONFIG["lasso"]["tol"]))
            models = {k: model}
            timing = {"shared_fit_seconds": perf_counter() - started - model.predictor_fit_seconds,
                      "downstream_seconds": model.predictor_fit_seconds, "path_reuse": False}
        else:
            models, timing = fit_group(X, y, config, method, case["grid"],
                                       max_iter=CONFIG["lasso"]["max_iter"], tolerance=CONFIG["lasso"]["tol"])
        records = []
        for k, model in models.items():
            selection = compact_selection(model.selection.metadata)
            record = {"method": method, "requested_k": k,
                      "selected_k": selection["selected_k"], "selection": selection,
                      "status": "ok" if model.predictor is not None else "invalid_for_LR",
                      "rmse": None, "mae": None, "r2": None,
                      "runtime_seconds": timing["shared_fit_seconds"] / len(models) + model.predictor_fit_seconds}
            # pilot deliberately does not publish any scores/support: runtime only.
            if not case.get("pilot") and model.predictor is not None:
                evaluation = train if valid is None else valid
                record.update(regression_metrics(partition.y.iloc[evaluation], model.predict(partition.X.iloc[evaluation])))
            if not case.get("pilot"):
                records.append(record)
            if case.get("save_model") and model.predictor is not None:
                model_dir = OUTPUT / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                temporary_model = model_dir / "final_linear_regression.joblib.tmp"
                joblib.dump(model, temporary_model, compress=3)
                temporary_model.replace(model_dir / "final_linear_regression.joblib")
                loaded = joblib.load(model_dir / "final_linear_regression.joblib")
                np.testing.assert_array_equal(model.predict(X.iloc[:20]), loaded.predict(X.iloc[:20]))
    return {"status": "ok", "case": case, "runtime_seconds": perf_counter() - started,
            "timing": timing, "fit_counts": counts, "records": records,
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught],
            "holdout_target_read": False, "training_rows": len(train),
            "train_range": case["train"], "validation_range": case.get("validation")}


def checkpoint_case(case, *, cache=CACHE, timeout=120.):
    token = fingerprint()
    digest = hashlib.sha256(json.dumps({"case": case, "fingerprint": token}, sort_keys=True).encode()).hexdigest()
    path = Path(cache) / f"{digest}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["fingerprint"] != token or result["case"] != case:
            raise ValueError("checkpoint fingerprint mismatch")
        if case.get("save_model") and result["status"] == "ok":
            model_path = OUTPUT / "models/final_linear_regression.joblib"
            if not model_path.is_file() or hashlib.sha256(model_path.read_bytes()).hexdigest() != result.get("model_sha256"):
                raise ValueError("saved final model missing or changed; checkpoint cannot be reused")
        return result, True
    command = [sys.executable, "-B", "-m", "src.phase3b_runner", "--worker", json.dumps(case)]
    bounded = run_bounded(command, timeout, env=thread_environment())
    if bounded["status"] == "ok":
        result = json.loads(bounded["stdout"])
    else:
        result = {"status": bounded["status"], "case": case, "records": [],
                  "timeout": bounded["timeout"], "child_reaped": bounded["child_reaped"],
                  "error_summary": bounded["stderr"].splitlines()[-1].replace(str(ROOT), "<repository>") if bounded["stderr"] else None}
    result.update({"fingerprint": token, "wall_seconds": bounded["wall_seconds"],
                   "checkpoint_id": digest})
    if case.get("save_model") and result["status"] == "ok":
        result["model_sha256"] = hashlib.sha256((OUTPUT / "models/final_linear_regression.joblib").read_bytes()).hexdigest()
    atomic_json(path, result)
    return result, False


def run_pilot():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT / "experiment_config.json", {**CONFIG, "preprocessing": asdict(load_energy_training()[1]),
                                                   "environment": environment(), "fingerprint": fingerprint()})
    case = {"method": "forward_sfs", "grid": list(SFS_GRID), "train": [0, 15788],
            "validation": None, "pilot": True}
    result, resumed = checkpoint_case(case)
    # 5 outer curve groups + 5*3 nested inner groups + 3 final-choice CV groups + final refit.
    groups = {"outer_curves_and_selected_outer_evaluation": 5,
              "nested_inner_k_selection": 15, "full_training_inner_selection": 3, "final_refit": 1}
    estimate = result.get("runtime_seconds", float("inf")) * sum(groups.values()) * 2
    allowed = result["status"] == "ok" and estimate <= CONFIG["sfs_gate_seconds"]
    payload = {"purpose": "runtime_only_not_model_selection", "status": "PASS" if allowed else "STOP_BUDGET_GATE",
               "resumed": resumed, "pilot": result, "partition_groups": groups,
               "safety_factor": 2, "estimated_sfs_seconds": estimate if np.isfinite(estimate) else None,
               "threshold_seconds": 3600, "full_sfs_authorized_by_gate": allowed,
               "projection_limit": "conservative full-n cost for smaller folds; not a timing guarantee; deterministic prefix reuse tested separately",
               "benchmark_support_or_scores_used_for_selection": False}
    atomic_json(OUTPUT / "sfs_budget.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "pilot"}, ensure_ascii=False), flush=True)
    return allowed


def fold_ranges(n, splits):
    return [(tr, va) for tr, va in TimeSeriesSplit(n_splits=splits, gap=6).split(range(n))]


def mean_candidates(records, grid):
    output = []
    for k in grid:
        rows = [r for r in records if r["requested_k"] == k]
        valid = bool(rows) and all(r["status"] == "ok" for r in rows)
        output.append({"requested_k": k, "status": "ok" if valid else "incomplete_or_invalid",
                       "fold_count": len(rows), "mean_selected_k": float(np.mean([r["selected_k"] for r in rows])) if rows and all(r["selected_k"] is not None for r in rows) else None,
                       **{metric: float(np.mean([r[metric] for r in rows])) if valid else None for metric in ("rmse", "mae", "r2")},
                       "runtime_seconds": sum(r["runtime_seconds"] for r in rows)})
    return output


def choose(candidates):
    usable = [r for r in candidates if r["status"] == "ok"]
    if not usable:
        return None
    minimum = min(r["rmse"] for r in usable)
    tied = [r for r in usable if r["rmse"] <= minimum + 1e-9]
    return min(tied, key=lambda r: (r["mean_selected_k"], r["runtime_seconds"],
                                   list(METHODS).index(r["method"]) if "method" in r else 0))


def run_formal():
    if not run_pilot():
        print("STOP: SFS budget gate requires human decision", flush=True)
        return
    all_records, cases, choices = [], [], []
    started = perf_counter()
    sfs_seconds = 0.

    def run(method, train, valid, level, outer=None, inner=None, grid=None, final=False, save=False):
        nonlocal sfs_seconds
        case = {"method": method, "grid": CONFIG["grids"][method] if grid is None else grid,
                "train": [int(train[0]), int(train[-1]) + 1],
                "validation": [int(valid[0]), int(valid[-1]) + 1] if valid is not None else None,
                "final_refit": final, "save_model": save}
        timeout = min(120., 3600. - sfs_seconds) if method == "forward_sfs" else 120.
        if timeout <= 0:
            raise RuntimeError("SFS measured budget exhausted before next case")
        result, resumed = checkpoint_case(case, timeout=timeout)
        if method == "forward_sfs":
            sfs_seconds += result["wall_seconds"]
            if sfs_seconds > 3600:
                raise RuntimeError("SFS measured budget exhausted; checkpoints retained; no retry")
        cases.append({**result, "resumed": resumed})
        records = []
        source_rows = result["records"]
        if result["status"] != "ok":
            source_rows = [{"method": method, "requested_k": k, "selected_k": None,
                            "selection": {}, "status": result["status"], "rmse": None, "mae": None,
                            "r2": None, "runtime_seconds": result["wall_seconds"] / len(case["grid"])} for k in case["grid"]]
        for row in source_rows:
            record = {**row, "level": level, "outer_fold": outer, "inner_fold": inner,
                      "train_range": case["train"], "validation_range": case["validation"],
                      "random_state": 42, "checkpoint_id": result["checkpoint_id"]}
            records.append(record)
            all_records.append(record)
        print(f"{level} {method} outer={outer} inner={inner}: {result['status']} ({result['wall_seconds']:.2f}s; resumed={resumed})", flush=True)
        return records

    for outer, (train, valid) in enumerate(fold_ranges(15788, 5), 1):
        for method in METHODS:
            if method in {"lasso_cv", "rfecv"}:
                run(method, train, valid, "outer_evaluation", outer)
                continue
            inner_rows = []
            for inner, (itr, iva) in enumerate(fold_ranges(len(train), 3), 1):
                inner_rows += run(method, train[itr], train[iva], "inner_k_cv", outer, inner)
            candidates = mean_candidates(inner_rows, CONFIG["grids"][method])
            best = choose(candidates)
            choices.append({"method": method, "outer_fold": outer, "best_inner": best,
                            "candidates": candidates, "selection_source": "inner_k_cv_only"})
            curve = run(method, train, valid, "outer_k_curve", outer)
            if best is not None:
                selected = next(r for r in curve if r["requested_k"] == best["requested_k"])
                all_records.append({**selected, "level": "outer_evaluation"})
            else:
                all_records.append({"method": method, "requested_k": None, "selected_k": None,
                                    "selection": {}, "status": "no_valid_inner_candidate", "rmse": None,
                                    "mae": None, "r2": None, "runtime_seconds": 0., "level": "outer_evaluation",
                                    "outer_fold": outer, "inner_fold": None,
                                    "train_range": [int(train[0]), int(train[-1])+1],
                                    "validation_range": [int(valid[0]), int(valid[-1])+1], "random_state": 42,
                                    "checkpoint_id": None})
    finalists = []
    for method in METHODS:
        rows = []
        for inner, (train, valid) in enumerate(fold_ranges(15788, 3), 1):
            rows += run(method, train, valid, "final_selection_cv", inner=inner)
        candidates = mean_candidates(rows, CONFIG["grids"][method])
        best = choose(candidates)
        if best is not None:
            finalists.append({**best, "method": method})
    winner = choose(finalists)
    final_records = []
    for finalist in finalists:
        final_records += run(finalist["method"], np.arange(15788), None,
                             "final_refit_training", grid=[finalist["requested_k"]], final=True,
                             save=finalist["method"] == winner["method"])
    payload = {"status": ("complete" if all(c["status"] == "ok" for c in cases) else "incomplete") if winner is not None else "no_valid_final_model",
               "config": CONFIG, "fingerprint": fingerprint(), "records": all_records,
               "inner_choices": choices, "finalists": finalists, "winner": winner,
               "final_refits": final_records, "cases": [{k: v for k, v in c.items() if k != "records"} for c in cases],
               "wall_seconds_this_invocation": perf_counter() - started,
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(OUTPUT / "feature_selection_results.json", payload)
    from .phase3b_report import generate_outputs
    generate_outputs(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(execute_case(json.loads(args.worker)), allow_nan=False))
    elif args.run:
        run_formal()
    elif args.pilot:
        run_pilot()
    else:
        parser.error("choose --pilot or --run")


if __name__ == "__main__":
    main()
