"""Phase 3B 驗證；不覆寫 Phase 1／2／3A 紀錄或模型。"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import unittest
import zlib

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LinearRegression

from .feature_selection import SelectionSpec, fit_selected_regression
from .phase2_common import ROOT, CSV_PATH, EXPECTED_SHA256
from .phase3b_runner import CONFIG, OUTPUT, atomic_json, checkpoint_case
from .selection_runner import load_energy_training, regression_metrics
from .validate_data import validate

PARENT = "8f83cd336e02dfb026d1979c835d5c5e871b22de"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def protected_integrity():
    paths = git("ls-tree", "-r", "--name-only", PARENT).decode().splitlines()
    changed = []
    for name in paths:
        path = ROOT / name
        expected = hashlib.sha256(git("show", f"{PARENT}:{name}")).hexdigest()
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(name)
    assert not changed, changed
    assert hashlib.sha256(CSV_PATH.read_bytes()).hexdigest() == EXPECTED_SHA256
    duplicate = ROOT / "KAG_energydata_complete.csv"
    assert not duplicate.exists() or hashlib.sha256(duplicate.read_bytes()).hexdigest() == EXPECTED_SHA256
    return {"compared_to_commit": PARENT, "original_tracked_files_checked": len(paths),
            "changed_original_files": changed, "raw_csv_sha256": EXPECTED_SHA256,
            "phase1_2_3a_artifacts_unchanged": True, "baseline_metrics_and_models_unchanged": True}


def verify_records(payload, frame):
    assert len(frame) == len(payload["records"])
    for method, grid in CONFIG["grids"].items():
        for fold in range(1, 6):
            level = "outer_evaluation" if method in {"lasso_cv", "rfecv"} else "outer_k_curve"
            rows = frame[(frame.method == method) & (frame.level == level) & (frame.outer_fold == fold)]
            actual = [None if pd.isna(k) else int(k) for k in rows.requested_k]
            assert actual == grid
        for fold in range(1, 4):
            rows = frame[(frame.method == method) & (frame.level == "final_selection_cv") & (frame.inner_fold == fold)]
            actual = [None if pd.isna(k) else int(k) for k in rows.requested_k]
            assert actual == grid
    common_folds = {}
    for fold in range(1, 6):
        rows = frame[(frame.level == "outer_evaluation") & (frame.outer_fold == fold)]
        assert len(rows) == 6
        assert rows.train_range.nunique() == 1 and rows.validation_range.nunique() == 1
        common_folds[fold] = {"train_range": json.loads(rows.iloc[0].train_range),
                              "validation_range": json.loads(rows.iloc[0].validation_range)}
    for record in payload["records"]:
        start, end = record["train_range"]
        assert 0 <= start < end <= 15788
        valid = record["validation_range"]
        if valid:
            assert end + 6 <= valid[0] < valid[1] <= 15788
        selection = record["selection"]
        indices = selection.get("selected_feature_indices", [])
        if record["status"] == "ok":
            assert len(indices) == record["selected_k"]
            assert indices == sorted(set(indices)) and all(0 <= i < 25 for i in indices)
            assert all(np.isfinite(record[m]) for m in ("rmse", "mae", "r2"))
        elif record["status"] == "invalid_for_LR":
            assert record["selected_k"] == 0
            assert all(record[m] is None for m in ("rmse", "mae", "r2"))
    for choice in payload["inner_choices"]:
        inner = frame[(frame.level == "inner_k_cv") & (frame.method == choice["method"]) & (frame.outer_fold == choice["outer_fold"])]
        for candidate in choice["candidates"]:
            selected = inner[inner.requested_k == candidate["requested_k"]]
            assert len(selected) == 3
            if candidate["status"] == "ok":
                np.testing.assert_allclose(candidate["rmse"], selected.rmse.mean(), rtol=1e-12)
        valid = [c for c in choice["candidates"] if c["status"] == "ok"]
        if valid:
            assert choice["best_inner"]["rmse"] <= min(c["rmse"] for c in valid) + 1e-9
        rows = frame[(frame.level == "outer_evaluation") & (frame.method == choice["method"]) & (frame.outer_fold == choice["outer_fold"])]
        if choice["best_inner"]:
            assert len(rows) == 1 and rows.iloc[0].requested_k == choice["best_inner"]["requested_k"]
    winner = payload["winner"]
    assert winner is not None
    assert winner["rmse"] <= min(r["rmse"] for r in payload["finalists"]) + 1e-9
    assert all(not c.get("holdout_target_read", False) for c in payload["cases"])
    return {"records_checked": len(frame), "levels": dict(Counter(frame.level)),
            "status_counts": dict(Counter(frame.status)), "range_and_gap_checks": "PASS",
            "all_predeclared_k_executed": True, "common_outer_folds": common_folds,
            "inner_choices_match_outer_evaluation": True,
            "winner_uses_final_selection_cv_only": True, "holdout_used_for_selection": False,
            "limitation": "Range checks, loader contract and fold-local integration tests; not a claim of unbiased best-of-six assessment."}


def verify_figures(frame):
    manifest = json.loads((OUTPUT / "plot_data.json").read_text())
    assert manifest["csv_sha256"] == hashlib.sha256((OUTPUT / "feature_selection_results.csv").read_bytes()).hexdigest()
    point_count = 0
    for name, figure in manifest["figures"].items():
        with Image.open(ROOT / "reports/phase3/figures" / name) as img:
            img.verify()
        for point in figure.get("points", []):
            rows = frame[(frame.level == point["level"]) & (frame.method == point["method"])]
            if point.get("requested_k") is not None:
                rows = rows[rows.requested_k == point["requested_k"]]
            assert np.array_equal(rows.selected_k.to_numpy(), point["selected_k"])
            for metric in ("rmse", "mae", "r2"):
                if metric not in point:
                    continue
                expected = rows[metric].mean() if len(rows) == 5 and (rows.status == "ok").all() else None
                if expected is None:
                    assert point[metric] is None
                else:
                    np.testing.assert_allclose(point[metric], expected, rtol=1e-12, atol=1e-10)
            point_count += 1
    selected_data = manifest["figures"]["selected_features_comparison.png"]["selections"]
    for point in selected_data:
        row = frame[(frame.level == "final_refit_training") & (frame.method == point["method"])].iloc[0]
        assert point["indices"] == json.loads(row.selected_feature_indices)
    for method, values in manifest["figures"]["runtime_comparison.png"]["methods"].items():
        dedup = frame[frame.method == method].drop_duplicates(["checkpoint_id", "requested_k"])
        np.testing.assert_allclose(values["csv_fit_seconds"], dedup.runtime_seconds.sum(), rtol=1e-12)
    assert len(manifest["figures"]) == 7
    return {"figures_verified": len(manifest["figures"]), "points_compared_to_csv": point_count,
            "csv_hash_matches": True, "selected_features_and_runtime_match_csv": True,
            "sfs_missing_values_interpolated": False}


def verify_reproduction(payload):
    partition, config = load_energy_training()
    winner = payload["winner"]
    saved = joblib.load(OUTPUT / "models/final_linear_regression.joblib")
    assert type(saved.predictor) is LinearRegression
    with threadpool_limits(limits=1):
        direct = fit_selected_regression(partition.X, partition.y, config, SelectionSpec(
            winner["method"], winner["requested_k"], max_iter=100000))
        np.testing.assert_array_equal(saved.selection.indices, direct.selection.indices)
        np.testing.assert_allclose(saved.predict(partition.X), direct.predict(partition.X), rtol=1e-12, atol=1e-10)
        # Real 25-feature SFS prefix check against independently executed sklearn k=10.
        reference = next(r for r in payload["records"] if r["level"] == "outer_k_curve" and r["method"] == "forward_sfs" and r["outer_fold"] == 5 and r["requested_k"] == 10)
        train = slice(*reference["train_range"])
        valid = slice(*reference["validation_range"])
        independent = fit_selected_regression(partition.X.iloc[train], partition.y.iloc[train], config, SelectionSpec("forward_sfs", 10))
        np.testing.assert_array_equal(reference["selection"]["selected_feature_indices"], independent.selection.indices)
        scores = regression_metrics(partition.y.iloc[valid], independent.predict(partition.X.iloc[valid]))
        for metric in ("rmse", "mae", "r2"):
            np.testing.assert_allclose(reference[metric], scores[metric], rtol=1e-12, atol=1e-10)
    resumed = 0
    for case in payload["cases"]:
        result, reused = checkpoint_case(case["case"])
        assert reused and result["checkpoint_id"] == case["checkpoint_id"]
        resumed += 1
    return {"independent_winner_full_training_refit_matches_saved_model": True,
            "independent_real_data_sfs_k10_matches_reused_prefix_and_metrics": True,
            "completed_cases_resumed_without_reexecution": resumed,
            "timeout_and_no_retry": "covered by real subprocess tests",
            "timings_are_nondeterministic": True}


def safety_scan():
    new_paths = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    new_paths = [name for name in new_paths if name]
    patterns = {
        "private_absolute_path": re.compile(r"/(?:Users|home)/[^\s\"']+|[A-Za-z]:\\Users\\"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "provider_token": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,}|xox[baprs]-[A-Za-z0-9-]{20,}"),
        "literal_secret": re.compile(r"(?i)(?:password|api_key|access_token|secret_key)\s*[:=]\s*['\"][A-Za-z0-9_/-]{12,}['\"]"),
    }
    findings, large = [], []
    def model_string_metadata(model):
        # Email regex 不能掃 raw BININT2／ndarray bytes：整數序列也可拼出 @、點與字母。
        # 對自己產生且已驗證可載入的 model，另遍歷實際字串 metadata。
        strings, seen = [], set()
        def visit(value):
            if isinstance(value, str):
                strings.append(value)
                return
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(key)
                    visit(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
            elif isinstance(value, np.ndarray) and value.dtype.kind in {"O", "U", "S"}:
                for item in value.flat:
                    visit(item)
            elif hasattr(value, "__dict__") and not isinstance(value, type):
                for item in vars(value).values():
                    visit(item)
        visit(model)
        return "\n".join(strings)

    for name in new_paths:
        path = ROOT / name
        assert not any(part in {".venv", "__pycache__", ".env"} for part in path.relative_to(ROOT).parts)
        if path.stat().st_size > 5 * 1024 * 1024:
            large.append({"path": name, "bytes": path.stat().st_size})
        if path.suffix == ".png":
            with Image.open(path) as img:
                text = json.dumps(img.info)
        elif path.suffix == ".joblib":
            text = zlib.decompress(path.read_bytes()).decode("latin1")
        else:
            text = path.read_text()
        for kind, pattern in patterns.items():
            if pattern.search(text):
                findings.append({"path": name, "kind": kind})
        email_text = model_string_metadata(joblib.load(path)) if path.suffix == ".joblib" else text
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email_text)
        if any(email != "101464522+bambamboo0321@users.noreply.github.com" for email in emails):
            findings.append({"path": name, "kind": "private_email"})
        if path.suffix not in {".png", ".joblib"}:
            assert all(line.rstrip() == line for line in text.splitlines()), name
    assert not findings, findings
    assert not large, large
    assert git("diff", "--check") == b""
    assert git("diff", "--cached", "--name-only") == b""
    return {"findings": findings, "new_files_over_5_mib": large, "untracked_files_scanned": len(new_paths),
            "diff_check": "PASS", "staging_empty": True,
            "scan_limit": "Pattern-based text/PNG-metadata/decompressed-joblib scan; model emails scanned on actual string metadata to exclude BININT2 numerical false positives; binary image pixels excluded; not exhaustive."}


def main():
    assert git("rev-parse", "HEAD").decode().strip() == PARENT
    assert git("branch", "--show-current").decode().strip() == "main"
    payload = json.loads((OUTPUT / "feature_selection_results.json").read_text())
    frame = pd.read_csv(OUTPUT / "feature_selection_results.csv")
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / "tests")))
    (ROOT / ".venv/phase3b_tests_final.log").write_text(log.getvalue())
    assert tests.wasSuccessful(), log.getvalue()
    phase1 = validate(CSV_PATH)
    assert phase1["overall_status"] != "FAIL"
    atomic_json(ROOT / "reports/phase3/phase3b_phase1_validation.json", phase1)
    report = {"phase": "3B", "status": "PASS" if payload["status"] == "complete" else "INCOMPLETE",
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "branch": "main", "head": PARENT,
              "tests": {"run": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors),
                        "skipped": len(tests.skipped), "phase2": 40, "phase3a": 27, "phase3b": tests.testsRun - 67},
              "phase1_validation": {"overall_status": phase1["overall_status"],
                                    "checks": dict(Counter(c["status"] for c in phase1["checks"])),
                                    "note": "Existing rv1/rv2 duplicate WARN retained; zero failures."},
              "integrity": protected_integrity(), "records": verify_records(payload, frame),
              "figures": verify_figures(frame), "reproduction_and_resume": verify_reproduction(payload),
              "git_operations": {"add": False, "commit": False, "push": False},
              "phase3c_started": False}
    # Write once before scanning so this report is included in the scan.
    atomic_json(ROOT / "reports/phase3/phase3b_verification.json", {**report, "status": "PENDING_SAFETY_CHECK"})
    report["safety"] = safety_scan()
    atomic_json(ROOT / "reports/phase3/phase3b_verification.json", report)
    print(json.dumps({"status": report["status"], "tests": report["tests"], "phase1": report["phase1_validation"],
                      "figures": report["figures"], "reproduction": report["reproduction_and_resume"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
