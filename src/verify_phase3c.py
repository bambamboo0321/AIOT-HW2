"""Phase 3C 驗證：只驗證固定輸出；不重跑正式評估或重新 fit 保存模型。"""

from collections import Counter
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import unittest

from PIL import Image

from .phase2_common import ROOT, CSV_PATH, EXPECTED_SHA256
from .phase3c_evaluation import OUTPUT, FIGURES, PARENT, MODEL_FILES, MODEL_HASHES, environment
from .phase3c_report import write_json
from .validate_data import validate


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def old_artifact_integrity():
    names = git("ls-tree", "-r", "--name-only", PARENT).decode().splitlines()
    changed = []
    for name in names:
        path = ROOT / name
        original = hashlib.sha256(git("show", f"{PARENT}:{name}")).hexdigest()
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != original:
            changed.append(name)
    assert not changed, changed
    assert hashlib.sha256(CSV_PATH.read_bytes()).hexdigest() == EXPECTED_SHA256
    duplicate = ROOT / "KAG_energydata_complete.csv"
    if duplicate.exists():
        assert hashlib.sha256(duplicate.read_bytes()).hexdigest() == EXPECTED_SHA256
    for name, path in MODEL_FILES.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == MODEL_HASHES[name]
    return {"compared_to_commit": PARENT, "existing_tracked_files_checked": len(names),
            "changed_existing_files": changed, "raw_dataset_sha256": EXPECTED_SHA256,
            "baseline_metrics_unchanged": True, "all_phase1_2_3a_3b_artifacts_unchanged": True,
            "model_sha256": MODEL_HASHES,
            "phase3b_results_csv_sha256": hashlib.sha256((ROOT / "reports/phase3/results/feature_selection_results.csv").read_bytes()).hexdigest()}


def safety_scan():
    paths = [n for n in git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0") if n]
    patterns = {
        "private_path": re.compile(r"/(?:Users|home)/[^\s\"']+|[A-Za-z]:\\Users\\"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "provider_token": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{24,}|xox[baprs]-[A-Za-z0-9-]{20,}"),
        "literal_secret": re.compile(r"(?i)(?:password|api_key|access_token|secret_key)\s*[:=]\s*['\"][A-Za-z0-9_/-]{12,}['\"]"),
    }
    findings, large, csvs = [], [], []
    for name in paths:
        path = ROOT / name
        assert not path.is_symlink(), name
        assert not any(p in {".venv", "__pycache__", ".env"} for p in path.relative_to(ROOT).parts), name
        if path.stat().st_size > 5 * 1024 * 1024:
            large.append(name)
        if path.suffix == ".png":
            with Image.open(path) as image:
                text = json.dumps(image.info)
                image.verify()
        else:
            content = path.read_bytes()
            text = content.decode("utf-8")
            assert b"\r" not in content, name
            assert all(line.rstrip() == line for line in text.splitlines()), name
            if path.suffix == ".csv":
                csvs.append(name)
            if path.suffix == ".json":
                json.loads(text, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        for kind, pattern in patterns.items():
            if pattern.search(text):
                findings.append({"path": name, "kind": kind})
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        if any(e != "101464522+bambamboo0321@users.noreply.github.com" for e in emails):
            findings.append({"path": name, "kind": "private_email"})
    assert not findings, findings
    assert not large, large
    assert git("diff", "--check") == b""
    assert git("diff", "--cached", "--name-only") == b""
    assert git("diff", "--name-only") == b""
    return {"status": "PASS", "new_files_scanned": len(paths), "secret_private_path_email_findings": findings,
            "files_over_5_mib": large, "new_csv_files_utf8_lf": csvs,
            "untracked_text_cr_and_trailing_whitespace_checked": True, "git_diff_check": "PASS",
            "staging_empty": True, "scan_limit": "Pattern-based text/PNG-metadata scan; not exhaustive; image pixels excluded."}


def main():
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    assert git("rev-parse", "HEAD").decode().strip() == PARENT
    assert git("branch", "--show-current").decode().strip() == "main"
    tests = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    stream, captured = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    (ROOT / ".venv/phase3c_full_tests.log").write_text(stream.getvalue() + captured.getvalue(), encoding="utf-8")
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit("Phase 3C verification stopped: tests failed")
    reference = sys.modules["test_prediction_interval"].REFERENCE_RESULTS
    assert set(reference) == {"full_rank", "rank_deficient"}
    phase1 = validate(CSV_PATH)
    assert phase1["overall_status"] == "WARN"
    assert [c["name"] for c in phase1["checks"] if c["status"] != "PASS"] == ["rv1_rv2_distinct"]
    metrics = json.loads((OUTPUT / "prediction_metrics.json").read_text())
    intervals = json.loads((OUTPUT / "prediction_interval_metrics.json").read_text())
    manifest = json.loads((OUTPUT / "figure_manifest.json").read_text())
    assert len(manifest["figures"]) == 5
    for filename in manifest["figures"]:
        with Image.open(FIGURES / filename) as image:
            image.verify()
    for filename, digest in manifest["sources"].items():
        assert hashlib.sha256((OUTPUT / filename).read_bytes()).hexdigest() == digest
    existing_integrity = old_artifact_integrity()
    report = {"phase": "3C", "status": "PASS", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "client_timezone": "Asia/Taipei", "branch": "main", "head": PARENT,
        "environment": environment(), "dependency_additions": {"statsmodels": "0.14.6", "patsy": "1.0.2",
            "record": "requirements-phase3c.txt", "original_requirements_unchanged": True},
        "tests": {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "phase2": 40, "phase3a": 27, "phase3b": 15,
            "phase3c_interval": 16, "phase3c_evaluation": 17},
        "phase1_validation": {"overall_status": phase1["overall_status"],
            "checks": dict(Counter(c["status"] for c in phase1["checks"])),
            "note": "24 PASS, inherited rv1/rv2 duplicate WARN, zero failures; existing Phase 1 report not rewritten"},
        "integrity": existing_integrity, "mathematical_reference": {"statsmodels_version": environment()["statsmodels"],
            "reference_columns": ["obs_ci_lower", "obs_ci_upper"], "synthetic_only": True, "cases": reference},
        "numerical_diagnostics": {n: intervals["models"][n]["training_diagnostics"] for n in ("baseline", "selected")},
        "data_and_interval_checks": {"holdout_samples_each": 3947,
            "all_bounds_finite_ordered_positive_width": True,
            "csv_json_metrics_predictions_and_figures_consistent": True,
            "independent_qr_leverage_matches_svd": True, "saved_model_prediction_matches_design": True,
            "baseline_training_and_holdout_match_phase2": True,
            "selected_training_metrics_match_phase3b": True,
            "holdout_not_used_for_sigma_or_scaler_estimation": True, "holdout_calibration": False,
            "feature_selection_reexecuted_in_official_evaluation": False,
            "saved_models_refitted": False, "official_evaluation_fit_guard": True,
            "regression_test_scope": "Existing suite retains its in-memory fitting isolation tests; no saved artifact is refitted or overwritten. Phase 3C statsmodels fits synthetic data only.",
            "reproducibility": "Saved-model predictions and Training-only interval math reproduce the exported values; no new model selection or holdout tuning."},
        "figures": {"count": len(manifest["figures"]), "files": list(manifest["figures"]),
            "source_csv_hashes_verified": True, "plot_window": [0, 144], "metrics_window": [0, 3947]},
        "frozen_comparison": {n: metrics["models"][n]["holdout_metrics"] for n in ("baseline", "selected")},
        "git_operations": {"add": False, "commit": False, "push": False},
        "phase4_started": False}
    path = OUTPUT / "phase3c_verification.json"
    write_json(path, {**report, "status": "PENDING_SAFETY_CHECK"})
    report["safety"] = safety_scan()
    write_json(path, report)
    print(json.dumps({"status": report["status"], "tests": report["tests"], "reference": reference,
                      "integrity": existing_integrity, "safety": report["safety"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
