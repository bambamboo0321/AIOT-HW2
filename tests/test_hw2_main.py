"""Phase 4B-2：唯讀成果介接、驗證層級、匯出界線及失敗處理。"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import hw2_main
from src.dashboard_data import (BASELINE, BEST, EXPORT_NAMES, INTERVALS, METRICS,
                                ROOT, ResultError, ResultStore, validate_document)


def read(path):
    return json.loads((ROOT / path).read_text())


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for folder in ("reports", "data"):
            shutil.copytree(ROOT / folder, root / folder)
        yield root, ResultStore(root)


class EntryPointTests(unittest.TestCase):
    def test_cli_help_runs_from_repository_root(self):
        result = subprocess.run([sys.executable, "-B", "hw2_main.py", "--help"], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in hw2_main.COMMANDS:
            self.assertIn(command, result.stdout)

    def test_cli_dataset_prints_valid_json(self):
        result = subprocess.run([sys.executable, "-B", "hw2_main.py", "dataset-summary"], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["artifact_type"], "dataset_summary")

    def test_dataset_summary_matches_existing_metadata(self):
        doc = ResultStore().document("dataset_summary")
        expected = read(BASELINE)
        self.assertEqual(doc["data"]["dataset"], expected["dataset_information"])
        self.assertEqual(doc["data"]["baseline_features"], expected["feature_names"])
        self.assertEqual(doc["data"]["profile"]["rows"], expected["dataset_information"]["shape"]["rows"])

    def test_crisp_dm_has_six_stages_and_local_deployment_boundary(self):
        stages = ResultStore().document("crisp_dm")["data"]["stages"]
        self.assertEqual(len(stages), 6)
        self.assertEqual(stages[-1]["name"], "Deployment")
        self.assertIn("未正式部署", stages[-1]["summary"])
        self.assertTrue(all((ROOT / row["evidence"]).is_file() for row in stages))

    def test_six_methods_keep_training_cv_nested_cv_and_refit_separate(self):
        data = ResultStore().document("feature_selection")["data"]
        expected = read(BEST)
        self.assertEqual(len(data["methods"]), 6)
        self.assertEqual(data["winner"], expected["winner"])
        for row, cv, nested, refit in zip(data["methods"], expected["by_method"],
                                         expected["nested_outer_assessment"], expected["full_training_refits"]):
            self.assertEqual(row["training_cv"], cv)
            self.assertEqual(row["nested_cv"], nested)
            self.assertEqual(row["full_training_refit"], refit)
            self.assertNotIn("holdout_metrics", row)
        lasso = next(row for row in data["methods"] if row["method"] == "lasso_cv")
        self.assertIsNone(lasso["training_cv"]["requested_k"])
        self.assertNotEqual(lasso["training_cv"]["mean_selected_k"], lasso["full_training_refit"]["selected_k"])
        self.assertFalse(data["holdout_used_for_selection"])

    def test_model_comparison_reads_metrics_without_rounding_or_mixing_random_split(self):
        data = ResultStore().document("model_comparison")["data"]
        expected = read(METRICS)
        for name, model in data["models"].items():
            self.assertEqual(model["holdout_metrics"], expected["models"][name]["holdout_metrics"])
            self.assertEqual(model["training_metrics"], expected["models"][name]["training_metrics"])
        random = read(BASELINE)["baselines"]["random"]
        self.assertEqual(data["random_baseline_comparison_only"]["test_metrics"], random["test_metrics"])
        self.assertIn("already viewed", data["plan"]["holdout_role"])

    def test_prediction_intervals_keep_coverage_as_fraction_and_width_in_wh(self):
        doc = ResultStore().document("prediction_intervals")
        self.assertEqual(doc["data"]["models"], read(INTERVALS)["models"])
        self.assertEqual(doc["units"]["coverage"], "fraction [0,1]")
        self.assertEqual(doc["units"]["interval_width"], "Wh")
        self.assertEqual(doc["data"]["interval_type"], "individual_observation_prediction_interval")

    def test_figure_manifest_includes_every_existing_figure_and_relative_paths(self):
        figures = ResultStore().document("figures_manifest")["data"]["figures"]
        actual = {row["path"] for row in figures}
        expected = {p.relative_to(ROOT).as_posix() for base in ("reports/phase2/figures", "reports/phase3/figures")
                    for p in (ROOT / base).rglob("*.png")}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(figures))
        self.assertTrue(all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in actual))

    def test_export_is_deterministic_and_confined_to_dashboard_directory(self):
        with fixture() as (root, store):
            before = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file()}
            store.export_dashboard()
            first = {p.name: p.read_bytes() for p in (root / "dashboard_data").iterdir()}
            store.export_dashboard()
            second = {p.name: p.read_bytes() for p in (root / "dashboard_data").iterdir()}
            self.assertEqual(first, second)
            self.assertEqual(set(first), {name + ".json" for name in EXPORT_NAMES})
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256((root / path).read_bytes()).hexdigest(), digest, str(path))
            added = {p.relative_to(root) for p in root.rglob("*") if p.is_file()} - set(before)
            self.assertEqual(added, {Path("dashboard_data") / name for name in first})
            for content in first.values():
                self.assertNotIn(b"\r", content)
                document = json.loads(content.decode("utf-8"))
                validate_document(document)
                for source in document["sources"]:
                    self.assertEqual(hashlib.sha256((root / source["path"]).read_bytes()).hexdigest(), source["sha256"])

    def test_schema_rejects_missing_fields_unsafe_paths_and_nan(self):
        doc = ResultStore().document("dataset_summary")
        for key in ("artifact_type", "data", "units", "sources", "missing_fields"):
            malformed = dict(doc)
            del malformed[key]
            with self.assertRaises(ResultError):
                validate_document(malformed)
        for path in ("/private/input.csv", "../input.csv"):
            malformed = {**doc, "sources": [{"path": path, "sha256": "a" * 64}]}
            with self.assertRaises(ResultError):
                validate_document(malformed)
        with self.assertRaises(ValueError):
            validate_document({**doc, "data": {"invalid": float("nan")}})

    def test_missing_file_is_explicit_and_export_does_not_start(self):
        with fixture() as (root, store):
            (root / INTERVALS).unlink()
            with self.assertRaisesRegex(ResultError, "prediction_interval_metrics.json"):
                store.export_dashboard()
            self.assertFalse((root / "dashboard_data").exists())

    def test_cli_missing_file_returns_nonzero_without_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            out, err = io.StringIO(), io.StringIO()
            with patch("src.dashboard_data.ResultStore", return_value=ResultStore(folder)), redirect_stdout(out), redirect_stderr(err):
                code = hw2_main.main(["model-comparison"])
            self.assertEqual(code, 1)
            self.assertIn("缺少必要結果檔案", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())
            self.assertEqual(out.getvalue(), "")

    def test_malformed_json_and_missing_metric_never_create_fallback(self):
        with fixture() as (root, store):
            path = root / METRICS
            original = path.read_text()
            for content in ("{broken", '{"x": NaN}', '[]'):
                path.write_text(content)
                with self.assertRaises(ResultError):
                    store.document("model_comparison")
            model = json.loads(original)
            del model["models"]["selected"]["holdout_metrics"]["rmse"]
            path.write_text(json.dumps(model))
            with self.assertRaisesRegex(ResultError, "rmse"):
                store.document("model_comparison")

    def test_modified_dataset_and_model_are_rejected_without_loading_pickle(self):
        with fixture() as (root, store):
            dataset = root / read(BASELINE)["dataset_information"]["csv_path"]
            original = dataset.read_bytes()
            dataset.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ResultError, "SHA-256"):
                store.document("dataset_summary")
            dataset.write_bytes(original)
            model = root / read(METRICS)["models"]["selected"]["model_file"]
            model.write_bytes(model.read_bytes() + b"corrupt")
            with patch("joblib.load", side_effect=AssertionError("must not deserialize")):
                with self.assertRaisesRegex(ResultError, "SHA-256"):
                    store.document("model_comparison")

    def test_duplicate_method_and_inconsistent_csv_are_rejected(self):
        with fixture() as (root, store):
            csv = root / "reports/phase3/results/feature_selection_results.csv"
            original_csv = csv.read_bytes()
            csv.write_bytes(original_csv + b"\n")
            with self.assertRaisesRegex(ResultError, "SHA-256"):
                store.document("feature_selection")
            csv.write_bytes(original_csv)
            best = read(BEST)
            best["nested_outer_assessment"][1] = best["nested_outer_assessment"][0]
            (root / BEST).write_text(json.dumps(best))
            with self.assertRaisesRegex(ResultError, "六種方法"):
                store.document("feature_selection")

    def test_export_rejects_symlink_directory_and_file(self):
        with fixture() as (root, store), tempfile.TemporaryDirectory() as outside:
            directory = root / "dashboard_data"
            directory.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ResultError, "一般目錄"):
                store.export_dashboard()
            self.assertEqual(list(Path(outside).iterdir()), [])
            directory.unlink()
            directory.mkdir()
            destination = Path(outside) / "preserve.txt"
            destination.write_text("preserve")
            (directory / "model_comparison.json").symlink_to(destination)
            with self.assertRaisesRegex(ResultError, "不安全"):
                store.export_dashboard()
            self.assertEqual(destination.read_text(), "preserve")
            self.assertEqual(len(list(directory.iterdir())), 1)

    def test_read_only_commands_never_train_or_load_models_and_preserve_artifacts(self):
        protected = [p for base in ("data/raw", "reports", "references") for p in (ROOT / base).rglob("*") if p.is_file()]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        with patch("sklearn.linear_model.LinearRegression.fit", side_effect=AssertionError("no fit")), \
                patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("no fit")), \
                patch("joblib.load", side_effect=AssertionError("no model loading")):
            store = ResultStore()
            for name in (*EXPORT_NAMES, "crisp_dm"):
                validate_document(store.document(name))
            self.assertEqual(store.verify()["status"], "PASS")
        for p, digest in before.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest, str(p))


if __name__ == "__main__":
    unittest.main()
