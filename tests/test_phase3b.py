"""Phase 3B：路徑等價、隔離、無效結果、Checkpoint 與硬 Timeout。"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.feature_selection import SelectionSpec, count_fits, fit_selected_regression
from src.phase3b_selection import fit_group
from src.phase3b_runner import checkpoint_case, choose, execute_case, mean_candidates, run_formal
from src.phase3b_report import aggregate, export_csv
from src.preprocessing_config import PreprocessingConfig


def fixture():
    rng = np.random.default_rng(53)
    X = pd.DataFrame(rng.normal(size=(96, 5)), columns=[f"x{i}" for i in range(5)])
    X.iloc[48:] *= [1., 2., 10., 100., .01]
    y = pd.Series(3 * X.x0 - X.x1 + rng.normal(size=96), name="y")
    X.loc[[4, 35, 68], "x2"] = np.nan
    return X, y, PreprocessingConfig("y", list(X.columns), numerical_scaling="standard")


class Phase3BTests(unittest.TestCase):
    def test_reused_paths_match_independent_sklearn_selection_and_predictions(self):
        X, y, config = fixture()
        for method in ("f_test", "mutual_information", "rfe", "forward_sfs"):
            models, _ = fit_group(X, y, config, method, [1, 2, 4, 5])
            for k, model in models.items():
                with self.subTest(method=method, k=k):
                    direct = fit_selected_regression(X, y, config, SelectionSpec(method, k))
                    np.testing.assert_array_equal(model.selection.indices, direct.selection.indices)
                    np.testing.assert_allclose(model.predict(X), direct.predict(X), atol=1e-10)
                    self.assertIsInstance(model.predictor, LinearRegression)

    def test_sfs_prefix_reuse_reduces_fits_without_dropping_k(self):
        X, y, config = fixture()
        with count_fits() as counts:
            models, _ = fit_group(X, y, config, "forward_sfs", [1, 2, 4, 5])
        self.assertEqual(set(models), {1, 2, 4, 5})
        self.assertEqual(counts["selector_linear_regression_fits"], 3 * (5+4+3+2))

    def test_rfe_grid_without_one_still_matches_direct_selection(self):
        X, y, config = fixture()
        models, _ = fit_group(X, y, config, "rfe", [2, 4])
        for k, model in models.items():
            direct = fit_selected_regression(X, y, config, SelectionSpec("rfe", k))
            np.testing.assert_array_equal(model.selection.indices, direct.selection.indices)

    def test_preprocessing_is_fit_on_current_training_only(self):
        X, y, config = fixture()
        models, _ = fit_group(X.iloc[:48], y.iloc[:48], config, "rfe", [1, 3])
        processor = models[3].selection.preprocessor
        scaler = processor.named_steps["columns"].named_transformers_["c0"].named_steps["scale"]
        self.assertEqual(scaler.n_samples_seen_, 48)
        self.assertAlmostEqual(scaler.mean_[0], X.iloc[:48].x0.mean())
        before = models[3].predict(X.iloc[48:]).copy()
        X.iloc[48:] = 1e10
        self.assertEqual(scaler.n_samples_seen_, 48)
        self.assertTrue(np.isfinite(before).all())

    def test_group_rejects_invalid_or_duplicate_grids(self):
        X, y, config = fixture()
        for grid in ([0], [6], [1, 1], []):
            with self.assertRaises(ValueError):
                fit_group(X, y, config, "f_test", grid)

    def test_worker_rejects_holdout_and_missing_gap_before_fit(self):
        X, y, config = fixture()
        with patch("src.phase3b_runner.load_energy_training", return_value=(SimpleNamespace(X=X, y=y), config)):
            for train, valid in (([0, 97], None), ([0, 72], [75, 90]), ([0, 72], [79, 99])):
                with self.assertRaises(ValueError):
                    execute_case({"method": "f_test", "grid": [1], "train": train, "validation": valid})

    def test_lasso_zero_support_keeps_null_lr_scores(self):
        X, y, config = fixture()
        y[:] = 0.
        with patch("src.phase3b_runner.load_energy_training", return_value=(SimpleNamespace(X=X, y=y), config)):
            result = execute_case({"method": "lasso_cv", "grid": [None], "train": [0, 72], "validation": [79, 96]})
        record = result["records"][0]
        self.assertEqual(record["selected_k"], 0)
        self.assertEqual(record["status"], "invalid_for_LR")
        self.assertTrue(all(record[m] is None for m in ("rmse", "mae", "r2")))
        self.assertNotIn("downstream_linear_regression_fits", result["fit_counts"])

    def test_inner_choice_excludes_any_failed_candidate(self):
        records = []
        for k in (1, 2):
            for fold in range(3):
                records.append({"requested_k": k, "selected_k": k, "status": "invalid_for_LR" if k == 1 and fold == 2 else "ok",
                                "rmse": None if k == 1 and fold == 2 else k,
                                "mae": None if k == 1 and fold == 2 else k,
                                "r2": None if k == 1 and fold == 2 else 0., "runtime_seconds": 1.})
        candidates = mean_candidates(records, [1, 2])
        self.assertIsNone(candidates[0]["rmse"])
        self.assertEqual(choose(candidates)["requested_k"], 2)

    def test_predeclared_tie_rule_prefers_fewer_features(self):
        candidates = [{"status": "ok", "rmse": 10., "mean_selected_k": k, "runtime_seconds": 20. - k} for k in (10, 2)]
        self.assertEqual(choose(candidates)["mean_selected_k"], 2)
        self.assertIsNone(choose([]))

    def test_checkpoint_skips_completed_case_and_invalidates_changed_code(self):
        case = {"method": "f_test", "grid": [1], "train": [0, 72], "validation": [79, 96]}
        bounded = {"status": "ok", "stdout": json.dumps({"status": "ok", "case": case, "records": []}), "wall_seconds": .1}
        with tempfile.TemporaryDirectory() as folder, patch("src.phase3b_runner.run_bounded", return_value=bounded) as run:
            first, reused = checkpoint_case(case, cache=folder)
            self.assertFalse(reused)
            second, reused = checkpoint_case(case, cache=folder)
            self.assertTrue(reused)
            self.assertEqual(first, second)
            self.assertEqual(run.call_count, 1)
            with patch("src.phase3b_runner.fingerprint", return_value="different-code"):
                _, reused = checkpoint_case(case, cache=folder)
                self.assertFalse(reused)
            self.assertEqual(run.call_count, 2)

    def test_timeout_kills_worker_and_is_not_automatically_retried(self):
        case = {"method": "forward_sfs", "grid": [1, 2, 4], "train": [0, 72], "validation": [79, 96]}
        with tempfile.TemporaryDirectory() as folder:
            result, reused = checkpoint_case(case, cache=folder, timeout=.05)
            self.assertFalse(reused)
            self.assertEqual(result["status"], "timeout")
            self.assertTrue(result["child_reaped"])
            with patch("src.phase3b_runner.run_bounded", side_effect=AssertionError("must not retry")):
                again, reused = checkpoint_case(case, cache=folder)
            self.assertTrue(reused)
            self.assertEqual(again, result)

    def test_plot_aggregation_requires_all_outer_folds(self):
        rows = [{"method": "f_test", "level": "outer_evaluation", "selected_k": 2,
                 "status": "ok", "rmse": 3., "mae": 2., "r2": .5} for _ in range(5)]
        self.assertEqual(aggregate(pd.DataFrame(rows), "outer_evaluation", "f_test")["rmse"], 3.)
        self.assertIsNone(aggregate(pd.DataFrame(rows[:4]), "outer_evaluation", "f_test")["rmse"])

    def test_fixed_group_is_reproducible(self):
        X, y, config = fixture()
        first, _ = fit_group(X, y, config, "mutual_information", [1, 3, 5])
        second, _ = fit_group(X, y, config, "mutual_information", [1, 3, 5])
        for k in first:
            np.testing.assert_array_equal(first[k].selection.indices, second[k].selection.indices)
            np.testing.assert_array_equal(first[k].predict(X), second[k].predict(X))

    def test_failed_sfs_gate_never_starts_formal_cases(self):
        with patch("src.phase3b_runner.run_pilot", return_value=False), patch("src.phase3b_runner.checkpoint_case", side_effect=AssertionError("must stop")):
            run_formal()

    def test_csv_preserves_invalid_status_and_null_metrics(self):
        record = {"method": "lasso_cv", "level": "outer_evaluation", "requested_k": None,
                  "selected_k": 0, "selection": {"selected_feature_indices": [], "selected_feature_names": []},
                  "status": "invalid_for_LR", "rmse": None, "mae": None, "r2": None,
                  "train_range": [0, 72], "validation_range": [79, 96]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            export_csv([record], path)
            row = pd.read_csv(path).iloc[0]
        self.assertEqual(row.status, "invalid_for_LR")
        self.assertEqual(row.selected_k, 0)
        self.assertTrue(pd.isna(row.rmse))
        self.assertEqual(json.loads(row.selected_feature_indices), [])


if __name__ == "__main__":
    unittest.main()
