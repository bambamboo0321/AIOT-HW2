"""Phase 3D：抽樣、時間隔離、快取等價與已執行成果的一致性。"""

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.phase2_common import ROOT
from src.feature_selection import fit_preprocessor
from src.phase3d_search import (K, OUTPUT, TIE_TOLERANCE, check_protected, load_conditions,
                                output_path, prepare_cv, sample_combinations, score_subset)
from src.selection_runner import TrainingPartition
from src.validate_data import sha256_file


def synthetic():
    _, cfg, _ = load_conditions()
    rng = np.random.default_rng(10)
    names = cfg.feature_columns
    X = pd.DataFrame(rng.normal(size=(120, len(names))), columns=names)
    X.loc[[1, 50, 90], names[1]] = np.nan
    X["date"] = pd.date_range("2020-01-01", periods=len(X), freq="10min")
    y = pd.Series(2 * X[names[0]] - X[names[5]] + rng.normal(size=len(X)), name="Appliances")
    return TrainingPartition(X, y, tuple(X.index), tuple(range(120, 150)), "synthetic"), cfg


def read_csv(name):
    with (OUTPUT / name).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("feature_indices", "features", "fold_rmse_wh", "fold_mae_wh", "fold_r2"):
            row[key] = json.loads(row[key])
        for key in ("sample_id", "rank"):
            row[key] = int(row[key])
        for key in ("rmse_wh", "mae_wh", "r2", "runtime_seconds"):
            row[key] = float(row[key])
    return rows


class SamplingAndScoringTests(unittest.TestCase):
    def test_1000_unique_legal_six_feature_combinations(self):
        _, cfg, exp = load_conditions()
        combos = sample_combinations(cfg.feature_columns, 1000, exp["random_state"])
        self.assertEqual(len(combos), len(set(combos)))
        self.assertTrue(all(len(c) == K and len(set(c)) == K and min(c) >= 0 and max(c) < 25 for c in combos))

    def test_seed_is_reproducible_and_can_change_sample(self):
        names = list(range(25))
        first = sample_combinations(names, 1000, 42)
        self.assertEqual(first, sample_combinations(names, 1000, 42))
        self.assertNotEqual(first, sample_combinations(names, 1000, 43))

    def test_sample_budget_and_invalid_candidates_are_rejected(self):
        for names, count in ((list(range(25)), 1001), (list(range(25)), 0), (["a"] * 25, 10), (list(range(6)), 2)):
            with self.assertRaises(ValueError):
                sample_combinations(names, count, 42)

    def test_cache_fits_only_training_prefix_and_observes_gap(self):
        part, cfg = synthetic()
        seen = []
        def record(X, y, config):
            seen.append(list(X.index))
            return fit_preprocessor(X, y, config)
        with patch("src.phase3d_search.fit_preprocessor", side_effect=record):
            folds = prepare_cv(part, cfg, 3, 6)
        for fold, observed in zip(folds, seen):
            self.assertEqual(observed, list(part.X.index[fold.train_indices]))
            self.assertEqual(set(observed) & set(part.X.index[fold.validation_indices]), set())
            self.assertGreater(fold.validation_indices[0] - fold.train_indices[-1], 6)
            self.assertLess(fold.validation_indices[-1], len(part.X))
            scaler = fold.processor.named_steps["columns"].named_transformers_["c0"].named_steps["scale"]
            self.assertAlmostEqual(scaler.mean_[0], part.X.iloc[fold.train_indices][cfg.feature_columns[0]].mean())

    def test_holdout_indices_rejected_before_preprocessing(self):
        part, cfg = synthetic()
        part.forbidden_indices = (30,)
        with patch("src.phase3d_search.fit_preprocessor", side_effect=AssertionError("must not fit")):
            with self.assertRaisesRegex(ValueError, "Holdout"):
                prepare_cv(part, cfg, 3, 6)

    def test_energy_loader_parses_training_rows_only(self):
        original = pd.read_csv
        calls = []
        def record(*args, **kwargs):
            calls.append(kwargs.get("nrows"))
            return original(*args, **kwargs)
        with patch("src.selection_runner.pd.read_csv", side_effect=record):
            part, _, exp = load_conditions()
        self.assertEqual(calls, [exp["training_rows"]])
        self.assertEqual(set(part.X.index) & set(part.forbidden_indices), set())

    def test_cached_25_columns_equal_independent_six_column_preprocessing(self):
        part, cfg = synthetic()
        folds = prepare_cv(part, cfg, 3, 6)
        indices = (0, 1, 5, 10, 15, 20)
        names = [cfg.feature_columns[i] for i in indices]
        selected_cfg = replace(cfg, feature_columns=names, column_types={n: "numeric" for n in names})
        direct = []
        for fold in folds:
            train, valid = fold.train_indices, fold.validation_indices
            processor, X = fit_preprocessor(part.X.iloc[train], part.y.iloc[train], selected_cfg)
            validation = processor.transform(part.X.iloc[valid])
            np.testing.assert_allclose(X, fold.train[:, indices], rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(validation, fold.validation[:, indices], rtol=1e-13, atol=1e-13)
            predicted = LinearRegression().fit(X, part.y.iloc[train]).predict(validation)
            direct.append(float(np.sqrt(np.mean((part.y.iloc[valid].to_numpy()-predicted)**2))))
        scored = score_subset(folds, indices)
        np.testing.assert_allclose(scored["fold_rmse_wh"], direct, rtol=1e-12)
        self.assertAlmostEqual(scored["rmse_wh"], np.mean(direct), places=12)

    def test_rmse_matches_independent_numpy_ols_and_fold_mean(self):
        part, cfg = synthetic()
        folds = prepare_cv(part, cfg, 3, 6)
        indices = (0, 2, 5, 7, 13, 24)
        values = []
        for fold in folds:
            design = np.column_stack([np.ones(len(fold.train)), fold.train[:, indices]])
            validation = np.column_stack([np.ones(len(fold.validation)), fold.validation[:, indices]])
            coef = np.linalg.lstsq(design, fold.training_y, rcond=None)[0]
            values.append(float(np.sqrt(np.mean((fold.validation_y-validation@coef)**2))))
        actual = score_subset(folds, indices)
        np.testing.assert_allclose(actual["fold_rmse_wh"], values, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(actual["rmse_wh"], np.mean(values), places=12)

    def test_score_rejects_not_six_or_unknown_features(self):
        part, cfg = synthetic()
        folds = prepare_cv(part, cfg, 3, 6)
        for indices in ((1, 2), (0, 0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 25)):
            with self.assertRaises(ValueError):
                score_subset(folds, indices)

    def test_output_paths_are_confined_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch("src.phase3d_search.OUTPUT", Path(folder)):
            for name in ("../results.csv", "/private/test.csv", ".", ""):
                with self.assertRaises(ValueError):
                    output_path(name)
            target = Path(folder) / "preserved"
            target.write_text("preserve")
            (Path(folder) / "link.csv").symlink_to(target)
            with self.assertRaises(ValueError):
                output_path("link.csv")
            self.assertEqual(target.read_text(), "preserve")


class SavedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((OUTPUT / "comparison_summary.json").read_text())
        cls.rows = read_csv("random_search_results.csv")

    def test_saved_sampling_matches_seed_and_all_1000_cases(self):
        s = self.summary
        ordered = sorted(self.rows, key=lambda r:r["sample_id"])
        names = s["conditions"]["preprocessing"]["feature_columns"]
        expected = sample_combinations(names, 1000, s["conditions"]["random_seed"])
        self.assertEqual([tuple(r["feature_indices"]) for r in ordered], expected)
        self.assertEqual(len(self.rows), 1000)
        for row in ordered:
            self.assertEqual(row["features"], [names[i] for i in row["feature_indices"]])

    def test_csv_json_distribution_rank_and_top_worst_ten_match(self):
        s = self.summary["random_search"]
        self.assertEqual([r["rank"] for r in self.rows], list(range(1,1001)))
        values = np.array([r["rmse_wh"] for r in self.rows])
        self.assertTrue((np.diff(values) >= 0).all())
        self.assertEqual(self.rows[0], s["best"])
        self.assertEqual(self.rows[:10], s["top10"])
        self.assertEqual(read_csv("top10_combinations.csv"), s["top10"])
        self.assertEqual(read_csv("worst10_combinations.csv"), s["worst10"])
        self.assertEqual(s["distribution"]["min"], float(values.min()))
        self.assertAlmostEqual(s["distribution"]["mean"], float(values.mean()), places=12)
        self.assertEqual(s["distribution"]["median"], float(np.median(values)))
        for row in self.rows:
            self.assertEqual(row["rmse_wh"], float(np.mean(row["fold_rmse_wh"])))

    def test_mi_reference_is_separate_and_count_is_reproducible(self):
        s = self.summary
        mi = s["fixed_combinations"]["mutual_information"]
        values = np.array([r["rmse_wh"] for r in self.rows])
        ref = s["random_search"]["mi_reference"]
        self.assertEqual(ref["random_better_than_mi"], int(np.count_nonzero(values < mi["rmse_wh"]-TIE_TOLERANCE)))
        self.assertEqual(ref["competition_rank_among_random_and_mi"], ref["random_better_than_mi"]+1)
        self.assertEqual(ref["in_random_sample"], mi["feature_indices"] in [r["feature_indices"] for r in self.rows])
        self.assertEqual(ref["rank_population"], 1000+(not ref["in_random_sample"]))

    def test_four_selector_methods_keep_fold_local_features_and_ranges(self):
        s = self.summary
        self.assertEqual(len(s["selector_cv"]), 4)
        for method in s["selector_cv"]:
            self.assertEqual(method["evaluation"], "B_selector_inside_each_training_fold")
            self.assertEqual(len(method["folds"]), s["conditions"]["cv_splits"])
            for row, declared in zip(method["folds"], s["conditions"]["folds"]):
                self.assertEqual(row["train_range"], declared["train_range"])
                self.assertEqual(row["validation_range"], declared["validation_range"])
                self.assertEqual(len(row["features"]), 6)
                self.assertLess(row["train_range"][1], row["validation_range"][0])
                self.assertLessEqual(row["validation_range"][1], s["conditions"]["training_rows"])

    def test_csv_encoding_lf_and_graph_data_match_summary(self):
        for name in ("random_search_results.csv", "top10_combinations.csv", "worst10_combinations.csv"):
            raw = (OUTPUT/name).read_bytes();raw.decode("utf-8")
            self.assertNotIn(b"\r",raw)
            self.assertTrue(all(line.rstrip()==line for line in raw.splitlines()))
        plot = json.loads((OUTPUT/"plot_data.json").read_text())["figures"]
        self.assertEqual(set(plot), {"feature_correlation.png","rmse_distribution.png","method_comparison.png"})
        self.assertEqual(plot["rmse_distribution.png"]["rmse_wh"], self.summary["random_search"]["rmse_values_in_sample_order"])
        self.assertEqual(plot["method_comparison.png"]["selector_rmse_wh"], [r["rmse"] for r in self.summary["selector_cv"]])
        for name in plot:
            self.assertTrue((OUTPUT/name).is_file())

    def test_all_original_models_data_research_and_dashboard_unchanged(self):
        self.assertEqual(check_protected()["changed_files"], [])
        for name,digest in self.summary["sources"].items():
            self.assertEqual(sha256_file(ROOT/name),digest)

    def test_search_never_evaluated_holdout_or_exhaustive_space(self):
        self.assertFalse(self.summary["conditions"]["holdout_used"])
        self.assertEqual(self.summary["conditions"]["sample_count"], 1000)
        self.assertFalse(self.summary["exhaustive_search"]["executed"])
        self.assertEqual(self.summary["fixed_sfs_k6"]["status"], "not_available")


if __name__ == "__main__":
    unittest.main()
