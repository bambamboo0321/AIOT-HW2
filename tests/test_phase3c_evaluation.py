"""Phase 3C 保存模型、Training-only PI、完整 Holdout 與輸出一致性。"""

import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from src.phase2_common import ROOT, TARGET, BASELINE_FEATURES, load_dataset
from src.phase3c_evaluation import OUTPUT, SELECTED_FEATURES, MODEL_FILES, MODEL_HASHES
from src.phase3c_evaluation import fixed_partitions, load_models, no_fitting, prediction_metrics, validate_transformation
from src.prediction_interval import design_matrix, estimate_interval, interval_bounds, interval_metrics


class Phase3CEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_dataset()
        cls.train, cls.holdout = fixed_partitions(cls.frame)
        cls.models = load_models()
        cls.predictions = pd.read_csv(OUTPUT / "prediction_interval_predictions.csv", float_precision="round_trip")
        cls.metrics = json.loads((OUTPUT / "prediction_metrics.json").read_text())
        cls.interval_metrics = json.loads((OUTPUT / "prediction_interval_metrics.json").read_text())
        cls.states, cls.bounds = {}, {}
        # 僅使用已保存 Predictor，從 Training 殘差重算數學量以驗證；無 fit。
        with no_fitting():
            for name, model in cls.models.items():
                design = design_matrix(model.features(cls.train))
                cls.states[name] = estimate_interval(design, cls.train[TARGET], model.predict(cls.train))
                cls.bounds[name] = interval_bounds(cls.states[name], design_matrix(model.features(cls.holdout)), model.predict(cls.holdout))

    def test_baseline_loads_expected_saved_estimator_and_25_feature_order(self):
        model = self.models["baseline"]
        self.assertEqual(model.original_features, BASELINE_FEATURES)
        self.assertEqual(model.predictor.n_features_in_, 25)
        self.assertTrue(model.predictor.fit_intercept)

    def test_selected_saved_mi_support_and_six_feature_order(self):
        model = self.models["selected"]
        self.assertEqual(model.original_features, SELECTED_FEATURES)
        self.assertEqual(model.predictor.n_features_in_, 6)
        self.assertEqual(model.artifact.selection.indices.tolist(), [5, 7, 9, 12, 15, 17])
        self.assertEqual(model.artifact.selection.metadata["fit_indices"], list(range(15788)))

    def test_chronological_manifest_and_full_holdout_counts(self):
        self.assertEqual((len(self.train), len(self.holdout)), (15788, 3947))
        self.assertEqual(self.holdout.index.tolist(), list(range(15788, 19735)))
        self.assertLess(self.train.date.max(), self.holdout.date.min())
        self.assertEqual(self.predictions["Row Index"].tolist(), self.holdout.index.tolist())
        np.testing.assert_array_equal(pd.to_datetime(self.predictions.Timestamp).to_numpy(), self.holdout.date.to_numpy())
        np.testing.assert_array_equal(self.predictions.Actual, self.holdout[TARGET])

    def test_reversed_dataset_order_is_not_silently_repaired(self):
        with self.assertRaises(ValueError):
            fixed_partitions(self.frame.iloc[::-1])

    def test_saved_preprocessing_matches_training_statistics(self):
        for model in self.models.values():
            check = validate_transformation(model, self.train, model.features(self.train))
            self.assertTrue(check["matches_saved_input_order"])

    def test_saved_predict_matches_design_and_exported_values(self):
        for name, prefix in (("baseline", "Baseline"), ("selected", "Selected")):
            model = self.models[name]
            calculated = model.features(self.holdout) @ model.predictor.coef_ + model.predictor.intercept_
            np.testing.assert_allclose(model.predict(self.holdout), calculated, rtol=1e-12, atol=1e-9)
            np.testing.assert_array_equal(model.predict(self.holdout), self.predictions[f"{prefix} Prediction"])

    def test_sklearn_metrics_match_independent_numpy_and_json(self):
        actual = self.predictions.Actual.to_numpy()
        for name, prefix in (("baseline", "Baseline"), ("selected", "Selected")):
            pred = self.predictions[f"{prefix} Prediction"].to_numpy()
            errors = actual - pred
            independent = {"mae": np.mean(np.abs(errors)), "mse": np.mean(errors ** 2),
                           "rmse": np.sqrt(np.mean(errors ** 2)),
                           "r2": 1 - np.sum(errors ** 2) / np.sum((actual - actual.mean()) ** 2)}
            measured = prediction_metrics(actual, pred)
            for key, value in independent.items():
                np.testing.assert_allclose(measured[key], value, rtol=1e-12)
                np.testing.assert_allclose(self.metrics["models"][name]["holdout_metrics"][key], value, rtol=1e-12)

    def test_training_residual_variance_not_holdout_calibration(self):
        for name in self.models:
            recorded = self.interval_metrics["models"][name]["training_diagnostics"]
            state = self.states[name].diagnostics
            self.assertFalse(recorded["holdout_residuals_used"])
            self.assertEqual(recorded["training_rows"], 15788)
            for key in ("sse", "sigma_squared", "rank", "degrees_of_freedom"):
                np.testing.assert_allclose(recorded[key], state[key], rtol=1e-12)
            # 當前圖／Coverage 都無法回饋到已保存的 Training sigma。
            prefix = "Baseline" if name == "baseline" else "Selected"
            holdout_mse = np.mean((self.predictions.Actual - self.predictions[f"{prefix} Prediction"]) ** 2)
            self.assertNotAlmostEqual(recorded["sigma_squared"], holdout_mse, places=3)

    def test_rank_df_and_independent_qr_leverage(self):
        for name, model in self.models.items():
            training = design_matrix(model.features(self.train))
            future = design_matrix(model.features(self.holdout))
            _, R = np.linalg.qr(training, mode="reduced")
            independent = np.sum(np.linalg.solve(R.T, future.T) ** 2, axis=0)
            np.testing.assert_allclose(self.bounds[name]["leverage"], independent, rtol=1e-10, atol=1e-12)
            rank = np.linalg.matrix_rank(training)
            self.assertEqual(self.states[name].diagnostics["rank"], rank)
            self.assertEqual(self.states[name].diagnostics["degrees_of_freedom"], 15788 - rank)

    def test_all_interval_bounds_finite_and_ordered_and_match_csv(self):
        for name, prefix in (("baseline", "Baseline"), ("selected", "Selected")):
            bounds = self.bounds[name]
            for key, column in (("lower", f"{prefix} Lower 95"), ("upper", f"{prefix} Upper 95")):
                self.assertTrue(np.isfinite(bounds[key]).all())
                np.testing.assert_array_equal(bounds[key], self.predictions[column])
            self.assertTrue((bounds["lower"] <= bounds["prediction"]).all())
            self.assertTrue((bounds["prediction"] <= bounds["upper"]).all())
            self.assertTrue((bounds["upper"] > bounds["lower"]).all())

    def test_coverage_and_width_use_every_holdout_row(self):
        for name, bounds in self.bounds.items():
            measured = interval_metrics(self.predictions.Actual, bounds["lower"], bounds["upper"])
            for key, value in measured.items():
                np.testing.assert_allclose(self.interval_metrics["models"][name][key], value, rtol=1e-12)
            self.assertEqual(measured["total_holdout_samples"], 3947)
            self.assertEqual(measured["below_lower_count"] + measured["above_upper_count"] + measured["covered_count"], 3947)

    def test_comparison_csv_matches_both_json_metric_files(self):
        comparison = pd.read_csv(OUTPUT / "prediction_comparison.csv", float_precision="round_trip").set_index("Model")
        for name in self.models:
            row = comparison.loc[name]
            for label, key in (("MAE", "mae"), ("MSE", "mse"), ("RMSE", "rmse"), ("R2", "r2")):
                self.assertEqual(row[label], self.metrics["models"][name]["holdout_metrics"][key])
            for label, key in (("Empirical Coverage", "empirical_coverage"), ("Average Width", "average_width"),
                               ("Median Width", "median_width"), ("Width Standard Deviation", "width_standard_deviation")):
                self.assertEqual(row[label], self.interval_metrics["models"][name][key])

    def test_figures_reference_csv_hashes_fixed_window_and_actual_points(self):
        manifest = json.loads((OUTPUT / "figure_manifest.json").read_text())
        self.assertEqual(len(manifest["figures"]), 5)
        for name, digest in manifest["sources"].items():
            self.assertEqual(hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest(), digest)
        for name in ("baseline", "selected"):
            figure = manifest["figures"][f"{name}_prediction_interval.png"]
            self.assertEqual(figure["holdout_positions"], [0, 144])
            self.assertEqual(figure["row_indices"], [15788, 15931])
            for column, values in figure["points"].items():
                np.testing.assert_array_equal(values, self.predictions[column].iloc[:144])
        for name, metrics in manifest["figures"]["model_performance_comparison.png"]["metrics"].items():
            for label, key in (("MAE", "mae"), ("MSE", "mse"), ("RMSE", "rmse"), ("R2", "r2")):
                self.assertEqual(metrics[label], self.metrics["models"][name]["holdout_metrics"][key])
        for name, value in manifest["figures"]["interval_coverage_comparison.png"]["empirical_coverage"].items():
            self.assertEqual(value, self.interval_metrics["models"][name]["empirical_coverage"])
        for name, widths in manifest["figures"]["interval_width_comparison.png"]["widths"].items():
            for label, key in (("Average Width", "average_width"), ("Median Width", "median_width"),
                               ("Width Standard Deviation", "width_standard_deviation")):
                self.assertEqual(widths[label], self.interval_metrics["models"][name][key])

    def test_new_csv_files_are_utf8_lf_without_trailing_whitespace(self):
        for path in OUTPUT.glob("*.csv"):
            content = path.read_bytes()
            content.decode("utf-8")
            self.assertNotIn(b"\r", content)
            self.assertTrue(content.endswith(b"\n"))
            self.assertTrue(all(line.rstrip() == line for line in content.splitlines()))

    def test_no_fitting_guard_blocks_regression_and_selection_but_allows_prediction(self):
        model = self.models["selected"]
        coefficients = model.predictor.coef_.copy()
        with no_fitting():
            with self.assertRaisesRegex(RuntimeError, "forbids"):
                model.predictor.fit(np.zeros((8, 6)), np.zeros(8))
            with self.assertRaisesRegex(RuntimeError, "forbids"):
                model.artifact.selection.selector.fit(np.zeros((8, 25)), np.zeros(8))
            model.predict(self.holdout.iloc[:3])
        np.testing.assert_array_equal(coefficients, model.predictor.coef_)

    def test_saved_models_bytes_unchanged(self):
        for name, path in MODEL_FILES.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), MODEL_HASHES[name])

    def test_bounds_reproduce_without_holdout_target_or_refitting(self):
        for name, model in self.models.items():
            first = self.bounds[name]
            # 只重算已固定設計與保存 prediction 的區間；不做第二次選模／評估。
            again = interval_bounds(self.states[name], design_matrix(model.features(self.holdout)), first["prediction"].copy())
            np.testing.assert_array_equal(first["lower"], again["lower"])
            np.testing.assert_array_equal(first["upper"], again["upper"])


if __name__ == "__main__":
    unittest.main()
