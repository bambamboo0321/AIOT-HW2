"""Phase 2 驗收：時間邊界、held-out Target 隔離、模型與逐筆結果一致性。"""

import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.phase2_common import ROOT, BASELINE_FEATURES, load_dataset, prepare_features
from src.train_baseline import create_splits, fit_baseline


class Phase2Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_dataset()
        cls.features, cls.target = prepare_features(cls.frame)
        cls.splits = create_splits(cls.frame)

    def test_predefined_features_exclude_target_time_and_random_columns(self):
        self.assertEqual(self.features.shape, (19735, 25))
        self.assertEqual(list(self.features.columns), BASELINE_FEATURES)
        self.assertFalse({"Appliances", "date", "rv1", "rv2"}.intersection(self.features.columns))
        pd.testing.assert_index_equal(self.features.index, self.target.index)

    def test_time_holdout_is_after_training_and_partitions_cover_dataset(self):
        for train, test in self.splits.values():
            self.assertEqual((len(train), len(test)), (15788, 3947))
            self.assertFalse(set(train).intersection(test))
            self.assertEqual(set(train).union(test), set(self.frame.index))
        train, test = self.splits["chronological"]
        np.testing.assert_array_equal(train, np.arange(15788))
        np.testing.assert_array_equal(test, np.arange(15788, 19735))
        self.assertLess(self.frame.loc[train, "date"].max(), self.frame.loc[test, "date"].min())

    def test_random_split_repeats_and_manifest_matches_fitted_partitions(self):
        repeated = create_splits(self.frame)
        manifest = json.loads((ROOT / "reports/phase2/split_indices.json").read_text())
        for name, (train, test) in self.splits.items():
            np.testing.assert_array_equal(train, repeated[name][0])
            np.testing.assert_array_equal(test, repeated[name][1])
            self.assertEqual(manifest[name]["train"], train.tolist())
            self.assertEqual(manifest[name]["test"], test.tolist())

    def test_heldout_target_cannot_change_fitted_model(self):
        original_fit = LinearRegression.fit
        for name, (train, test) in self.splits.items():
            with self.subTest(split=name):
                altered_target = self.target.copy()
                altered_target.loc[test] = altered_target.loc[test] + 100000

                def fit_only_expected_training(model, x, y):
                    np.testing.assert_array_equal(x.index, train)
                    np.testing.assert_array_equal(y.index, train)
                    self.assertFalse(set(x.index).intersection(test))
                    self.assertEqual(list(x.columns), BASELINE_FEATURES)
                    return original_fit(model, x, y)

                with patch.object(LinearRegression, "fit", autospec=True, side_effect=fit_only_expected_training) as fitting:
                    first, _, predicted = fit_baseline(self.features, self.target, train, test)
                    second, _, altered_predicted = fit_baseline(self.features, altered_target, train, test)
                    self.assertEqual(fitting.call_count, 2)
                np.testing.assert_array_equal(first.coef_, second.coef_)
                self.assertEqual(first.intercept_, second.intercept_)
                np.testing.assert_array_equal(predicted, altered_predicted)

    def test_saved_models_predictions_residuals_and_metrics_agree(self):
        metrics = json.loads((ROOT / "reports/phase2/baseline_metrics.json").read_text())
        for name, (train, test) in self.splits.items():
            with self.subTest(split=name):
                artifact = joblib.load(ROOT / f"reports/phase2/models/{name}_baseline.joblib")
                model = artifact["model"]
                self.assertEqual(model.get_params(), LinearRegression().get_params())
                self.assertEqual(artifact["feature_names"], BASELINE_FEATURES)
                self.assertEqual(list(model.feature_names_in_), BASELINE_FEATURES)
                predictions = pd.read_csv(ROOT / f"reports/phase2/predictions_{name}.csv")
                self.assertEqual(set(predictions["row_index"]), set(test))
                self.assertEqual(len(predictions), len(test))
                predicted = model.predict(self.features.loc[predictions["row_index"]])
                np.testing.assert_allclose(predicted, predictions["predicted_wh"], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(self.target.loc[predictions["row_index"]], predictions["actual_wh"], rtol=0, atol=0)
                np.testing.assert_allclose(predictions["actual_wh"] - predictions["predicted_wh"], predictions["residual_wh"], rtol=1e-12, atol=1e-12)
                for partition, indices in (("train", train), ("test", test)):
                    actual = self.target.loc[indices].to_numpy()
                    estimate = model.predict(self.features.loc[indices])
                    errors = actual - estimate
                    mse = np.mean(errors ** 2)
                    independent = {"mae": np.mean(np.abs(errors)), "mse": mse,
                                   "rmse": np.sqrt(mse),
                                   "r2": 1 - np.sum(errors ** 2) / np.sum((actual - actual.mean()) ** 2)}
                    for key, value in independent.items():
                        self.assertTrue(np.isfinite(value))
                        self.assertAlmostEqual(value, metrics["baselines"][name][f"{partition}_metrics"][key], places=9)

    def test_changed_source_checksum_is_rejected(self):
        with patch("src.phase2_common.sha256_file", return_value="unexpected"):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_dataset()


if __name__ == "__main__":
    unittest.main()
