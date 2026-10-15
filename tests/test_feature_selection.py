"""真實 sklearn 整合、mapping 與內層資料隔離測試。"""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV, SelectKBest, SequentialFeatureSelector
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from src.feature_selection import AUTO_METHODS, FIXED_METHODS, METHODS, NumericFoldRegressor
from src.feature_selection import SelectionSpec, ZeroFeaturesError, count_fits, fit_selected_regression, make_selector
from src.preprocessing import ColumnValues
from src.preprocessing_config import PreprocessingConfig


def fixture(n=72, p=4):
    random = np.random.default_rng(42)
    X = pd.DataFrame(random.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = pd.Series(5 * X.x0 - 2 * X.x1 + random.normal(scale=.2, size=n), name="y")
    config = PreprocessingConfig("y", list(X.columns), numerical_scaling="none")
    return X, y, config


class FeatureSelectionTests(unittest.TestCase):
    def test_six_real_sklearn_selectors_initialize(self):
        _, _, cfg = fixture()
        expected = [SelectKBest, SelectKBest, SequentialFeatureSelector, RFE, LassoCV, RFECV]
        for method, kind in zip(METHODS, expected):
            spec = SelectionSpec(method, 2 if method in FIXED_METHODS else None, gap=0)
            self.assertIsInstance(make_selector(spec, cfg), kind)

    def test_fixed_methods_support_k_one_middle_and_all(self):
        X, y, cfg = fixture()
        for method in FIXED_METHODS:
            for k in (1, 2, 4):
                with self.subTest(method=method, k=k):
                    model = fit_selected_regression(X, y, cfg, SelectionSpec(method, k, inner_splits=2, gap=0))
                    self.assertEqual(model.selection.metadata["selected_k"], k)
                    self.assertEqual(model.selection.transform(X).shape, (len(X), k))
                    self.assertIsInstance(model.predictor, LinearRegression)

    def test_k_25_boundary_does_not_search_all_subsets(self):
        X, y, cfg = fixture(n=48, p=25)
        for method in FIXED_METHODS:
            with self.subTest(method=method):
                model = fit_selected_regression(X, y, cfg, SelectionSpec(method, 25, gap=0))
                self.assertEqual(model.selection.indices.tolist(), list(range(25)))
                self.assertEqual(model.selection.metadata["search_bypassed"], "k_equals_p_all_features")

    def test_automatic_methods_report_support_without_forcing_k(self):
        X, y, cfg = fixture()
        for method in AUTO_METHODS:
            with self.subTest(method=method):
                model = fit_selected_regression(X, y, cfg, SelectionSpec(method, inner_splits=2, gap=0))
                meta = model.selection.metadata
                self.assertEqual(meta["selected_k"], len(meta["selected_feature_indices"]))
                self.assertTrue(0 <= meta["selected_k"] <= 4)
                self.assertIsNone(meta["requested_k"])
                self.assertIsInstance(model.predictor, LinearRegression)

    def test_selected_names_mapping_and_order_match_transformed_columns(self):
        X, y, cfg = fixture()
        model = fit_selected_regression(X, y, cfg, SelectionSpec("rfe", 2, gap=0))
        meta = model.selection.metadata
        names = list(model.selection.preprocessor.get_feature_names_out())
        self.assertEqual(meta["selected_feature_indices"], sorted(meta["selected_feature_indices"]))
        self.assertEqual(meta["selected_feature_names"], [names[i] for i in meta["selected_feature_indices"]])
        self.assertEqual(sum(map(len, meta["original_feature_mapping"].values())), 2)
        self.assertNotIn("y", meta["original_feature_mapping"])
        self.assertTrue(all("y" != name for name in meta["selected_feature_names"]))

    def test_target_in_features_or_input_is_rejected(self):
        X, y, cfg = fixture()
        with self.assertRaisesRegex(ValueError, "Target"):
            fit_selected_regression(X.assign(y=y), y, cfg, SelectionSpec("f_test", 1))

    def test_coefficient_methods_force_training_only_scaling(self):
        X, y, cfg = fixture()
        for method in ("rfe", "rfecv", "lasso_cv"):
            model = fit_selected_regression(X, y, cfg, SelectionSpec(method, 2 if method == "rfe" else None, inner_splits=2, gap=0))
            self.assertEqual(model.selection.metadata["scaling"], "standard")
            scaler = model.selection.preprocessor.named_steps["columns"].named_transformers_["c0"].named_steps["scale"]
            self.assertEqual(scaler.n_samples_seen_, len(X))
            self.assertAlmostEqual(scaler.mean_[0], X.x0.mean())

    def test_actual_selectors_accept_missing_numeric_with_fold_local_imputation(self):
        X, y, cfg = fixture()
        X.loc[3, "x0"] = np.nan
        for method in METHODS:
            with self.subTest(method=method):
                model = fit_selected_regression(X, y, cfg, SelectionSpec(method, 1 if method in FIXED_METHODS else None, inner_splits=2, gap=0))
                self.assertTrue(np.isfinite(model.predict(X.iloc[-3:])).all())

    def test_lasso_zero_features_skips_linear_regression(self):
        X, y, cfg = fixture()
        with count_fits() as counts:
            model = fit_selected_regression(X, y, cfg, SelectionSpec("lasso_cv", inner_splits=2, gap=0, alphas=(1e8,)))
        self.assertEqual(model.selection.metadata["selected_k"], 0)
        self.assertEqual(model.selection.metadata["status"], "no_features")
        self.assertIsNone(model.predictor)
        self.assertEqual(counts.get("downstream_linear_regression_fits", 0), 0)
        with self.assertRaises(ZeroFeaturesError):
            model.predict(X)

    def test_lassocv_paths_equal_manual_fold_local_lasso_cv(self):
        X, y, cfg = fixture()
        spec = SelectionSpec("lasso_cv", inner_splits=3, gap=1, alphas=(1., .1, .01), tolerance=1e-10)
        model = fit_selected_regression(X, y, cfg, spec)
        selector = model.selection.selector
        manual = np.empty_like(selector.mse_path_)
        for j, (train, valid) in enumerate(TimeSeriesSplit(n_splits=3, gap=1).split(X)):
            processor = selector.inner_preprocessors_[j]
            scaler = processor.named_steps["columns"].named_transformers_["c0"].named_steps["scale"]
            self.assertEqual(scaler.n_samples_seen_, len(train))
            self.assertAlmostEqual(scaler.mean_[0], X.iloc[train].x0.mean())
            prepared, validation = processor.transform(X.iloc[train]), processor.transform(X.iloc[valid])
            for i, alpha in enumerate(selector.alphas_):
                fitted = Lasso(alpha=alpha, max_iter=spec.max_iter, tol=spec.tolerance).fit(prepared, y.iloc[train])
                manual[i, j] = np.mean((y.iloc[valid].to_numpy() - fitted.predict(validation)) ** 2)
        np.testing.assert_allclose(selector.mse_path_, manual, rtol=1e-5, atol=1e-7)
        self.assertEqual(selector.alpha_, selector.alphas_[np.argmin(manual.mean(axis=1))])
        self.assertTrue(model.selection.metadata["discarded_lassocv_refit"])

    def test_sfs_and_rfecv_preprocessing_uses_each_internal_training_prefix(self):
        X, _, cfg = fixture(n=60)
        # 唯一 y 僅供監看當次 estimator 真正收到的列，仍執行真實 fit。
        y = pd.Series(np.arange(len(X), dtype=float), name="y")
        permitted = {tuple(train) for train, _ in TimeSeriesSplit(n_splits=3, gap=1).split(X)}
        permitted.add(tuple(range(len(X))))
        original = NumericFoldRegressor.fit
        for method in ("forward_sfs", "rfecv"):
            seen = []

            def observed(estimator, values, target):
                seen.append(tuple(np.asarray(target, dtype=int)))
                fitted = original(estimator, values, target)
                scaler = fitted.preprocessor_.named_steps["columns"].named_transformers_["c0"].named_steps["scale"]
                self.assertEqual(scaler.n_samples_seen_, len(target))
                self.assertAlmostEqual(scaler.mean_[0], np.asarray(values)[:, 0].mean())
                return fitted

            with patch.object(NumericFoldRegressor, "fit", autospec=True, side_effect=observed):
                fit_selected_regression(X, y, replace(cfg, numerical_scaling="standard"), SelectionSpec(method, 2 if method == "forward_sfs" else None, inner_splits=3, gap=1))
            self.assertTrue(seen)
            self.assertTrue(all(indices in permitted for indices in seen))
            self.assertTrue(permitted - {tuple(range(len(X)))} <= set(seen))

    def test_random_state_and_prediction_are_reproducible(self):
        X, y, cfg = fixture()
        spec = SelectionSpec("mutual_information", 2, gap=0)
        first = fit_selected_regression(X, y, cfg, spec)
        second = fit_selected_regression(X, y, cfg, spec)
        self.assertEqual(first.selection.metadata, second.selection.metadata)
        np.testing.assert_array_equal(first.predict(X), second.predict(X))

    def test_predict_does_not_refit_and_uses_independent_linear_regression(self):
        X, y, cfg = fixture()
        for method in METHODS:
            model = fit_selected_regression(X, y, cfg, SelectionSpec(method, 2 if method in FIXED_METHODS else None, inner_splits=2, gap=0))
            self.assertIsInstance(model.predictor, LinearRegression)
            with patch.object(ColumnValues, "fit", side_effect=AssertionError("predict must not fit")):
                np.testing.assert_array_equal(model.predict(X), model.predictor.predict(model.selection.transform(X)))

    def test_invalid_k_and_expanding_preprocessing_are_rejected(self):
        X, y, cfg = fixture()
        for k in (0, True, -1):
            with self.assertRaises(ValueError):
                SelectionSpec("rfe", k)
        with self.assertRaises(ValueError):
            fit_selected_regression(X, y, cfg, SelectionSpec("rfe", 5))
        for strategy in ("flag", "remove"):
            with self.assertRaisesRegex(ValueError, "一對一"):
                fit_selected_regression(X, y, replace(cfg, outlier_strategy=strategy), SelectionSpec("rfe", 1))
        with self.assertRaises(ValueError):
            SelectionSpec("lasso_cv", 3)


if __name__ == "__main__":
    unittest.main()
