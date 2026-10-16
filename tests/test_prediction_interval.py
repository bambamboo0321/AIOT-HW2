"""合成資料與 statsmodels 的 observation CI 比對；不重訓正式模型。"""

import inspect
import unittest

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from src.prediction_interval import design_matrix, estimate_interval, interval_bounds, interval_metrics

REFERENCE_RESULTS = {}


def fixture(rank_deficient=False):
    rng = np.random.default_rng(314)
    features = rng.normal(size=(90, 3))
    future = rng.normal(size=(11, 3))
    if rank_deficient:
        features[:, 2], future[:, 2] = features[:, 0], future[:, 0]
    actual = 2 + features @ np.array([3., -.6, 1.2]) + rng.normal(scale=.7, size=90)
    model = LinearRegression().fit(features, actual)
    return features, actual, future, model


class PredictionIntervalTests(unittest.TestCase):
    def check_reference(self, rank_deficient):
        features, actual, future, model = fixture(rank_deficient)
        design = design_matrix(features)
        state = estimate_interval(design, actual, model.predict(features))
        custom = interval_bounds(state, design_matrix(future), model.predict(future))
        reference = sm.OLS(actual, design).fit().get_prediction(design_matrix(future)).summary_frame(alpha=.05)
        errors = {}
        for ours, theirs in (("lower", "obs_ci_lower"), ("upper", "obs_ci_upper")):
            np.testing.assert_allclose(custom[ours], reference[theirs], rtol=1e-10, atol=1e-10)
            errors[ours + "_max_abs_error"] = float(np.max(np.abs(custom[ours] - reference[theirs])))
        self.assertEqual(state.diagnostics["rank"], int(np.linalg.matrix_rank(design)))
        REFERENCE_RESULTS["rank_deficient" if rank_deficient else "full_rank"] = {
            **errors, "rtol": 1e-10, "atol": 1e-10, "rank": state.diagnostics["rank"],
            "degrees_of_freedom": state.diagnostics["degrees_of_freedom"], "samples": len(actual)}
        return state

    def test_statsmodels_observation_interval_full_rank(self):
        self.check_reference(False)

    def test_statsmodels_observation_interval_rank_deficient(self):
        state = self.check_reference(True)
        self.assertTrue(state.diagnostics["rank_deficient"])

    def test_svd_leverage_matches_pseudoinverse_formula(self):
        features, y, future, model = fixture()
        design, new = design_matrix(features), design_matrix(future)
        state = estimate_interval(design, y, model.predict(features))
        result = interval_bounds(state, new, model.predict(future))
        gram_inverse = np.linalg.pinv(design.T @ design)
        direct = np.einsum("ij,jk,ik->i", new, gram_inverse, new)
        np.testing.assert_allclose(result["leverage"], direct, rtol=1e-11, atol=1e-12)

    def test_residual_sse_and_df_match_manual_values(self):
        features, y, _, model = fixture()
        design, predicted = design_matrix(features), model.predict(features)
        state = estimate_interval(design, y, predicted)
        self.assertEqual(state.diagnostics["degrees_of_freedom"], len(y) - np.linalg.matrix_rank(design))
        self.assertAlmostEqual(state.diagnostics["sse"], np.sum((y - predicted) ** 2), places=12)
        self.assertAlmostEqual(state.diagnostics["sigma_squared"], np.sum((y - predicted) ** 2) / 86, places=12)

    def test_nonpositive_df_is_rejected(self):
        design = design_matrix(np.array([[0.], [1.]]))
        with self.assertRaisesRegex(ValueError, "degrees of freedom"):
            estimate_interval(design, [1., 2.], [1., 2.])

    def test_zero_residual_variance_is_not_repaired_with_epsilon(self):
        design = design_matrix(np.arange(8.).reshape(-1, 1))
        with self.assertRaisesRegex(ValueError, "zero.*variance"):
            estimate_interval(design, np.arange(8.), np.arange(8.))

    def test_missing_intercept_is_rejected(self):
        X, y, _, model = fixture()
        with self.assertRaisesRegex(ValueError, "intercept"):
            estimate_interval(X, y, model.predict(X))

    def test_nonfinite_design_or_targets_are_rejected(self):
        X, y, _, model = fixture()
        predicted = model.predict(X)
        bad = design_matrix(X)
        bad[2, 1] = np.nan
        with self.assertRaises(ValueError):
            estimate_interval(bad, y, predicted)
        y[2] = np.inf
        with self.assertRaises(ValueError):
            estimate_interval(design_matrix(X), y, predicted)

    def test_mismatched_saved_predictions_fail_ols_orthogonality(self):
        X, y, _, model = fixture()
        with self.assertRaisesRegex(ValueError, "orthogonality"):
            estimate_interval(design_matrix(X), y, model.predict(X) + 3)

    def test_invalid_alpha_is_rejected(self):
        X, y, _, model = fixture()
        for alpha in (0., 1., -1., np.nan):
            with self.assertRaises(ValueError):
                estimate_interval(design_matrix(X), y, model.predict(X), alpha=alpha)

    def test_positive_bounds_and_more_leverage_widens_interval(self):
        X, y, _, model = fixture()
        state = estimate_interval(design_matrix(X), y, model.predict(X))
        new = np.array([[0., 0., 0.], [100., 100., 100.]])
        result = interval_bounds(state, design_matrix(new), model.predict(new))
        self.assertTrue((result["lower"] <= result["prediction"]).all())
        self.assertTrue((result["prediction"] <= result["upper"]).all())
        self.assertGreater(np.ptp([result["lower"][1], result["upper"][1]]), np.ptp([result["lower"][0], result["upper"][0]]))

    def test_prediction_interval_includes_observation_noise(self):
        X, y, future, model = fixture()
        state = estimate_interval(design_matrix(X), y, model.predict(X))
        result = interval_bounds(state, design_matrix(future), model.predict(future))
        mean_ci_radius = state.diagnostics["t_critical"] * state.diagnostics["sigma"] * np.sqrt(result["leverage"])
        self.assertTrue(((result["upper"] - result["lower"]) / 2 > mean_ci_radius).all())

    def test_rank_deficient_nonestimable_future_row_is_rejected(self):
        X, y, future, model = fixture(True)
        state = estimate_interval(design_matrix(X), y, model.predict(X))
        future[0, 2] += 1.
        with self.assertRaisesRegex(ValueError, "row space"):
            interval_bounds(state, design_matrix(future), model.predict(future))

    def test_coverage_is_inclusive_and_counts_and_widths_are_correct(self):
        result = interval_metrics([-1., 0., 2., 3.], np.zeros(4), np.full(4, 2.))
        self.assertEqual(result["empirical_coverage"], .5)
        self.assertEqual((result["below_lower_count"], result["above_upper_count"], result["covered_count"]), (1, 1, 2))
        self.assertEqual(result["average_width"], 2.)
        self.assertEqual(result["median_width"], 2.)
        self.assertEqual(result["width_standard_deviation"], 0.)

    def test_estimation_api_has_no_holdout_or_coverage_argument(self):
        self.assertEqual(list(inspect.signature(estimate_interval).parameters),
                         ["training_design", "training_actual", "training_prediction", "alpha"])

    def test_leverage_trace_and_reconstruction_confirm_stable_rank(self):
        X, y, _, model = fixture()
        state = estimate_interval(design_matrix(X), y, model.predict(X))
        self.assertAlmostEqual(state.diagnostics["training_leverage_sum"], state.diagnostics["rank"], places=12)
        self.assertLess(state.diagnostics["svd_reconstruction_relative_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
