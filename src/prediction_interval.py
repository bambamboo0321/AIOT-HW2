"""以 Training-only SVD 為已保存 OLS 建立單筆觀測值 Prediction Interval。

不估計或修改 regression coefficients；不接收 Holdout Target。
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import t


def finite_vector(values, length, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != length or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array


def design_matrix(features):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("features must be a finite, nonempty 2-D matrix")
    return np.column_stack((np.ones(len(values)), values))


def checked_design(design):
    values = np.asarray(design, dtype=np.float64)
    if (values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all()
            or not np.array_equal(values[:, 0], np.ones(len(values)))):
        raise ValueError("design must be finite and contain a leading intercept column of ones")
    return values


@dataclass(frozen=True)
class OLSIntervalState:
    right_basis: np.ndarray
    singular_values_kept: np.ndarray
    diagnostics: dict


def estimate_interval(training_design, training_actual, training_prediction, *, alpha=.05):
    """Training 殘差估 sigma；SVD cutoff 固定為 eps*max(n,p)*s_max。

    與 numpy matrix_rank 預設相同的有效秩定義。零變異直接拒絕，不補任意 epsilon。
    """
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    X = checked_design(training_design)
    n, p = X.shape
    actual = finite_vector(training_actual, n, "training_actual")
    prediction = finite_vector(training_prediction, n, "training_prediction")
    U, singular, Vt = np.linalg.svd(X, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(X.shape) * singular[0]
    keep = singular > cutoff
    rank = int(keep.sum())
    df = n - rank
    if df <= 0:
        raise ValueError("residual degrees of freedom must be positive")
    residual = actual - prediction
    sse = float(residual @ residual)
    variance = sse / df
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("zero or nonfinite training residual variance; positive-width PI is undefined")
    # 已保存預測須是這個 training design 的 OLS 結果；不另 fit 係數。
    normalized_score = float(np.linalg.norm(X.T @ residual) /
                             (singular[0] * max(np.linalg.norm(residual), np.finfo(float).tiny)))
    if normalized_score > 1e-8:
        raise ValueError("saved predictions fail OLS residual orthogonality in this design space")
    critical = float(t.ppf(1 - alpha / 2, df))
    basis = Vt[keep]
    reconstruction = (U[:, keep] * singular[keep]) @ basis
    reconstruction_error = float(np.linalg.norm(X - reconstruction) / np.linalg.norm(X))
    training_leverage = np.sum(U[:, keep] ** 2, axis=1)
    diagnostics = {
        "method": "SVD of training design; no inversion of X.T@X",
        "alpha": alpha, "nominal_coverage": 1 - alpha,
        "training_rows": n, "design_columns_including_intercept": p,
        "rank": rank, "degrees_of_freedom": df, "rank_deficient": rank < p,
        "rank_threshold": float(cutoff), "rank_policy": "eps(float64)*max(n,p)*s_max",
        "singular_values": singular.tolist(),
        "effective_condition_number": float(singular[0] / singular[keep][-1]),
        "sse": sse, "sigma_squared": variance, "sigma": float(np.sqrt(variance)),
        "t_critical": critical, "residual_mean": float(residual.mean()),
        "normalized_residual_orthogonality_error": normalized_score,
        "svd_reconstruction_relative_error": reconstruction_error,
        "training_leverage_sum": float(training_leverage.sum()),
        "training_leverage_min": float(training_leverage.min()),
        "training_leverage_max": float(training_leverage.max()),
        "holdout_residuals_used": False, "bounds_clipped_to_zero": False,
        "interpretation": "pointwise observation prediction interval, not a mean confidence interval or simultaneous band",
    }
    return OLSIntervalState(basis.copy(), singular[keep].copy(), diagnostics)


def interval_bounds(state, design, prediction):
    X = checked_design(design)
    if X.shape[1] != state.diagnostics["design_columns_including_intercept"]:
        raise ValueError("prediction design columns differ from training")
    estimate = finite_vector(prediction, len(X), "prediction")
    projected = X @ state.right_basis.T
    if state.diagnostics["rank_deficient"]:
        error = np.linalg.norm(X - projected @ state.right_basis, axis=1)
        relative = error / np.maximum(1., np.linalg.norm(X, axis=1))
        if np.any(relative > 1e-10):
            raise ValueError("rank-deficient training: new row is outside the estimable training row space")
    leverage = np.sum((projected / state.singular_values_kept) ** 2, axis=1)
    radius = (state.diagnostics["t_critical"] * state.diagnostics["sigma"] *
              np.sqrt(1 + leverage))
    lower, upper = estimate - radius, estimate + radius
    if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or not (lower <= estimate).all() or not (estimate <= upper).all()
            or not (upper > lower).all()):
        raise ValueError("prediction interval bounds must be finite, ordered and positive-width")
    return {"prediction": estimate, "lower": lower, "upper": upper, "leverage": leverage}


def interval_metrics(actual, lower, upper, *, nominal=.95):
    low = np.asarray(lower, dtype=float)
    if low.ndim != 1 or not len(low):
        raise ValueError("interval must be a nonempty vector")
    low = finite_vector(low, len(low), "lower")
    high = finite_vector(upper, len(low), "upper")
    y = finite_vector(actual, len(low), "actual")
    if not (high > low).all() or not 0 < nominal < 1:
        raise ValueError("positive widths and valid nominal coverage required")
    widths = high - low
    below, above = int((y < low).sum()), int((y > high).sum())
    return {"nominal_coverage": nominal,
            "empirical_coverage": float(((low <= y) & (y <= high)).mean()),
            "average_width": float(widths.mean()), "median_width": float(np.median(widths)),
            "width_standard_deviation": float(widths.std(ddof=0)), "width_std_ddof": 0,
            "below_lower_count": below, "above_upper_count": above,
            "covered_count": len(y) - below - above, "total_holdout_samples": len(y)}
