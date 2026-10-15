"""Phase 3B：在同一 training partition 重用確定性的選擇路徑。

不跨 fold 分享已學習的規則；底層選擇與前處理重用 Phase 3A／2。
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from .feature_selection import (
    FittedSelection, NumericFoldRegressor, SelectedRegression, SelectionSpec,
    fit_preprocessor, fit_selected_regression, fit_selection, method_config,
    validate_numeric_contract, increment,
)
from .preprocessing import feature_summary

SFS_GRID = (1, 3, 5, 8, 10, 13, 16, 20, 25)
FULL_GRID = tuple(range(1, 26))


class RecordingSFS(SequentialFeatureSelector):
    """僅記錄 sklearn 已選擇的 prefix；不重寫搜尋或評分演算法。

    固定 forward、tol=None 時，較小 k 是同一 greedy path 的 prefix。
    sklearn 版本與直接逐 k 執行的等價性由測試固定。
    """

    def fit(self, X, y=None, **params):
        self.selection_path_ = []
        return super().fit(X, y, **params)

    def _get_best_new_feature_score(self, estimator, X, y, cv, current_mask, **params):
        feature, score = super()._get_best_new_feature_score(
            estimator, X, y, cv, current_mask, **params)
        self.selection_path_.append(int(feature))
        return feature, score


def fit_group(X, y, config, method, grid, *, max_iter=100000, tolerance=1e-5):
    """單一 training fold 的完整預定 grid；回傳獨立 LR 與共享耗時。

    Filter scores、RFE elimination path、SFS greedy prefix 僅在本 fold 重用。
    不接受 validation／holdout；auto 方法沿用 Phase 3A 實作。
    """
    grid = tuple(grid)
    if method in {"lasso_cv", "rfecv"}:
        if grid != (None,):
            raise ValueError("auto 方法的 grid 必須為 (None,)")
        started = perf_counter()
        model = fit_selected_regression(
            X, y, config, SelectionSpec(method, max_iter=max_iter, tolerance=tolerance))
        return {None: model}, {"shared_fit_seconds": perf_counter() - started - model.predictor_fit_seconds,
                              "downstream_seconds": model.predictor_fit_seconds,
                              "grid": [None], "path_reuse": False}
    p = len(config.feature_columns)
    if not grid or any(type(k) is not int or not 1 <= k <= p for k in grid) or len(set(grid)) != len(grid):
        raise ValueError("grid 必須是互異且合法的固定 k")
    started = perf_counter()
    cfg = method_config(config, method)
    values = validate_numeric_contract(X, y, cfg)
    searched = [k for k in grid if k < p]
    if method == "forward_sfs" and searched:
        selector = RecordingSFS(
            NumericFoldRegressor(cfg), n_features_to_select=max(searched),
            direction="forward", tol=None, scoring="neg_root_mean_squared_error",
            cv=TimeSeriesSplit(n_splits=3, gap=6), n_jobs=1).fit(values, y)
        processor, prepared = fit_preprocessor(X, y, cfg)
        base = None
    else:
        base = fit_selection(X, y, cfg, SelectionSpec(method, 1 if method == "rfe" and searched else min(grid)))
        processor, selector = base.preprocessor, base.selector
        prepared = processor.transform(X)
    summary = feature_summary(processor, cfg)
    shared = perf_counter() - started
    models, downstream = {}, 0.
    for k in grid:
        if k == p:
            indices = np.arange(p)
        elif method == "forward_sfs":
            indices = np.sort(selector.selection_path_[:k])
        elif method == "rfe":
            indices = np.flatnonzero(selector.ranking_ <= k)
        else:
            indices = np.flatnonzero(deepcopy(selector).set_params(k=k).get_support())
        names = [summary["feature_names"][i] for i in indices]
        metadata = {"method": method, "requested_k": k, "selected_k": len(indices),
                    "selected_feature_indices": indices.tolist(), "selected_feature_names": names,
                    "original_feature_mapping": {
                        name: [n for n in transformed if n in names]
                        for name, transformed in summary["original_to_transformed"].items()},
                    "scaling": cfg.numerical_scaling, "status": "ok",
                    "path_reuse_within_training_partition": True,
                    "search_bypassed": "k_equals_p_all_features" if k == p else None}
        # selector 保存完整共享路徑；真正轉換由明確的 indices 決定。
        selected = FittedSelection(processor, selector, indices, metadata,
                                   {"shared_selection_seconds": shared})
        clock = perf_counter()
        predictor = LinearRegression().fit(prepared[:, indices], y)
        increment("downstream_linear_regression_fits")
        seconds = perf_counter() - clock
        downstream += seconds
        models[k] = SelectedRegression(selected, predictor, seconds)
    return models, {"shared_fit_seconds": shared, "downstream_seconds": downstream,
                    "grid": list(grid), "path_reuse": True}
