"""Phase 3A：六種 Selector、fold-local 前處理與獨立 LinearRegression。

本版只接受一對一 numeric 特徵；不會修改 Phase 2 模組或既有模型。
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import RFE, RFECV, SelectKBest, SequentialFeatureSelector
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from .data_splitting import assert_aligned
from .preprocessing import build_preprocessor, feature_summary

FIXED_METHODS = ("f_test", "mutual_information", "forward_sfs", "rfe")
AUTO_METHODS = ("lasso_cv", "rfecv")
METHODS = FIXED_METHODS + AUTO_METHODS
_COUNTS = ContextVar("selection_fit_counts", default=None)


@contextmanager
def count_fits():
    counts = {}
    token = _COUNTS.set(counts)
    try:
        yield counts
    finally:
        _COUNTS.reset(token)


def increment(key, value=1):
    counts = _COUNTS.get()
    if counts is not None:
        counts[key] = counts.get(key, 0) + value


@dataclass(frozen=True)
class SelectionSpec:
    method: str
    k: int | None = None
    inner_splits: int = 3
    gap: int = 6
    random_state: int = 42
    alphas: tuple = (100., 10., 1., .1, .01, .001)
    coefficient_tolerance: float = 1e-8
    max_iter: int = 10000
    tolerance: float = 1e-5

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"未知 Selector：{self.method}")
        if self.method in FIXED_METHODS and (type(self.k) is not int or self.k < 1):
            raise ValueError("固定 k 方法必須指定正整數 k。")
        if self.method in AUTO_METHODS and self.k is not None:
            raise ValueError("自動 k 方法不得強制指定 k。")
        if type(self.inner_splits) is not int or self.inner_splits < 2:
            raise ValueError("inner_splits 必須 >=2。")
        if type(self.gap) is not int or self.gap < 0:
            raise ValueError("gap 必須為非負整數。")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state 無效。")
        if not self.alphas or any(not np.isfinite(a) or a <= 0 for a in self.alphas):
            raise ValueError("alphas 必須為事先指定的有限正數，不能由完整資料推算。")
        if not np.isfinite(self.coefficient_tolerance) or self.coefficient_tolerance < 0:
            raise ValueError("coefficient_tolerance 必須為非負有限數。")
        if type(self.max_iter) is not int or self.max_iter < 1 or not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Lasso 迭代設定無效。")


def method_config(config, method):
    return replace(config, numerical_scaling="standard") if method in {"rfe", "rfecv", "lasso_cv"} else config


def validate_numeric_contract(X, y, config):
    assert_aligned(X, y)
    if config.target_column in X or config.target_column in config.feature_columns:
        raise ValueError("Features 不得包含 Target。")
    if config.outlier_strategy not in {"keep", "clip"} or config.missing_numeric not in {"mean", "median", "none"}:
        raise ValueError("Phase 3A 僅支援一對一 numeric 轉換；不支援 flag／刪列。")
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Target 必須是有限且非缺失的數值。")
    for name in config.feature_columns:
        if (name not in X or config.column_types.get(name, "numeric") != "numeric"
                or not pd.api.types.is_numeric_dtype(X[name]) or pd.api.types.is_bool_dtype(X[name])
                or pd.api.types.is_complex_dtype(X[name])):
            raise ValueError(f"{name}: Phase 3A 只接受一對一 numeric Features；Encoding／日期展開尚未支援。")
    values = X[config.feature_columns].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Feature 不接受 Infinity。")
    return values


def fit_preprocessor(X, y, config):
    increment("preprocessor_fits")
    processor = build_preprocessor(X, config)
    transformed = processor.fit_transform(X, y)
    if transformed.shape[1] != len(config.feature_columns):
        raise ValueError("本版 Selector 要求轉換後欄位與原始 numeric 欄位一對一。")
    return processor, np.asarray(transformed, dtype=float)


class NumericFoldRegressor(RegressorMixin, BaseEstimator):
    """供 sklearn SFS/RFE/RFECV clone 的 estimator；每次 fit 重用 Phase 2。

    sklearn 內部刪欄會轉為 ndarray，因此臨時欄名只代表當次欄順序；
    最終 mapping 一律由外層的原始 schema 建立。
    """

    def __init__(self, config):
        self.config = config

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        columns = [f"__candidate_{i}__" for i in range(self.n_features_in_)]
        cfg = replace(self.config, feature_columns=columns, column_types={c: "numeric" for c in columns})
        frame = pd.DataFrame(values, columns=columns)
        self.preprocessor_, prepared = fit_preprocessor(frame, y, cfg)
        increment("selector_linear_regression_fits")
        self.regressor_ = LinearRegression().fit(prepared, y)
        self.coef_ = self.regressor_.coef_
        return self

    def predict(self, X):
        frame = pd.DataFrame(np.asarray(X, dtype=float), columns=[f"__candidate_{i}__" for i in range(self.n_features_in_)])
        return self.regressor_.predict(self.preprocessor_.transform(frame))


def make_selector(spec, config):
    """建立真實 sklearn Selector；LassoCV 的 CV blocks 於 fit 時組裝。"""
    cv = TimeSeriesSplit(n_splits=spec.inner_splits, gap=spec.gap)
    estimator = NumericFoldRegressor(method_config(config, spec.method))
    if spec.method == "f_test":
        return SelectKBest(f_regression, k=spec.k)
    if spec.method == "mutual_information":
        return SelectKBest(partial(mutual_info_regression, random_state=spec.random_state, n_jobs=1), k=spec.k)
    if spec.method == "forward_sfs":
        return SequentialFeatureSelector(estimator, n_features_to_select=spec.k, direction="forward",
                                         scoring="neg_root_mean_squared_error", cv=cv, n_jobs=1)
    if spec.method == "rfe":
        return RFE(estimator, n_features_to_select=spec.k, step=1, importance_getter="coef_")
    if spec.method == "rfecv":
        return RFECV(estimator, step=1, min_features_to_select=1, cv=cv,
                     scoring="neg_root_mean_squared_error", importance_getter="coef_", n_jobs=1)
    return LassoCV(alphas=spec.alphas, cv=cv, random_state=spec.random_state, n_jobs=1,
                   max_iter=spec.max_iter, tol=spec.tolerance, selection="cyclic", copy_X=True)


def fold_local_lasso_blocks(X, y, config, spec):
    """LassoCV 每個 split 僅索引自己的 train/valid block，不跨 block 借用規則。"""
    blocks, targets, splits, processors, manifest = [], [], [], [], []
    offset = 0
    for train, valid in TimeSeriesSplit(n_splits=spec.inner_splits, gap=spec.gap).split(X):
        processor, transformed = fit_preprocessor(X.iloc[train], y.iloc[train], config)
        validation = processor.transform(X.iloc[valid])
        blocks.extend([transformed, validation])
        targets.extend([y.iloc[train].to_numpy(), y.iloc[valid].to_numpy()])
        splits.append((np.arange(offset, offset + len(train)), np.arange(offset + len(train), offset + len(train) + len(valid))))
        offset += len(train) + len(valid)
        processors.append(processor)
        manifest.append({"train_indices": list(X.index[train]), "validation_indices": list(X.index[valid])})
    return np.asfortranarray(np.vstack(blocks)), np.concatenate(targets), splits, processors, manifest


@dataclass
class FittedSelection:
    preprocessor: object
    selector: object
    indices: np.ndarray
    metadata: dict
    timing: dict

    def transform(self, X):
        return self.preprocessor.transform(X)[:, self.indices]


class ZeroFeaturesError(ValueError):
    pass


def fit_selection(X, y, config, spec, audit=None, scope="training"):
    """只有 X/y training 進入本函式；不接收 validation 或 Holdout。"""
    started = perf_counter()
    cfg = method_config(config, spec.method)
    values = validate_numeric_contract(X, y, cfg)
    p = values.shape[1]
    if spec.k is not None and spec.k > p:
        raise ValueError(f"k={spec.k} 超過轉換後 p={p}。")
    if audit is not None:
        audit.append({"event": "selector_fit", "scope": scope, "method": spec.method, "fit_indices": list(X.index)})
    selector = make_selector(spec, cfg)
    processor = None
    outer_preprocessing_seconds = 0.
    extra = {}
    if spec.method in FIXED_METHODS and spec.k == p:
        # SFS 不接受 k=p；所有欄位已選時不需要搜尋，也不冒充已執行搜尋。
        support = np.ones(p, dtype=bool)
        extra["search_bypassed"] = "k_equals_p_all_features"
    elif spec.method in {"f_test", "mutual_information"}:
        clock = perf_counter()
        processor, prepared = fit_preprocessor(X, y, cfg)
        outer_preprocessing_seconds += perf_counter() - clock
        increment("filter_score_calls")
        selector.fit(prepared, y)
        support = selector.get_support()
    elif spec.method == "lasso_cv":
        blocks, target_blocks, cv, processors, manifest = fold_local_lasso_blocks(X, y, cfg, spec)
        selector.set_params(cv=cv)
        increment("lasso_cv_fits")
        increment("lasso_cv_fold_paths", len(cv))
        increment("lasso_cv_alpha_path_points_estimate", len(cv) * len(spec.alphas))
        increment("lasso_cv_discarded_refits")
        selector.fit(blocks, target_blocks)
        # LassoCV 在拼接資料上的內建 refit 係數具有混合尺度，完全不使用。
        selector.inner_preprocessors_ = processors
        clock = perf_counter()
        processor, prepared = fit_preprocessor(X, y, cfg)
        outer_preprocessing_seconds += perf_counter() - clock
        increment("lasso_support_refits")
        support_model = Lasso(alpha=selector.alpha_, max_iter=spec.max_iter, tol=spec.tolerance,
                              random_state=spec.random_state, selection="cyclic").fit(prepared, y)
        support = np.abs(support_model.coef_) > spec.coefficient_tolerance
        extra.update({"alpha": float(selector.alpha_), "alpha_grid": list(spec.alphas),
                      "support_coefficients": list(map(float, support_model.coef_)),
                      "inner_cv_manifest": manifest, "stacked_rows": len(blocks),
                      "discarded_lassocv_refit": True,
                      "zero_feature_policy": "report_no_features_and_skip_predictor; no forced fallback"})
    elif spec.method == "rfecv" and p == 1:
        support = np.ones(1, dtype=bool)
        extra["search_bypassed"] = "one_available_feature"
    else:
        if spec.method in {"forward_sfs", "rfecv"}:
            extra["inner_cv_manifest"] = [
                {"train_indices": list(X.index[train]), "validation_indices": list(X.index[valid])}
                for train, valid in TimeSeriesSplit(n_splits=spec.inner_splits, gap=spec.gap).split(X)]
        selector.fit(values, y)
        support = selector.get_support()
    if processor is None:
        clock = perf_counter()
        processor, _ = fit_preprocessor(X, y, cfg)
        outer_preprocessing_seconds += perf_counter() - clock
    summary = feature_summary(processor, cfg)
    indices = np.flatnonzero(support)
    names = [summary["feature_names"][i] for i in indices]
    mapping = {name: [n for n in transformed if n in names]
               for name, transformed in summary["original_to_transformed"].items()}
    metadata = {"method": spec.method, "requested_k": spec.k, "selected_k": len(indices),
                "selected_feature_indices": indices.tolist(), "selected_feature_names": names,
                "original_feature_mapping": mapping, "available_feature_names": summary["feature_names"],
                "scaling": cfg.numerical_scaling, "fit_indices": list(X.index),
                "k_definition": "preprocessed numeric columns passed to independent LinearRegression",
                "status": "ok" if len(indices) else "no_features", **extra}
    total = perf_counter() - started
    return FittedSelection(processor, selector, indices, metadata,
                           {"selector_seconds": max(0., total - outer_preprocessing_seconds),
                            "outer_preprocessing_seconds": outer_preprocessing_seconds})


@dataclass
class SelectedRegression:
    selection: FittedSelection
    predictor: object
    predictor_fit_seconds: float

    def predict(self, X):
        if self.predictor is None:
            raise ZeroFeaturesError("Selector 選出 0 Features；依預定規則跳過 LinearRegression。")
        return self.predictor.predict(self.selection.transform(X))


def fit_selected_regression(X, y, config, spec, audit=None, scope="training"):
    selected = fit_selection(X, y, config, spec, audit, scope)
    predictor = None
    started = perf_counter()
    if selected.metadata["selected_k"]:
        if audit is not None:
            audit.append({"event": "predictor_fit", "scope": scope, "fit_indices": list(X.index)})
        increment("downstream_linear_regression_fits")
        predictor = LinearRegression().fit(selected.transform(X), y)
    return SelectedRegression(selected, predictor, perf_counter() - started)
