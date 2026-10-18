"""Custom CSV nested selectors. Research selector contracts/results stay untouched."""
from time import perf_counter

import numpy as np
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold, TimeSeriesSplit

from .feature_selection import (FittedSelection, NumericFoldRegressor, fit_preprocessor,
                                fit_selection, validate_numeric_contract)

NESTED_METHODS = ('forward_sfs', 'rfecv', 'lasso_cv')
AUTO_METHODS = ('rfecv', 'lasso_cv')
ALPHAS = (10000., 1000., 100., 10., 1., .1, .01, .001, .0001)
INNER_SPLITS = 3
MAX_WORK = 100_000_000


class CustomSelectionError(ValueError):
    """Only fixed public messages from this module may be shown to clients."""


class EmptySelectionError(CustomSelectionError):
    pass


def inner_partitions(rows, config):
    cv = (TimeSeriesSplit(n_splits=INNER_SPLITS, gap=config.cv_gap)
          if config.split_strategy == 'chronological'
          else KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=config.random_state))
    try:
        result = list(cv.split(np.empty((rows, 0))))
    except ValueError as error:
        raise CustomSelectionError('內部 3-fold CV 無法切分；請增加資料、減少外層 folds 或 gap。') from error
    if min(len(t) for t, _ in result) < 4:
        raise CustomSelectionError('每個內部 CV Training 至少需要 4 筆；請增加資料、減少外層 folds 或 gap。')
    return result


def estimated_fits(method, k, p):
    if method == 'forward_sfs':
        return 1 if k == p else INNER_SPLITS*sum(p-j for j in range(k))+1
    if method == 'rfecv':
        return 1 if p == 1 else (INNER_SPLITS+1)*p+2
    if method == 'lasso_cv':
        return INNER_SPLITS*len(ALPHAS)+3
    return p-k+2 if method == 'rfe' and k < p else 1


def check_nested_request(method, k, config, validation, include_final=True):
    if method not in NESTED_METHODS:
        return
    if config.numerical_scaling != 'standard':
        raise CustomSelectionError('SFS／RFECV／LassoCV 需要 StandardScaler；請調整設定、確認並重新驗證。')
    p = len(config.feature_columns)
    if p > 25 or validation['dataset']['rows'] > 5000:
        raise CustomSelectionError('RFECV／LassoCV 限 5,000 筆、25 個候選欄；請縮小資料或候選欄位。')
    if method == 'forward_sfs' and (p > 15 or validation['dataset']['rows'] > 2000):
        raise CustomSelectionError('Forward SFS 較慢，限 2,000 筆、15 個候選欄；請縮小資料或候選欄位。')
    rows = [f['training_rows'] for f in validation['split']['cv_folds']]
    if include_final:
        rows.append(validation['split']['training_rows'])
    bypass = (method == 'forward_sfs' and k == p) or (method == 'rfecv' and p == 1)
    if not bypass:
        for n in rows:
            inner_partitions(n, config)  # Reject before any estimator/preprocessor fit.
    if estimated_fits(method, k, p)*p*sum(rows) > MAX_WORK:
        raise CustomSelectionError('選欄運算量過大；請減少資料／候選特徵／folds 或縮小比較範圍。')


def fit_custom_selection(X, y, config, spec):
    if spec.method not in NESTED_METHODS:
        return fit_selection(X, y, config, spec)
    started = perf_counter()
    values = validate_numeric_contract(X, y, config)
    p = values.shape[1]
    if config.numerical_scaling != 'standard':
        raise CustomSelectionError('選欄需要明確指定 StandardScaler。')
    bypass = (spec.method == 'forward_sfs' and spec.k == p) or (spec.method == 'rfecv' and p == 1)
    partitions = [] if bypass else inner_partitions(len(X), config)
    details = {'criterion': None if bypass else 'mse' if spec.method == 'lasso_cv' else 'rmse',
               'alpha': None, 'alpha_grid': list(ALPHAS) if spec.method == 'lasso_cv' else [],
               'inner_folds': [{'fold': i, 'training_rows': len(t), 'validation_rows': len(v)} for i, (t, v) in enumerate(partitions, 1)],
               'search_bypassed': bypass}
    processor = None
    selector = None
    if bypass:
        support = np.ones(p, dtype=bool)
    elif spec.method == 'forward_sfs':
        selector = SequentialFeatureSelector(NumericFoldRegressor(config), n_features_to_select=spec.k,
                                             direction='forward', scoring='neg_root_mean_squared_error', cv=partitions, n_jobs=1)
        selector.fit(values, y)
        support = selector.get_support()
    elif spec.method == 'rfecv':
        selector = RFECV(NumericFoldRegressor(config), step=1, min_features_to_select=1,
                         cv=partitions, scoring='neg_root_mean_squared_error', importance_getter='coef_', n_jobs=1)
        selector.fit(values, y)
        support = selector.get_support()
    else:
        # Each explicit CV index references only its independently prepared block.
        # Fixed alpha grid never derives information from validation/Holdout labels.
        blocks, targets, block_cv, offset = [], [], [], 0
        for train, valid in partitions:
            inner_processor, prepared = fit_preprocessor(X.iloc[train], y.iloc[train], config)
            transformed_valid = inner_processor.transform(X.iloc[valid])
            blocks.extend([prepared, transformed_valid])
            targets.extend([y.iloc[train].to_numpy(), y.iloc[valid].to_numpy()])
            block_cv.append((np.arange(offset, offset+len(train)), np.arange(offset+len(train), offset+len(train)+len(valid))))
            offset += len(train)+len(valid)
        selector = LassoCV(alphas=ALPHAS, cv=block_cv, random_state=config.random_state,
                           n_jobs=1, max_iter=10000, tol=1e-5, selection='cyclic')
        selector.fit(np.asfortranarray(np.vstack(blocks)), np.concatenate(targets))
        # Discard the library's mixed-block final refit. Refit support on this
        # outer Training's own scale; the downstream predictor is independent LR.
        processor, prepared = fit_preprocessor(X, y, config)
        chosen = Lasso(alpha=selector.alpha_, random_state=config.random_state, max_iter=10000, tol=1e-5).fit(prepared, y)
        support = np.abs(chosen.coef_) > 1e-8
        details['alpha'] = float(selector.alpha_)
    if processor is None:
        processor, _ = fit_preprocessor(X, y, config)
    indices = np.flatnonzero(support)
    metadata = {'method': spec.method, 'selected_k': len(indices), 'custom_details': details}
    return FittedSelection(processor, selector, indices, metadata,
                           {'selector_seconds': perf_counter()-started})
