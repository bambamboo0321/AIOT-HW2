"""切分與刪列的明確邊界：維持 X/y 索引、fold 內 fit、訓練區內 CV。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split

from .preprocessing import ColumnValues, IQRTransformer, build_preprocessor, feature_summary, resolve_types
from .preprocessing_config import numeric_target
from .profiling import parse_datetime


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    dropped_target_indices: list


@dataclass
class PreparedFold:
    preprocessor: object
    X_train: object
    X_validation: object
    y_train: pd.Series
    y_validation: pd.Series
    summary: dict
    row_audit: dict
    removal_bounds: dict


def assert_aligned(X, y):
    if not X.index.is_unique or not y.index.is_unique or not X.index.equals(y.index):
        raise ValueError("X/y 必須具有相同且唯一的索引與順序；CSV 請保留唯一 row index。")
    if len(X) == 0:
        raise ValueError("資料列為空。")


def chronological_dates(X, config):
    if config.split_column not in X:
        raise ValueError("時間切分缺少 split_column；即使日期不作 Feature 也必須保留。")
    dates = parse_datetime(X[config.split_column], config.datetime_formats.get(config.split_column))
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("時間切分需要非缺失且唯一的時間戳；重複時間需後續實作群組切分。")
    return dates


def split_dataset(frame, config):
    """僅 deterministic Target 驗證與缺失列排除；不 fit 任何 Feature 轉換。"""
    config.validate_columns(frame)
    if not frame.index.is_unique:
        raise ValueError("資料索引必須唯一，避免 X/y 切分歧義。")
    target = numeric_target(frame[config.target_column])
    missing = target.isna()
    if missing.any() and config.missing_target == "error":
        raise ValueError("Target 含缺失值；設定 missing_target=drop 才可排除，不能補值。")
    dropped = list(frame.index[missing])
    X = frame.loc[~missing].drop(columns=config.target_column).copy()
    y = target.loc[X.index].copy()
    assert_aligned(X, y)
    positions = np.arange(len(X))
    if config.split_strategy == "chronological":
        dates = chronological_dates(X, config)
        positions = np.argsort(dates.to_numpy(), kind="stable")
        boundary = int(len(X) * (1 - config.test_size))
        train, test = positions[:boundary], positions[boundary:]
    else:
        train, test = train_test_split(positions, test_size=config.test_size, random_state=config.random_state)
    if not len(train) or not len(test):
        raise ValueError("切分後 Training 或 Test 為空；請調整 test_size。")
    return DatasetSplit(X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test], dropped)


def _missing_filter(X, y, config, types):
    assert_aligned(X, y)
    keep = np.ones(len(X), dtype=bool)
    for name, kind in types.items():
        active = not (kind == "exclude" or (kind == "categorical" and config.categorical_encoding == "drop")
                      or (kind == "datetime" and config.datetime_strategy == "exclude"))
        if not active:
            continue
        policy = config.missing_categorical if kind == "categorical" else config.missing_numeric
        if policy == "drop":
            keep &= X[name].notna().to_numpy()
    return X.loc[keep].copy(), y.loc[keep].copy(), list(X.index[~keep])


def prepare_fold(X_train, y_train, X_validation, y_validation, config):
    """一次 fold 前處理；validation 只 transform，IQR remove 僅影響 training。"""
    assert_aligned(X_train, y_train)
    assert_aligned(X_validation, y_validation)
    if X_train.index.intersection(X_validation.index).size:
        raise ValueError("Training／Validation 索引不可重疊。")
    for target in (y_train, y_validation):
        if not pd.api.types.is_numeric_dtype(target) or not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError("fold Target 必須為非缺失有限數值；請先使用 split_dataset 處理 Target。")
    if config.target_column in X_train or config.target_column in X_validation:
        raise ValueError("fold X 不得包含 Target。")
    if config.split_strategy == "chronological":
        train_dates, valid_dates = chronological_dates(X_train, config), chronological_dates(X_validation, config)
        if not train_dates.is_monotonic_increasing or not valid_dates.is_monotonic_increasing or train_dates.max() >= valid_dates.min():
            raise ValueError("Chronological fold 必須按時間遞增且 validation 完全晚於 training。")
    types = resolve_types(X_train, config)
    train_X, train_y, dropped_train = _missing_filter(X_train, y_train, config, types)
    valid_X, valid_y, dropped_valid = _missing_filter(X_validation, y_validation, config, types)
    dropped_outliers = []
    bounds = {}
    if config.outlier_strategy == "remove":
        names = [name for name, kind in types.items() if kind == "numeric"]
        if names and len(train_X):
            values = np.column_stack([ColumnValues().fit_transform(train_X[[name]])[:, 0] for name in names])
            rule = IQRTransformer(multiplier=config.iqr_multiplier).fit(values)
            mask = rule.row_mask(values)
            dropped_outliers = list(train_X.index[~mask])
            bounds = {name: {"lower": float(lo) if np.isfinite(lo) else None,
                             "upper": float(hi) if np.isfinite(hi) else None}
                      for name, lo, hi in zip(names, rule.lower_, rule.upper_)}
            train_X, train_y = train_X.loc[mask], train_y.loc[mask]
    if len(train_X) < 2 or len(valid_X) == 0:
        raise ValueError("過濾後 Training 少於兩列或 Validation 為空；請調整設定。")
    processor = build_preprocessor(train_X, config, column_types=types, rows_prepared=True)
    transformed_train = processor.fit_transform(train_X, train_y)
    transformed_valid = processor.transform(valid_X)
    summary = feature_summary(processor, config)
    return PreparedFold(processor, transformed_train, transformed_valid, train_y, valid_y, summary,
                        {"train_input_rows": len(X_train), "validation_input_rows": len(X_validation),
                         "train_retained_rows": len(train_y), "validation_retained_rows": len(valid_y),
                         "train_missing_dropped_indices": dropped_train,
                         "validation_missing_dropped_indices": dropped_valid,
                         "train_outlier_dropped_indices": dropped_outliers,
                         "validation_outlier_removal": False, "column_types_from_training": types}, bounds)


def iter_preprocessed_folds(X_train, y_train, config):
    """僅傳入 outer Training；每個 fold 重建並 fit 一份 Pipeline。"""
    assert_aligned(X_train, y_train)
    if config.split_strategy == "chronological":
        dates = chronological_dates(X_train, config)
        if not dates.is_monotonic_increasing:
            raise ValueError("TimeSeriesSplit 輸入必須先依時間排序。")
        splitter = TimeSeriesSplit(n_splits=config.cv_splits, gap=config.cv_gap)
    else:
        splitter = KFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_state)
    for train, valid in splitter.split(X_train):
        yield prepare_fold(X_train.iloc[train], y_train.iloc[train], X_train.iloc[valid], y_train.iloc[valid], config)
