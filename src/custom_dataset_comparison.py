"""Confirmed, bounded Training-only fixed-k comparison; never evaluates Holdout."""
import argparse
import hashlib
import json
import sys
import warnings

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from .custom_dataset_selectors import (AUTO_METHODS, NESTED_METHODS, CustomSelectionError, check_nested_request, estimated_fits, fit_custom_selection)

from .custom_dataset_validation import _bytes, _document, _frame, _numeric, validate_custom_dataset, MAX_BYTES
from .custom_dataset_training import metrics, summary
from .data_splitting import split_dataset
from .feature_selection import SelectionSpec, fit_selected_regression
from .preprocessing_config import PreprocessingConfig

METHODS = ('f_test', 'mutual_information', 'rfe', 'forward_sfs', 'rfecv', 'lasso_cv')
SCOPE = 'custom_feature_comparison'
MAX_WORK = 100_000_000


class ComparisonRequestError(ValueError):
    pass


def settings(experiment, config, validation):
    keys = {'schema_version', 'confirmed', 'dataset_sha256', 'config_sha256', 'methods', 'k_values'}
    if (not isinstance(experiment, dict) or set(experiment) != keys
            or type(experiment.get('schema_version')) is not int or experiment['schema_version'] != 1
            or experiment.get('confirmed') is not True
            or experiment['dataset_sha256'] != validation['dataset']['sha256']
            or experiment['config_sha256'] != validation['config_sha256']):
        raise ComparisonRequestError('確認與資料版本不一致。')
    methods, ks = experiment['methods'], experiment['k_values']
    p = len(config.feature_columns)
    if (not isinstance(methods, list) or not methods or len(methods) > 6
            or any(type(m) is not str or m not in METHODS for m in methods) or len(set(methods)) != len(methods)
            or not isinstance(ks, list) or len(ks) > 25
            or any(type(k) is not int or not 1 <= k <= p for k in ks) or ks != sorted(set(ks))):
        raise ComparisonRequestError('請指定支援的方法與遞增、唯一的整數 k。')
    fixed = [m for m in methods if m not in AUTO_METHODS]
    if bool(fixed) != bool(ks):
        raise ComparisonRequestError('固定方法需要 k 範圍；只有自動方法時 k_values 必須為空陣列。')
    for m in methods:
        for k in (ks if m not in AUTO_METHODS else [None]):
            check_nested_request(m, k, config, validation, include_final=False)
    if config.numerical_scaling != 'standard':
        raise ComparisonRequestError('比較需要 standard 標準化；請回上方調整並重新驗證。')
    if p > 25 or validation['dataset']['rows'] > 5000:
        raise ComparisonRequestError('比較限 5,000 筆、25 個候選特徵；請縮小資料或候選欄位。')
    if min(f['training_rows'] for f in validation['split']['cv_folds']) < 4:
        raise ComparisonRequestError('每個 CV Training 至少需要 4 筆。')
    # Conservative bound counts each elimination/refit against the full matrix.
    fits = sum(sum(estimated_fits(m, k, p) for k in (ks if m not in AUTO_METHODS else [None])) for m in methods)
    work = fits * p * sum(f['training_rows'] for f in validation['split']['cv_folds'])
    if work > MAX_WORK:
        raise ComparisonRequestError('比較運算量過大；請縮小 k 範圍、減少方法／folds 或資料筆數。')
    return [m for m in METHODS if m in methods], ks


def compare_custom_dataset(csv_path, config_path, review_path, experiment_path):
    common = {'schema_version': 1, 'scope': SCOPE}
    started, stage = False, 'confirmation'
    try:
        validation = validate_custom_dataset(csv_path, config_path, review_path)
        if validation['status'] != 'valid':
            return {**common, 'status': 'rejected', 'comparison_started': False, 'errors': validation['errors']}
        if not validation['review_checked']:
            raise ComparisonRequestError('請先確認並驗證資料設定。')
        config = PreprocessingConfig.from_dict(_document(config_path, 128*1024))
        normalized = json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if hashlib.sha256(normalized.encode()).hexdigest() != validation['config_sha256']:
            raise ComparisonRequestError('設定版本已改變，請重新驗證。')
        methods, ks = settings(_document(experiment_path, 4096), config, validation)
        stage = 'data'
        raw = _bytes(csv_path, MAX_BYTES, '.csv')
        if hashlib.sha256(raw).hexdigest() != validation['dataset']['sha256']:
            raise ComparisonRequestError('資料版本已改變。')
        frame = _frame(raw)
        for name in set(config.feature_columns) | {config.target_column}:
            frame[name] = _numeric(frame[name])
        split = split_dataset(frame, config)
        # Materialize exactly once; identical raw indices for every method/k.
        splitter = (TimeSeriesSplit(n_splits=config.cv_splits, gap=config.cv_gap)
                    if config.split_strategy == 'chronological'
                    else KFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_state))
        partitions = list(splitter.split(split.X_train))
        started, stage = True, 'comparison'
        points, automatic, notes = [], [], list(validation['warnings'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for method in methods:
                for k in (ks if method not in AUTO_METHODS else [None]):
                    folds = []
                    for i, (train, valid) in enumerate(partitions, 1):
                        X, y = split.X_train.iloc[train], split.y_train.iloc[train]
                        spec = SelectionSpec(method=method, k=k, random_state=config.random_state, gap=config.cv_gap)
                        details = None
                        if method in NESTED_METHODS:
                            selected = fit_custom_selection(X, y, config, spec)
                            names = [config.feature_columns[int(j)] for j in selected.indices]
                            details = selected.metadata['custom_details']
                            if len(names):
                                lr = LinearRegression().fit(selected.transform(X), y)
                                scores = metrics(split.y_train.iloc[valid], lr.predict(selected.transform(split.X_train.iloc[valid])))
                            else:
                                scores = None
                        else:
                            fitted = fit_selected_regression(X, y, config, spec)
                            names = [config.feature_columns[int(j)] for j in fitted.selection.indices]
                            scores = metrics(split.y_train.iloc[valid], fitted.predict(split.X_train.iloc[valid]))
                        if method not in AUTO_METHODS and len(names) != k:
                            raise ComparisonRequestError('Selector 回傳欄位數不符。')
                        folds.append({'fold': i, 'training_rows': len(train), 'validation_rows': len(valid), 'features': names, 'scores': scores,
                                      **({'details': details} if details else {})})
                    if method in AUTO_METHODS:
                        nullable = [{'scores': f['scores'] or {'rmse': None, 'mae': None, 'r2': None}} for f in folds]
                        automatic.append({'method': method, 'folds': folds, 'metrics': summary(nullable, 'scores')})
                        if any(f['scores'] is None for f in folds):
                            notes.append('LassoCV 某些 fold 選出 0 個特徵，該 fold 不訓練 LR 且不計分；平均只涵蓋有分數的 folds，請勿直接當成完整評估排名。')
                    else:
                        points.append({'method': method, 'k': k, 'ranking_performed': k < len(config.feature_columns), 'folds': folds, 'metrics': summary(folds, 'scores')})
            if caught:
                notes.append('選欄或線性求解器發出數值警告；請檢查常數欄位、共線性及尺度。')
        return {**common, 'status': 'compared', 'comparison_started': True, 'validation': validation,
                'target': config.target_column, 'features': list(config.feature_columns), 'methods': methods, 'k_values': ks,
                'scaling': 'standard', 'points': points, 'automatic': automatic, 'warnings': list(dict.fromkeys(notes)),
                'limits': ['只在 Training 區內 CV 比較；沒有計算 Holdout 分數、完整 Training 最終模型或 Prediction Interval。',
                           '每個方法／k 使用相同切分；前處理、選欄與獨立 Linear Regression 只 fit 當前 fold Training。',
                           'k 是實際送入線性迴歸的數值特徵數；k 等於候選數時保留全部，不執行排名。',
                           '各 fold 等權平均與標準差 ddof=0；標準差不是信賴區間，R² 無法定義時為 null。',
                           '依曲線挑選方法或 k 屬於探索性調參，最佳 CV 分數可能偏樂觀；需固定設定後再做一次獨立 Holdout 評估。',
                           '固定方法為 F-test、MI、RFE、SFS 的 k 曲線；RFECV／LassoCV 為內部 3-fold 自動選 k 的外層 CV 結果，不補造完整 k 曲線。Lasso alpha 固定候選為 10000 到 0.0001 的 9 個十倍級距。']}
    except (ValueError, TypeError, KeyError, OverflowError, FloatingPointError, OSError, np.linalg.LinAlgError) as error:
        message = (str(error) if stage == 'confirmation' and isinstance(error, (ComparisonRequestError, CustomSelectionError))
                   else '資料版本不一致或無法讀取。' if not started
                   else '比較無法完成；請檢查缺失、尺度與切分設定，或縮小比較範圍。')
        return {**common, 'status': 'failed' if started else 'rejected', 'comparison_started': started,
                'errors': [{'stage': stage, 'message': message}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Training-only fixed-k feature comparison')
    for name in ('csv', 'config', 'review', 'experiment'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    result = compare_custom_dataset(args.csv, args.config, args.review, args.experiment)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['status'] == 'compared' else 2


if __name__ == '__main__':
    sys.exit(main())
