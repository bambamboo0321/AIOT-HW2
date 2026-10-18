"""Inference from a trusted, server-created temporary bundle. Never fits a model."""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from .custom_dataset_validation import _document


def save_model_bundle(path, final, model, indices, training, config, validation):
    columns = final.preprocessor.named_steps['columns']
    branches = []
    for index in indices:
        index = int(index)
        name = config.feature_columns[index]
        key = f'c{index}'
        section = columns.output_indices_[key]
        if section.start != index or section.stop != index + 1:
            raise ValueError('Only one-to-one numeric branches are supported.')
        branch = columns.named_transformers_[key]
        finite = training[name].dropna().to_numpy(dtype=float)
        branches.append({'name': name, 'transformer': branch,
                         'range': [float(finite.min()), float(finite.max())] if len(finite) else None})
    bundle = {'schema_version': 1, 'target': config.target_column,
              'missing_numeric': config.missing_numeric, 'branches': branches, 'model': model,
              'dataset_sha256': validation['dataset']['sha256'], 'config_sha256': validation['config_sha256']}
    # Exclusive creation: this internal option cannot overwrite an existing artifact.
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as output:
        joblib.dump(bundle, output)


def predict_saved_model(path, values):
    # path is resolved by the server registry, never accepted from HTTP clients.
    bundle = joblib.load(path)
    names = [b['name'] for b in bundle['branches']]
    if (bundle['schema_version'] != 1 or not isinstance(values, dict) or set(values) - set(names)
            or not any(value is not None for value in values.values())):
        raise ValueError('請至少輸入一個模型使用的特徵；不接受額外欄位。')
    transformed, imputed, outside = [], [], []
    for branch in bundle['branches']:
        name = branch['name']
        value = values.get(name)
        if value is None:
            if bundle['missing_numeric'] not in ('mean', 'median'):
                raise ValueError('訓練時選擇不補值，請填完整模型使用的特徵。')
            fill = float(branch['transformer'].named_steps['impute'].statistics_[0])
            imputed.append({'feature': name, 'value': fill})
            value = np.nan
        elif type(value) not in (int, float) or not np.isfinite(value) or abs(value) > 2**53 - 1:
            raise ValueError('特徵必須是有限數字，且絕對值不可超過安全數值上限。')
        elif branch['range'] and not branch['range'][0] <= value <= branch['range'][1]:
            outside.append(name)
        # Only frozen selected branches are needed, including when unselected
        # candidates used missing_numeric=none. Do not refit or rerun selection.
        column = branch['transformer'].transform(pd.DataFrame({name: [value]}))
        if column.shape != (1, 1) or not np.isfinite(column).all():
            raise ValueError('前處理無法產生有限數值。')
        transformed.append(column)
    prediction = float(bundle['model'].predict(np.concatenate(transformed, axis=1))[0])
    if not np.isfinite(prediction):
        raise ValueError('模型無法產生有限預測，請檢查數值尺度。')
    notes = []
    if imputed:
        notes.append('空白欄位已使用完整 Training 學到的補值；這不代表已知道該筆資料的真實特徵。')
    if outside:
        notes.append('部分輸入超出 Training 的觀測範圍，預測可靠性可能降低；沿用訓練時的極端值處理。')
    if prediction < 0:
        notes.append('模型產生負值，未做裁切；請依目標的實際意義判斷是否合理。')
    return {'schema_version': 1, 'status': 'predicted', 'training_started': False,
            'target': bundle['target'], 'features': names, 'prediction': prediction,
            'dataset_sha256': bundle['dataset_sha256'], 'config_sha256': bundle['config_sha256'],
            'imputed': imputed, 'out_of_range': outside, 'warnings': notes}



def predict_saved_batch(path, features, rows):
    """Vectorized inference; load once, transform frozen selected branches once."""
    bundle = joblib.load(path)
    names = [b['name'] for b in bundle['branches']]
    if (bundle['schema_version'] != 1 or features != names or not isinstance(rows, list)
            or not 1 <= len(rows) <= 500 or len(rows) * len(names) > 10000):
        raise ValueError('批次欄位或筆數不符。')
    for row in rows:
        if (not isinstance(row, list) or len(row) != len(names)
                or not any(v is not None for v in row)
                or any(v is not None and (type(v) not in (int, float) or not np.isfinite(v)
                                         or abs(v) > 2**53 - 1) for v in row)
                or (bundle['missing_numeric'] == 'none' and any(v is None for v in row))):
            raise ValueError('批次包含無效或未處理的缺失值。')
    matrix = np.asarray(rows, dtype=float)
    transformed, fills = [], []
    missing = np.isnan(matrix)
    outside = np.zeros(matrix.shape, dtype=bool)
    for j, branch in enumerate(bundle['branches']):
        if missing[:, j].any():
            if bundle['missing_numeric'] not in ('mean', 'median'):
                raise ValueError('模型不支援補值。')
            fills.append({'feature': names[j], 'value': float(branch['transformer'].named_steps['impute'].statistics_[0])})
        if branch['range']:
            outside[:, j] = (~missing[:, j]) & ((matrix[:, j] < branch['range'][0]) | (matrix[:, j] > branch['range'][1]))
        column = branch['transformer'].transform(pd.DataFrame({names[j]: matrix[:, j]}))
        if column.shape != (len(rows), 1) or not np.isfinite(column).all():
            raise ValueError('批次前處理無法產生有限數值。')
        transformed.append(column)
    predictions = bundle['model'].predict(np.concatenate(transformed, axis=1))
    if not np.isfinite(predictions).all():
        raise ValueError('批次模型無法產生有限數值。')
    notes = []
    if missing.any():
        notes.append('空白使用完整 Training 學到的補值；不是新資料的真實觀測值。')
    if outside.any():
        notes.append('部分輸入超出 Training 範圍，可靠性可能降低；沿用原本極端值處理。')
    if (predictions < 0).any():
        notes.append('部分預測為負值，未裁切；請依目標實際意義判斷。')
    return {'schema_version': 1, 'status': 'batch_predicted', 'training_started': False,
            'target': bundle['target'], 'features': names, 'dataset_sha256': bundle['dataset_sha256'],
            'config_sha256': bundle['config_sha256'], 'row_count': len(rows),
            'rows': [{'position': i + 1, 'prediction': float(v),
                      'imputed_features': [names[j] for j in np.flatnonzero(missing[i])],
                      'out_of_range_features': [names[j] for j in np.flatnonzero(outside[i])]}
                     for i, v in enumerate(predictions)],
            'imputation_values': fills, 'warnings': notes}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Use a trusted temporary model; no training.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.batch:
            payload = _document(args.input, 2 * 1024 * 1024)
            if not isinstance(payload, dict) or set(payload) != {'features', 'rows'}:
                raise ValueError('批次輸入格式不符。')
            result = predict_saved_batch(args.model, payload['features'], payload['rows'])
        else:
            result = predict_saved_model(args.model, _document(args.input, 64 * 1024))
    except (ValueError, TypeError, KeyError, OSError, EOFError, OverflowError, FloatingPointError):
        result = {'schema_version': 1, 'status': 'rejected', 'training_started': False,
                  'error': '無法預測；請檢查輸入欄位、有限數值與訓練時的補值設定。'}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['status'] in ('predicted', 'batch_predicted') else 2


if __name__ == '__main__':
    sys.exit(main())
