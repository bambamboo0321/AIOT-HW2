"""Browser transport only. All validation, selection, fitting and inference use src unchanged."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import platform
import numpy
import pandas
import scipy
import sklearn
import joblib
from src.custom_dataset_validation import validate_custom_dataset, _frame, _document
from src.custom_dataset_training import train_custom_dataset
from src.custom_dataset_comparison import compare_custom_dataset, settings
from src.custom_dataset_prediction import predict_saved_model, predict_saved_batch
from src.preprocessing_config import PreprocessingConfig

MODELS = {}
BASE = Path('/browser')

def versions():
    return dict(python=platform.python_version(), numpy=numpy.__version__, pandas=pandas.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)

def progress(method, phase):
    from js import browser_progress
    browser_progress(json.dumps(dict(stage='method', method=method, phase=phase)))

def sweep():
    now = int(time.time() * 1000)
    for key, model in list(MODELS.items()):
        if model['expires_at'] <= now:
            Path(model['path']).unlink(missing_ok=True)
            del MODELS[key]

def execute(raw):
    request = json.loads(raw)
    op, payload = request['operation'], request['payload']
    root = BASE / 'jobs' / request['jobId']
    sweep()
    try:
        if op in ('predict', 'predict-batch', 'delete-model'):
            model = MODELS.get(payload.get('model_id'))
            if model is None:
                return dict(error='Browser 模型已清除或到期，請重新訓練。', http_status=410)
            if op == 'delete-model':
                Path(model['path']).unlink(missing_ok=True)
                del MODELS[payload['model_id']]
                return dict(status='deleted')
            if op == 'predict':
                return predict_saved_model(model['path'], payload['values'])
            return predict_saved_batch(model['path'], payload['features'], payload['rows'])
        csv, config, review = root/'input.csv', root/'config.json', root/'review.json'
        cfg = _document(config, 128*1024)
        # Independent browser admission limits; production rules and algorithms stay unchanged.
        if csv.stat().st_size > 1024*1024 or len(cfg.get('feature_columns', [])) > 10:
            raise ValueError('Browser Mode 限 1 MiB、10 個數值候選特徵。')
        if len(_frame(csv.read_bytes())) > 500:
            raise ValueError('Browser Mode 限 500 列；不會自動抽樣。')
        validation = validate_custom_dataset(csv, config, review)
        if op == 'validate':
            return validation
        if validation['status'] != 'valid':
            # Keep the original formal entry point's rejection schema.
            entry = train_custom_dataset if op == 'train' else compare_custom_dataset
            return entry(csv, config, review, root/'experiment.json')
        experiment = _document(root/'experiment.json', 4096)
        if op == 'train':
            method = experiment.get('selection', {}).get('method', 'none')
            progress(method, 'started')
            model_id = request['modelId']
            model_path = BASE/'models'/(model_id+'.joblib')
            try:
                report = train_custom_dataset(csv, config, review, root/'experiment.json', model_output=model_path)
                if report['status'] == 'completed':
                    if len(MODELS) >= 3:
                        oldest = next(iter(MODELS))
                        Path(MODELS.pop(oldest)['path']).unlink(missing_ok=True)
                    model = dict(id=model_id, expires_at=int(time.time()*1000)+1800000,
                        target=report['target'], features=report['selection']['final_features'],
                        missing_numeric=cfg['missing_numeric'], dataset_sha256=validation['dataset']['sha256'],
                        config_sha256=validation['config_sha256'])
                    MODELS[model_id] = dict(model, path=str(model_path))
                    report['prediction_model'] = model
                else:
                    model_path.unlink(missing_ok=True)
            except Exception:
                model_path.unlink(missing_ok=True)
                raise
            progress(method, 'completed')
            return report
        if op == 'compare':
            # Validate aggregate work bounds before any fit. Each method still calls
            # the original complete comparison entry point with unchanged k and CV.
            conf = PreprocessingConfig.from_dict(cfg)
            try:
                methods, ks = settings(experiment, conf, validation)
            except ValueError:
                return compare_custom_dataset(csv, config, review, root/'experiment.json')
            result = None
            for method in methods:
                progress(method, 'started')
                one = dict(experiment, methods=[method], k_values=[] if method in ('rfecv','lasso_cv') else ks)
                path = root/'method.json'
                path.write_text(json.dumps(one, ensure_ascii=False, allow_nan=False))
                report = compare_custom_dataset(csv, config, review, path)
                if report['status'] != 'compared':
                    return report
                if result is None:
                    result = dict(report, methods=methods, k_values=ks, points=[], automatic=[], warnings=[])
                result['points'].extend(report['points'])
                result['automatic'].extend(report['automatic'])
                result['warnings'] = list(dict.fromkeys(result['warnings']+report['warnings']))
                progress(method, 'completed')
            return result
        raise ValueError('Unsupported browser operation')
    finally:
        if root.is_dir():
            shutil.rmtree(root)
