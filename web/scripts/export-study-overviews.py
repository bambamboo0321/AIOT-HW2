"""Explicit, dependency-free export from verified experiments; NEVER run by build/dev.
Usage: python3 web/scripts/export-study-overviews.py [--check]
No fitting, raw-row copying, model copying, or experiment writes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "web/data/research-overviews/v1"
STUDIES = {
    "automobile-price": ("experiments/multi-dataset-phase2", "automobile", "Automobile Price", "price", "原始價格單位（幣別未核實）"),
    "student-performance": ("experiments/student-performance-phase2", "", "Student Performance", "G3", "分數（0–20）"),
}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads((ROOT / path).read_bytes())

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)

def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()

def export(id, definition):
    experiment, sub, name, target, unit = definition
    base = str(Path(experiment) / sub)
    results_path, decision_path = base + "/results.json", base + "/locked-training-decision.json"
    config_path, verification_path = experiment + "/config.json", experiment + "/verification.json"
    results, config, decision, verification = [read(p) for p in [results_path, config_path, decision_path, verification_path]]
    assert verification['status'] == 'PASS'
    # Check the existing experiment's own recorded hashes, not just its PASS label.
    recorded = read(experiment + '/artifact-manifest.json')['sha256']
    for path in [results_path, decision_path, config_path]:
        relative = str(Path(path).relative_to(experiment))
        assert recorded[relative] == digest((ROOT/path).read_bytes()), path
    profile, prep, split = results['profile'], results['config'], results['split']
    assert profile['sha256'] == digest((ROOT/profile['file']).read_bytes())
    assert config['seed'] == prep['random_state'] == 42
    assert config['outer_folds'] == prep['cv_splits'] == 5 and config['inner_folds'] == 3 and config['fixed_k'] == 7
    assert prep['missing_numeric'] == 'median' and prep['numerical_scaling'] == 'standard' and prep['split_strategy'] == 'random'
    assert prep['target_column'] == target
    assert decision['method'] == 'mutual_information' and decision.get('locked_before_holdout', decision.get('locked_before_holdout_evaluation')) is True and decision['k_tuned'] is False
    assert not set(split['training_rows']) & set(split['holdout_rows'])
    models = []
    for method, label in [('baseline', 'Baseline LR'), ('mutual_information', 'MI + LR')]:
        m = results['methods'].get(method)
        if not m or m.get('status') != 'PASS' or 'holdout' not in m:
            models.append({'id': method, 'label': label, 'status': 'unavailable', 'reason': '實驗未提供完整且通過驗證的模型成果。'})
            continue
        hold = m['holdout']; actual, pred = hold['actual'], hold['prediction']; n = len(actual)
        assert n == len(split['holdout_rows']) == len(pred)
        assert hold['source_rows'] == split['holdout_rows']
        mean = sum(actual)/n
        metrics = {'mae': sum(abs(a-b) for a,b in zip(actual,pred))/n,
                   'rmse': math.sqrt(sum((a-b)**2 for a,b in zip(actual,pred))/n),
                   'r2': 1-sum((a-b)**2 for a,b in zip(actual,pred))/sum((a-mean)**2 for a in actual)}
        for k in metrics: close(metrics[k], hold['metrics'][k])
        assert len(m['folds']) == 5
        for k in metrics: close(sum(f['metrics'][k] for f in m['folds'])/5, m['cv_summary'][k]['mean'])
        interval = m.get('prediction_interval', m.get('pi95', {}))
        pi = {'status': 'unavailable', 'reason': '實驗未提供可驗證的 95% Prediction Interval。'}
        if interval.get('status') == 'PASS':
            lower, upper, pm = interval['lower'], interval['upper'], interval['metrics']
            assert len(lower) == len(upper) == n
            assert all(l <= h for l,h in zip(lower,upper))
            covered = sum(l <= a <= h for a,l,h in zip(actual,lower,upper))
            close(covered/n, pm['empirical_coverage']); close(sum(h-l for l,h in zip(lower,upper))/n,pm['average_width'])
            assert pm['nominal_coverage'] == .95 and pm['covered_count'] == covered and pm['total_holdout_samples'] == n
            assert interval['diagnostics']['holdout_residuals_used'] is False
            pi = {'status':'ready','nominal':.95,'coverage':pm['empirical_coverage'],'averageWidth':pm['average_width'],'covered':covered,'total':n}
        features = m.get('selected_features', m.get('features'))
        assert len(features) == m['selected_k'] and set(features) <= set(prep['feature_columns'])
        models.append({'id':method,'label':label,'status':'ready','features':features,'k':m['selected_k'],
                       'holdout':hold['metrics'],'cv':{k:m['cv_summary'][k]['mean'] for k in ['mae','rmse','r2']},'interval':pi})
    sources = [results_path, decision_path, config_path, verification_path, experiment+'/artifact-manifest.json']
    return {'schemaVersion':1,'datasetId':id,
            'dataset':{'name':name,'target':target,'unit':unit,'rawRows':profile.get('raw_rows',profile.get('rows')),'rows':profile['valid_rows'],
                       'candidates':prep['feature_columns'],'removedTargets':profile.get('missing_target_rows_removed',0),'rawFile':profile['file'],'rawSha256':profile['sha256']},
            'protocol':{'trainRows':len(split['training_rows']),'holdoutRows':len(split['holdout_rows']),'seed':42,'outerFolds':5,'innerFolds':3,'fixedK':7,
                        'split':'random','scaling':'standard','imputation':'median','selectionLockedBeforeHoldout':True,'selectedMethod':'mutual_information'},
            'sources':[{'path':p,'sha256':digest((ROOT/p).read_bytes())} for p in sources], 'models':models}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    outputs = {id+'.json':encode(export(id,d)) for id,d in STUDIES.items()}
    outputs['manifest.json'] = encode({'schemaVersion':1,'snapshots':{id:{'file':id+'.json','sha256':digest(outputs[id+'.json'])} for id in STUDIES}})
    if args.check:
        for name, data in outputs.items(): assert (DEST/name).read_bytes() == data, name + ' differs from verified source export'
    else:
        DEST.mkdir(parents=True,exist_ok=True)
        for name, data in outputs.items(): (DEST/name).write_bytes(data)
    print(json.dumps({'status':'PASS','checkOnly':args.check,'files':list(outputs),'bytes':sum(map(len,outputs.values()))}))

if __name__ == '__main__': main()
