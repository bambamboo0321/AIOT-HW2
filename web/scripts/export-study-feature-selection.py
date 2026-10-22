"""Export verified fixed-k/auto feature-selection evidence, never train.
Run explicitly from repository root:
python3 web/scripts/export-study-feature-selection.py [--check]
Production build/dev do not run this exporter or open experiments/.
"""
from pathlib import Path
import argparse
import json
import math
import runpy

BASE = runpy.run_path(str(Path(__file__).with_name('export-study-overviews.py')))
ROOT, read, digest, encode, close = [BASE[k] for k in ['ROOT','read','digest','encode','close']]
DEST = ROOT/'web/data/research-feature-selection/v1'
METHODS = ['f_test','mutual_information','forward_sfs','rfe','lasso_cv','rfecv']
AUTO = {'lasso_cv','rfecv'}
METRICS = ['rmse','mae','r2']

def valid_scores(v):
    return isinstance(v,dict) and all(isinstance(v.get(k),(int,float)) and math.isfinite(v[k]) for k in METRICS)

def export(id, definition):
    # Reuse Overview source verification, dataset/protocol adapter and source hashes.
    overview = BASE['export'](id,definition)
    overview_bytes = (ROOT/f'web/data/research-overviews/v1/{id}.json').read_bytes()
    assert encode(overview) == overview_bytes, 'Overview must still match its verified export'
    experiment, sub, *_ = definition
    results = read(str(Path(experiment)/sub/'results.json'))
    split, candidates = results['split'],overview['dataset']['candidates']
    training, holdout = set(split['training_rows']),set(split['holdout_rows'])
    outer = {f['fold']:f for f in split['outer_cv']}
    assert set(outer) == {1,2,3,4,5}
    for f in outer.values():
        tr,va=set(f['training_rows']),set(f['validation_rows'])
        assert not tr & va and tr | va == training and not (tr|va) & holdout
    records=[]
    for method in METHODS:
        mode='auto' if method in AUTO else 'fixed'
        m=results['methods'].get(method)
        identity={'id':method,'mode':mode,'requestedK':None if mode=='auto' else 7}
        if not m or m.get('status')!='PASS':
            records.append({**identity,'status':'unavailable','reason':'實驗未提供已通過驗證的方法成果。'})
            continue
        assert m['requested_k']==identity['requestedK']
        features=m.get('selected_features',m.get('features'))
        assert len(features)==m['selected_k'] and len(set(features))==len(features) and set(features)<=set(candidates)
        if mode=='fixed':assert m['selected_k']==7
        folds=[]
        for f in m.get('folds',[]):
            reference=outer[f['fold']]
            names=f.get('selected_features',f.get('features'))
            assert len(names)==f['selected_k'] and len(set(names))==len(names) and set(names)<=set(candidates)
            if mode=='fixed':assert f['selected_k']==7
            inner=f.get('inner_cv',[])
            if method in ['forward_sfs','lasso_cv','rfecv']:
                assert len(inner)==3
                for v in inner:
                    tr,va=set(v['training_rows']),set(v['validation_rows'])
                    assert not tr & va and tr | va == set(reference['training_rows']) and not (tr|va)&set(reference['validation_rows']) and not (tr|va)&holdout
            else:assert not inner
            scores=f.get('metrics')
            folds.append({'fold':f['fold'],'k':f['selected_k'],'features':names,'scores':{k:scores[k] for k in METRICS} if valid_scores(scores) else None})
        cv={'status':'unavailable','reason':'CV 摘要或五個有效 fold 分數缺失；未補造數值。'}
        summary=m.get('cv_summary')
        if len(folds)==5 and all(f['scores'] is not None for f in folds) and isinstance(summary,dict) and all(isinstance(summary.get(k),dict) and isinstance(summary[k].get('mean'),(int,float)) and isinstance(summary[k].get('std_ddof0'),(int,float)) for k in METRICS):
            for k in METRICS:
                mean=sum(f['scores'][k] for f in folds)/5
                std=math.sqrt(sum((f['scores'][k]-mean)**2 for f in folds)/5)
                close(summary[k]['mean'],mean);close(summary[k]['std_ddof0'],std)
            cv={'status':'ready','mean':{k:summary[k]['mean'] for k in METRICS},'std':{k:summary[k]['std_ddof0'] for k in METRICS},'foldCount':5,'stdDDoF':0}
        warning=m.get('warnings',{})
        warnings=[f'{key} ({value})' for key,value in warning.items()] if isinstance(warning,dict) else [str(w) for w in warning]
        records.append({**identity,'status':'ready','finalK':m['selected_k'],'features':features,'cv':cv,'folds':folds,'warnings':warnings})
    return {'schemaVersion':1,'artifactType':'study-feature-selection','datasetId':id,'overviewSha256':digest(overview_bytes),'sources':overview['sources'],'methods':records}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    outputs={id+'.json':encode(export(id,d)) for id,d in BASE['STUDIES'].items()}
    outputs['manifest.json']=encode({'schemaVersion':1,'snapshots':{id:{'file':id+'.json','sha256':digest(outputs[id+'.json'])} for id in BASE['STUDIES']}})
    if args.check:
        for name,data in outputs.items():assert (DEST/name).read_bytes()==data,name+' differs from verified export'
    else:
        DEST.mkdir(parents=True,exist_ok=True)
        for name,data in outputs.items():(DEST/name).write_bytes(data)
    print(json.dumps({'status':'PASS','checkOnly':args.check,'files':list(outputs),'methods':12,'outerFoldsVerified':60,'bytes':sum(map(len,outputs.values()))}))

if __name__=='__main__':main()
