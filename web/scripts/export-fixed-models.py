"""Phase 3F-B. Reproduce only four frozen fits; never run selectors or alter research artifacts.
Run from repository root: .venv/bin/python -B web/scripts/export-fixed-models.py
All writes are confined to this milestone's new model, fixture and verification files.
"""
from pathlib import Path
import gzip
import hashlib
import importlib.metadata
import json
import sys
from contextlib import ExitStack
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, RFE, RFECV
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.feature_selection import fit_preprocessor
from src.preprocessing_config import PreprocessingConfig

WEB = ROOT / 'web'
POLICY = json.loads((WEB/'data/fixed-models/v1/policy.json').read_text())
assert POLICY['absoluteTolerance'] == POLICY['relativeTolerance'] == 1e-10
VERSIONS = {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl']}
assert VERSIONS['scikit-learn'] == '1.7.2', 'Use original native sklearn; do not upgrade existing environment'
DATASETS = ['appliances-energy','automobile-price','student-performance']
MODELS = ['baseline','mutual_information']

def digest(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def read(p): return json.loads((ROOT/p).read_text())
def encoded_json(d): return (json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def compact_hash(d): return hashlib.sha256(json.dumps(d,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def error_stats(actual, reference):
    a,b=np.asarray(actual,float),np.asarray(reference,float)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    diff=np.abs(a-b); failures=int(np.count_nonzero(diff>POLICY['absoluteTolerance']+POLICY['relativeTolerance']*np.abs(b)))
    stats={'maxAbsoluteError':float(diff.max(initial=0)), 'maxRelativeError':float((diff/np.maximum(np.abs(b),POLICY['relativeErrorReportingFloor'])).max(initial=0)), 'failures':failures,'rows':len(a)}
    if failures: raise ValueError(f'Numerical FAIL: {stats}')
    return stats

def source_records(paths): return [{'path':p,'sha256':digest(p)} for p in dict.fromkeys(paths)]
COMMON_SOURCES=['web/scripts/export-fixed-models.py','web/data/fixed-models/v1/policy.json','src/preprocessing.py','src/preprocessing_config.py','src/feature_selection.py']

# Domain rules describe this supplied data version; training ranges are recorded separately.
STUDENT_CODES={'Medu':list(range(5)),'Fedu':list(range(5)),'traveltime':list(range(1,5)), 'studytime':list(range(1,5)), 'failures':list(range(4)), **{n:list(range(1,6)) for n in ['famrel','freetime','goout','Dalc','Walc','health']}}

def feature_spec(dataset,name,train,processor,index,mapping):
    raw=train[name].dropna().to_numpy(float)
    f={'name':name,'kind':'category' if mapping else ('integer' if dataset=='student-performance' else 'number'),
       'unit':'原始欄位單位（未另行核實）','trainingMin':float(raw.min()),'trainingMax':float(raw.max()),
       'domainMin':None,'domainMax':None,'allowedValues':None,'encoding':mapping,'imputer':None,'scaler':None,'outliers':'keep',
       'description':name}
    if dataset=='student-performance':
        f['unit']='歲' if name=='age' else '次' if name=='absences' else '分' if name in ['G1','G2'] else '序數編碼'
        f['allowedValues']=STUDENT_CODES.get(name)
        f['domainMin']=0
        if name in ['G1','G2']:f['domainMax']=20;f['description']='已知的第一期成績' if name=='G1' else '已知的第二期成績'
        if name=='absences':f['description']='缺席次數；原資料累計截止時點未明'
        if name=='failures':f['description']='此供應 CSV 版本保留的 0–3 編碼；不推廣為任意失敗次數'
    if processor is not None:
        branch=processor.named_steps['columns'].named_transformers_[f'c{index}'].named_steps
        assert branch['values'].kind=='numeric' and branch['outliers'].strategy=='keep'
        f['imputer']={'strategy':'median','value':float(branch['impute'].statistics_[0])}
        f['scaler']={'mean':float(branch['scale'].mean_[0]),'scale':float(branch['scale'].scale_[0]),'withMean':True,'withStd':True}
        assert f['scaler']['scale']>0
    # Original Automobile policy requires categories even though the numeric transformer has a median.
    if mapping:f['imputer']=None
    return f

def raw_inputs(row,features,mappings):
    result={}
    for n in features:
        v=row[n]
        if pd.isna(v):result[n]=None
        elif n in mappings:result[n]=next(k for k,x in mappings[n].items() if x==v)
        else:result[n]=float(v)
    return result

def validate_scalar(f,v):
    if v is None:return f['imputer'] is not None
    if f['encoding']:return v in f['encoding'] if isinstance(v,str) else v in f['encoding'].values()
    if not isinstance(v,(int,float)) or not np.isfinite(v):return False
    if f['kind']=='integer' and v!=int(v):return False
    return not ((f['domainMin'] is not None and v<f['domainMin']) or (f['domainMax'] is not None and v>f['domainMax']) or (f['allowedValues'] is not None and v not in f['allowedValues']))

def extra_cases(model,train,mappings,native):
    fields=model['features'];base=raw_inputs(train.iloc[0],[f['name'] for f in fields],mappings)
    cases=[]
    def add(name,values):
        assert all(validate_scalar(f,values[f['name']]) for f in fields),name
        cases.append({'name':name,'inputs':values})
    add('training-example',dict(base))
    for f in fields:
        name=f['name']
        if f['encoding']:
            for label in f['encoding']:add(f'category:{name}:{label}',{**base,name:label})
        else:
            for tag,key in [('min','trainingMin'),('max','trainingMax')]:add(f'boundary:{name}:{tag}',{**base,name:f[key]})
            candidate=f['trainingMax']+(1 if f['kind']=='integer' else max(1,(f['trainingMax']-f['trainingMin'])*.05))
            if validate_scalar(f,candidate):add(f'extrapolation:{name}',{**base,name:candidate})
            if f['imputer']:add(f'missing:{name}',{**base,name:None})
    for direction in ['min','max']:
        edge={}
        for i,f in enumerate(fields):
            high=(model['coefficients'][i]>=0)==(direction=='max')
            value=f['trainingMax'] if high else f['trainingMin']
            edge[f['name']]=next(k for k,v in f['encoding'].items() if v==value) if f['encoding'] else value
        add('linear-extreme:'+direction,edge)
    all_missing={f['name']:(None if f['imputer'] and not f['encoding'] else base[f['name']]) for f in fields}
    if all_missing!=base:add('missing:all-eligible',all_missing)
    frames=[]
    for case in cases:
        row=train.iloc[[0]].copy()
        for n,v in case['inputs'].items():row[n]=np.nan if v is None else mappings[n][v] if n in mappings and isinstance(v,str) else v
        frames.append(row)
    predictions=native(pd.concat(frames,ignore_index=True))
    for case,pred in zip(cases,predictions):case['nativePrediction']=float(pred)
    return cases

def prepare_dataset(dataset):
    if dataset=='appliances-energy':
        rawpath='data/raw/KAG_energydata_complete.csv'
        assert digest(rawpath)=='2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d'
        frame=pd.read_csv(ROOT/rawpath); frame['date']=pd.to_datetime(frame['date'])
        split=read('reports/phase2/split_indices.json')['chronological'];t,h=split['train'],split['test']
        assert t==list(range(15788)) and h==list(range(15788,19735))
        return frame.loc[t],frame.loc[h],None,None,{},None,[rawpath,'reports/phase2/split_indices.json','reports/phase3/phase3c/prediction_interval_predictions.csv','web/data/prediction-explorer.json']
    automobile=dataset=='automobile-price'
    folder='experiments/multi-dataset-phase2/automobile' if automobile else 'experiments/student-performance-phase2'
    configpath='experiments/multi-dataset-phase2/config.json' if automobile else folder+'/config.json'
    setting=read(configpath);spec=setting['datasets']['automobile'] if automobile else setting
    rawpath=spec['file'];assert digest(rawpath)==spec['sha256']
    raw=pd.read_csv(ROOT/rawpath,na_values=['?'],sep=spec.get('delimiter',','));frame=raw.loc[raw[spec['target']].notna()].copy()
    X=frame[spec['features']].copy();mappings=spec.get('binary_categories',{})
    for name,mapping in mappings.items():
        assert not X[name].isna().any() and set(X[name])<=set(mapping)
        X[name]=X[name].map(mapping)
    X=X.astype(float);y=frame[spec['target']].astype(float)
    saved=read(folder+'/results.json');split=saved['split'];t,h=split['training_rows'],split['holdout_rows']
    tp,hp=train_test_split(np.arange(len(X)),test_size=setting['holdout_fraction'],random_state=setting['seed'])
    assert X.index[tp].tolist()==t and X.index[hp].tolist()==h and not set(t)&set(h)
    assert saved['config']['outlier_strategy']=='keep' and saved['config']['numerical_scaling']=='standard' and saved['config']['missing_numeric']=='median'
    return X.loc[t],X.loc[h],y.loc[t],saved,mappings,PreprocessingConfig(**saved['config']),[rawpath,configpath,folder+'/results.json',folder+'/profile-and-split.json',folder+'/live-predictor-schema.json',folder+'/locked-training-decision.json',('experiments/multi-dataset-phase2/environment.json' if automobile else folder+'/environment.json')]


def run():
    artifacts={};fixture={'schemaVersion':1,'policy':POLICY,'datasets':{}};report={'status':'PASS','policy':POLICY,'models':{},'selectorFits':0,'energyFits':0,'reconstructedLinearRegressionFits':4,'holdoutUsedForFitting':False,'versions':{'python':sys.version.split()[0],**VERSIONS}}
    source_paths=set()
    with ExitStack() as guards,threadpool_limits(limits=1):
        for cls in [SelectKBest,SequentialFeatureSelector,RFE,RFECV,Lasso,LassoCV]:guards.enter_context(patch.object(cls,'fit',side_effect=RuntimeError('Selector fitting forbidden in Phase 3F-B')))
        for dataset in DATASETS:
            train,hold,y,saved,mappings,cfg,paths=prepare_dataset(dataset)
            sources=source_records(COMMON_SOURCES+paths);source_paths.update(x['path'] for x in sources)
            rows_fixture={};fixture['datasets'][dataset]=rows_fixture
            for model_id in MODELS:
                modelpaths=[]
                if dataset=='appliances-energy':
                    modelpath='reports/phase2/models/chronological_baseline.joblib' if model_id=='baseline' else 'reports/phase3/results/models/final_linear_regression.joblib'
                    expected='9f9330522d7b52d6d279bf4b05f9a5f9123e1693e760dbd540910d74cd6a33ed' if model_id=='baseline' else '8be97c965b56fed6d4a665703b44faf57508fd9a759b837d48cf3a2ad418eb12'
                    assert digest(modelpath)==expected
                    obj=joblib.load(ROOT/modelpath);modelpaths=[modelpath]
                    if model_id=='baseline':
                        reg=obj['model'];names=obj['feature_names'];original=names;processor=None;indices=list(range(len(names)))
                        def native(frame,reg=reg,names=names):return reg.predict(frame[names])
                    else:
                        reg=obj.predictor;processor=obj.selection.preprocessor;indices=obj.selection.indices.tolist();original=list(processor.feature_names_in_);names=[original[i] for i in indices]
                        def native(frame,obj=obj):return obj.predict(frame)
                    # All energy execution including prediction stays inside the no-fit guards.
                    energy_guards=ExitStack()
                    for cls in [LinearRegression,Pipeline,SimpleImputer,StandardScaler]:energy_guards.enter_context(patch.object(cls,'fit',side_effect=RuntimeError('Energy fit forbidden')))
                    energy_guards.enter_context(patch.object(Pipeline,'fit_transform',side_effect=RuntimeError('Energy fit_transform forbidden')))
                    saved_rows=read('web/data/prediction-explorer.json')['data']['rows']
                    saved_pred=[r['baseline' if model_id=='baseline' else 'selected']['prediction'] for r in saved_rows]
                    csv=pd.read_csv(ROOT/'reports/phase3/phase3c/prediction_interval_predictions.csv');assert csv['Row Index'].tolist()==hold.index.tolist()
                    assert np.array_equal(np.array(saved_pred),csv['Baseline Prediction' if model_id=='baseline' else 'Selected Prediction'].to_numpy()) or error_stats(saved_pred,csv['Baseline Prediction' if model_id=='baseline' else 'Selected Prediction'])['failures']==0
                    seed=None;strategy='chronological_first_80_percent';target='Appliances';unit='Wh'
                else:
                    r=saved['methods'][model_id];names=r.get('selected_features',r.get('features'));original=list(train.columns)
                    indices=[original.index(n) for n in names]
                    assert len(names)==(len(original) if model_id=='baseline' else 7)
                    processor,values=fit_preprocessor(train,y,cfg)
                    reg=LinearRegression().fit(values[:,indices],y)
                    def native(frame,processor=processor,reg=reg,indices=indices):return reg.predict(np.asarray(processor.transform(frame))[:,indices])
                    saved_pred=r['holdout']['prediction'];assert r['holdout']['source_rows']==hold.index.tolist()
                    # Reconstructed preprocessor must also match the saved, Training-only state.
                    schema=r.get('live_predictor',r.get('live_predictor_schema'));state=schema.get('training_only_preprocessing',schema.get('state'))
                    for i in indices:
                        n=original[i];steps=processor.named_steps['columns'].named_transformers_[f'c{i}'].named_steps;s=state[n]
                        for actual,expected in [(steps['impute'].statistics_[0],s.get('imputation_median',s.get('median'))),(steps['scale'].mean_[0],s.get('standard_scaler_mean',s.get('mean'))),(steps['scale'].scale_[0],s.get('standard_scaler_scale',s.get('scale')))]:
                            assert actual==expected,(dataset,n,'preprocessor state differs')
                    seed=42;strategy='fixed_saved_random_80_20';target='price' if dataset=='automobile-price' else 'G3';unit='原始價格單位（幣別未核實）' if dataset=='automobile-price' else '分數（0–20）'
                fields=[feature_spec(dataset,n,train,processor,i,mappings.get(n)) for n,i in zip(names,indices)]
                limitation=['固定研究模型；不是因果效果；外插可靠性可能降低；預測與輸入不裁切。']
                limitation+=['必須已知 G1、G2；非學期開始前預測。補值不代表實際已知成績。','absences 累計截止時點未明。'] if dataset=='student-performance' else ['價格幣別未核實。'] if dataset=='automobile-price' else ['同時點耗能估計，非多步未來預報；date 不參與數值推論。']
                model={'schemaVersion':1,'modelVersion':POLICY['modelVersion'],'datasetId':dataset,'modelId':model_id,'target':target,'unit':unit,
                  'featureOrder':names,'features':fields,'originalPreprocessorSchema':original,'coefficients':reg.coef_.tolist(),'intercept':float(reg.intercept_),
                  'preprocessingOrder':['validate','encode','impute','scale','linear'],'outlierStrategy':'keep','predictionClipped':False,
                  'trainingSplit':{'strategy':strategy,'randomSeed':seed,'trainingRows':len(train),'holdoutRows':len(hold),'trainingIndexSha256':compact_hash(train.index.tolist()),'holdoutIndexSha256':compact_hash(hold.index.tolist()),'holdoutUsedForFitting':False},
                  'versions':{'python':sys.version.split()[0],**VERSIONS},'sourceHashes':sources+source_records(modelpaths),'limitations':limitation,
                  'acquisition':'original-saved-model-no-fit' if dataset=='appliances-energy' else 'training-only-refit-with-frozen-selected-columns-no-selector'}
                native_pred=np.asarray(native(hold),float);stats=error_stats(native_pred,saved_pred)
                cases=extra_cases(model,train,mappings,native)
                if dataset=='appliances-energy':energy_guards.close()
                filename=f'{dataset}--{model_id}.json';artifacts[filename]=encoded_json(model)
                raw_rows=[raw_inputs(row,names,mappings) for _,row in hold.iterrows()]
                rows_fixture[model_id]={'inputs':raw_rows,'savedPredictions':saved_pred,'nativePredictions':native_pred.tolist(),'extraCases':cases}
                report['models'][dataset+'--'+model_id]={'status':'PASS','holdoutRows':len(hold),'features':names,'acquisition':model['acquisition'],'nativeVsSaved':stats,'extraCases':len(cases)}
                print(dataset,model_id,stats,flush=True)
    for filename,content in artifacts.items():
        path=WEB/'public/research-models/v1'/filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    compressed=gzip.compress(json.dumps(fixture,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(),mtime=0)
    fixture_path=WEB/'tests/fixtures/fixed-models/v1/holdout.json.gz';fixture_path.write_bytes(compressed)
    manifest={'schemaVersion':1,'modelVersion':POLICY['modelVersion'],'policy':POLICY,'models':{name.removesuffix('.json'):{'path':'/research-models/v1/'+name,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)} for name,content in artifacts.items()},'fixtureSha256':hashlib.sha256(compressed).hexdigest()}
    (WEB/'data/fixed-models/v1/manifest.json').write_bytes(encoded_json(manifest))
    report['totalHoldoutPredictions']=sum(v['holdoutRows'] for v in report['models'].values());assert report['totalHoldoutPredictions']==8236
    report['artifactHashes']={k:v['sha256'] for k,v in manifest['models'].items()};report['fixtureBytes']=len(compressed)
    (WEB/'verification/multi-dataset-phase3fb-native.json').write_bytes(encoded_json(report))
    print('PASS',report['totalHoldoutPredictions'],'predictions; fixture',len(compressed),'bytes',flush=True)

if __name__=='__main__':run()
