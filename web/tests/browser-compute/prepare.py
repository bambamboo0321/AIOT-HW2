"""Reuses B4.6 reference; creates additional deterministic input/refs with production entry points."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timedelta
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.custom_dataset_training import train_custom_dataset
from src.custom_dataset_validation import validate_custom_dataset
from src.custom_dataset_comparison import compare_custom_dataset
from src.custom_dataset_prediction import predict_saved_model, predict_saved_batch
OUT=ROOT/'web/tests/fixtures/browser-compute'
METHODS=['f_test','mutual_information','forward_sfs','rfe','lasso_cv','rfecv']
def put(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def case(name,text,config):
    sha=hashlib.sha256(text.encode()).hexdigest()
    review=dict(schema_version=1,artifact_type='custom_dataset_review',dataset=dict(sha256=sha),preprocessing=config,
        review=dict(confirmed=True,training_started=False,scope='configuration_only'))
    item=dict(name=name,csv=text,config=config,review=review,references=[])
    with tempfile.TemporaryDirectory(dir=ROOT/'web/artifacts') as directory:
        d=Path(directory);(d/'input.csv').write_text(text);(d/'config.json').write_text(json.dumps(config));(d/'review.json').write_text(json.dumps(review))
        validation=validate_custom_dataset(d/'input.csv',d/'config.json',d/'review.json');assert validation['status']=='valid',validation
        item['validation']=validation
        parsed=list(csv.DictReader(io.StringIO(text)))
        if name=='baseline':
            old=json.loads((ROOT/'experiments/m17-b4-6/native-reference.json').read_text())
            item['references']=[r['method']['production'] for r in old['results']]
            item['reference_environment']=old['environment']
        else:
            for method in METHODS:
                experiment=dict(schema_version=1,method='linear_regression',confirmed=True,dataset_sha256=sha,
                    config_sha256=validation['config_sha256'],selection=dict(method=method,k=None if method in ('lasso_cv','rfecv') else 2))
                (d/'experiment.json').write_text(json.dumps(experiment));model=d/(method+'.joblib')
                report=train_custom_dataset(d/'input.csv',d/'config.json',d/'review.json',d/'experiment.json',model_output=model)
                assert report['status']=='completed',(name,method,report)
                names=report['selection']['final_features'];row=parsed[report['preview'][0]['source_row']-1]
                values={n:float(row[n]) if row[n] else None for n in names};values.update({n:None for n in names[1:]})
                rows=[[float(row[n]) if row[n] else None for n in names],[values[n] for n in names]]
                item.setdefault('inference',{})[method]=dict(values=values,rows=rows,single=predict_saved_model(model,values),batch=predict_saved_batch(model,names,rows))
                item['references'].append(report)
        methods=['f_test','mutual_information','rfe','forward_sfs','rfecv','lasso_cv']
        ks=[1,3,5] if name=='baseline' else [1,2,3]
        experiment=dict(schema_version=1,confirmed=True,dataset_sha256=sha,config_sha256=validation['config_sha256'],methods=methods,k_values=ks)
        (d/'experiment.json').write_text(json.dumps(experiment));report=compare_custom_dataset(d/'input.csv',d/'config.json',d/'review.json',d/'experiment.json')
        assert report['status']=='compared',report
        item['comparison_experiment']=experiment;item['comparison_reference']=report
    return item

def main():
    OUT.mkdir(parents=True,exist_ok=True);(ROOT/'web/artifacts').mkdir(exist_ok=True)
    b=ROOT/'experiments/m17-b4-6';cfg=json.loads((b/'config.json').read_text())
    inputs=[case('baseline',(b/'fixture.csv').read_text(),cfg)]
    b2=ROOT/'experiments/m17-b2';cfg=json.loads((b2/'config.json').read_text());rows=list(csv.reader(io.StringIO((b2/'fixture.csv').read_text())))
    for i,row in enumerate(rows[1:]):
        if i%11==3:row[1]=''
        if i%17==4:row[2]=''
    buf=io.StringIO();csv.writer(buf,lineterminator='\n').writerows(rows);inputs.append(case('missing',buf.getvalue(),cfg))
    rows[0].append('date')
    for i,row in enumerate(rows[1:]):row.append((datetime(2020,1,1)+timedelta(minutes=10*i)).isoformat())
    cfg=dict(cfg,split_strategy='chronological',split_column='date')
    buf=io.StringIO();csv.writer(buf,lineterminator='\n').writerows(rows);inputs.append(case('chronological_missing',buf.getvalue(),cfg))
    put('cases.json',dict(rtol=1e-7,atol=1e-9,source='current production src; baseline reference reused read-only from B4.6',cases=inputs))
    print(json.dumps(dict(cases=[c['name'] for c in inputs],training_references=18,comparison_references=3)))
if __name__=='__main__':main()
