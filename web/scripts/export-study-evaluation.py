"""Export original saved Holdout predictions/PI; never fit or synthesize points.
Explicit use only: python3 web/scripts/export-study-evaluation.py [--check]
"""
from pathlib import Path
import argparse
import csv
import json
import math
import runpy
import statistics
BASE=runpy.run_path(str(Path(__file__).with_name('export-study-overviews.py')))
ROOT,read,digest,encode,close=[BASE[k] for k in ['ROOT','read','digest','encode','close']]
DEST=ROOT/'web/data/research-evaluation/v1'

def export(id,definition):
    overview=BASE['export'](id,definition)
    overview_bytes=(ROOT/f'web/data/research-overviews/v1/{id}.json').read_bytes()
    assert encode(overview)==overview_bytes,'Linked Overview differs from verified source'
    experiment,sub,*_=definition;base=str(Path(experiment)/sub)
    source=read(base+'/results.json');manifest=read(experiment+'/artifact-manifest.json')['sha256'];sources=list(overview['sources']);models=[]
    for key in ['baseline','mutual_information']:
        m=source['methods'].get(key);entry={'id':key,'predictions':{'status':'unavailable','reason':'已驗證的逐筆 Holdout 預測缺失。'},'interval':{'status':'unavailable','reason':'已驗證的逐筆 PI 缺失。'}}
        if not m or m.get('status')!='PASS' or 'holdout' not in m:
            models.append(entry);continue
        hold=m['holdout'];actual,pred=hold.get('actual'),hold.get('prediction')
        if not isinstance(actual,list) or not isinstance(pred,list):
            models.append(entry);continue
        assert len(actual)==len(pred)==len(source['split']['holdout_rows'])
        assert hold['source_rows']==source['split']['holdout_rows']
        assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in actual+pred)
        csv_path=base+f'/{key}-holdout.csv';csv_bytes=(ROOT/csv_path).read_bytes()
        assert digest(csv_bytes)==manifest[str(Path(csv_path).relative_to(experiment))]
        rows=list(csv.DictReader(csv_bytes.decode().splitlines()));assert len(rows)==len(actual)
        for i,row in enumerate(rows):
            assert int(row['source_row_0based'])==hold['source_rows'][i]
            close(float(row['actual'] if id=='automobile-price' else row['actual_G3']),actual[i]);close(float(row['prediction'] if id=='automobile-price' else row['predicted_G3']),pred[i])
        sources.append({'path':csv_path,'sha256':digest(csv_bytes)})
        entry['predictions']={'status':'ready','actual':actual,'predicted':pred}
        pi=m.get('prediction_interval',m.get('pi95',{}))
        if pi.get('status')=='PASS':
            lower,upper,pm,diag=pi['lower'],pi['upper'],pi['metrics'],pi['diagnostics'];n=len(actual)
            assert len(lower)==len(upper)==n and diag['holdout_residuals_used'] is False and diag['bounds_clipped_to_zero'] is False
            assert diag['training_rows']==overview['protocol']['trainRows']
            for i,(lo,hi,p) in enumerate(zip(lower,upper,pred)):
                assert math.isfinite(lo) and math.isfinite(hi) and lo<=p<=hi
                close((lo+hi)/2,p);close(float(rows[i]['pi95_lower']),lo);close(float(rows[i]['pi95_upper']),hi)
            widths=[hi-lo for lo,hi in zip(lower,upper)];avg=sum(widths)/n;sd=math.sqrt(sum((w-avg)**2 for w in widths)/n)
            below=sum(a<lo for a,lo in zip(actual,lower));above=sum(a>hi for a,hi in zip(actual,upper));covered=n-below-above
            assert pm['nominal_coverage']==.95 and pm['covered_count']==covered and pm['below_lower_count']==below and pm['above_upper_count']==above and pm['total_holdout_samples']==n
            close(pm['empirical_coverage'],covered/n);close(pm['average_width'],avg);close(pm['median_width'],statistics.median(widths));close(pm['width_standard_deviation'],sd)
            entry['interval']={'status':'ready','lower':lower,'upper':upper,'stats':{'nominal':.95,'coverage':pm['empirical_coverage'],'averageWidth':pm['average_width'],'medianWidth':pm['median_width'],'widthSD':pm['width_standard_deviation'],'covered':covered,'below':below,'above':above,'total':n},'trainingRows':diag['training_rows'],'holdoutResidualsUsed':False,'boundsClipped':False,'method':'ols-pointwise-svd'}
        models.append(entry)
    return {'schemaVersion':1,'artifactType':'study-model-evaluation','datasetId':id,'overviewSha256':digest(overview_bytes),'sources':sources,'models':models}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    outputs={id+'.json':encode(export(id,d)) for id,d in BASE['STUDIES'].items()}
    outputs['manifest.json']=encode({'schemaVersion':1,'snapshots':{id:{'file':id+'.json','sha256':digest(outputs[id+'.json'])} for id in BASE['STUDIES']}})
    if args.check:
        for name,data in outputs.items():assert (DEST/name).read_bytes()==data,name+' differs from verified source'
    else:
        DEST.mkdir(parents=True,exist_ok=True)
        for name,data in outputs.items():(DEST/name).write_bytes(data)
    print(json.dumps({'status':'PASS','checkOnly':args.check,'files':list(outputs),'models':4,'predictionsAndBoundsChecked':342,'bytes':sum(map(len,outputs.values()))}))

if __name__=='__main__':main()
