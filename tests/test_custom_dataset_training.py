import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing_config import PreprocessingConfig
from src.custom_dataset_validation import validate_custom_dataset
from src.custom_dataset_training import train_custom_dataset
from src.preprocessing import build_preprocessor


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv = self.root/'data.csv'; self.config = self.root/'config.json'
        self.review = self.root/'review.json'; self.experiment = self.root/'experiment.json'
        self.options = PreprocessingConfig(target_column='target', feature_columns=['x'], column_types={'x':'numeric','target':'numeric'}, split_strategy='chronological', split_column='date', cv_splits=3)
        self.rows = [(f'2024-01-{i//24+1:02d}T{i%24:02d}:00', i, 3*i+2+(i%3)*.2) for i in range(80)]
        self.write()

    def write(self):
        self.csv.write_text('date,x,target\n'+'\n'.join(','.join(map(str,r)) for r in self.rows),encoding='utf8')
        self.config.write_text(json.dumps(self.options.to_dict()))
        sha=hashlib.sha256(self.csv.read_bytes()).hexdigest()
        self.review.write_text(json.dumps({'schema_version':1,'artifact_type':'custom_dataset_review','dataset':{'sha256':sha},'preprocessing':self.options.to_dict(),'review':{'confirmed':True,'training_started':False,'scope':'configuration_only'}}))
        report=validate_custom_dataset(self.csv,self.config,self.review)
        self.experiment.write_text(json.dumps({'schema_version':1,'method':'linear_regression','confirmed':True,'dataset_sha256':sha,'config_sha256':report.get('config_sha256','0'*64)}))

    def run_training(self):
        return train_custom_dataset(self.csv,self.config,self.review,self.experiment)

    def test_real_scores_match_independent_fit_and_mean_baseline(self):
        before={p.name:p.read_bytes() for p in self.root.iterdir()}
        result=self.run_training(); self.assertEqual(result['status'],'completed')
        x=np.arange(80).reshape(-1,1); y=np.array([r[2] for r in self.rows])
        pred=LinearRegression().fit(x[:64],y[:64]).predict(x[64:]); actual=y[64:]
        expected={'rmse':float(np.sqrt(mean_squared_error(actual,pred))),'mae':float(mean_absolute_error(actual,pred)),'r2':float(r2_score(actual,pred))}
        for key in expected:self.assertAlmostEqual(result['holdout']['linear_regression'][key],expected[key],places=10)
        self.assertEqual(result['preview'][0]['source_row'],65)
        self.assertAlmostEqual(result['preview'][0]['training_mean'],np.mean(y[:64]))
        self.assertEqual({p.name:p.read_bytes() for p in self.root.iterdir()},before)
        self.assertEqual(len(result['cv']['folds']),3)
        json.dumps(result,allow_nan=False)

    def test_holdout_does_not_change_cv_or_predictions_and_preprocessors_fit_per_fold(self):
        first=self.run_training(); seen=[]
        import src.data_splitting as splitting
        original=splitting.build_preprocessor
        def capture(training,config,**kwargs):
            seen.append(list(training.index)); return original(training,config,**kwargs)
        self.rows=[(d,x,y if i<64 else y+10000) for i,(d,x,y) in enumerate(self.rows)];self.write()
        with patch.object(splitting,'build_preprocessor',side_effect=capture):second=self.run_training()
        self.assertEqual(first['cv'],second['cv'])
        self.assertEqual([r['prediction'] for r in first['preview']],[r['prediction'] for r in second['preview']])
        self.assertEqual(len(seen),4);self.assertTrue(all(max(indices)<64 for indices in seen))
        self.assertEqual([len(indices) for indices in seen],[16,32,48,64])

    def test_missing_confirmation_stale_digest_and_invalid_data_never_fit(self):
        for alteration in ['not_confirmed','method','digest','extra','invalid_data']:
            self.write();v=json.loads(self.experiment.read_text())
            if alteration=='not_confirmed':v['confirmed']=False
            elif alteration=='method':v['method']='random_forest'
            elif alteration=='digest':v['config_sha256']='0'*64
            elif alteration=='extra':v['command']='fit'
            else:self.csv.write_text('x,target\n1,wrong')
            self.experiment.write_text(json.dumps(v))
            with patch.object(LinearRegression,'fit',side_effect=AssertionError('must not fit')):
                r=self.run_training()
            self.assertEqual(r['status'],'rejected');self.assertFalse(r['training_started'])

    def test_random_split_missing_features_clip_and_target_drop(self):
        self.options=PreprocessingConfig(target_column='target',feature_columns=['x'],column_types={'x':'numeric','target':'numeric'},missing_target='drop',outlier_strategy='clip',split_strategy='random',cv_splits=3)
        self.rows=[(d,'' if i%9==0 else x,'' if i==2 else y) for i,(d,x,y) in enumerate(self.rows)];self.write()
        a=self.run_training();b=self.run_training();self.assertEqual(a,b)
        self.assertEqual(a['status'],'completed');self.assertEqual(a['validation']['split']['target_missing_dropped'],1)
        self.assertTrue(all(p['timestamp'] is None for p in a['preview']))
        self.assertEqual(a['holdout']['rows'],16)

    def test_constant_holdout_r2_is_null_and_short_preview_preserved(self):
        self.rows=[(d,x,y if i<64 else 10) for i,(d,x,y) in enumerate(self.rows)];self.write()
        r=self.run_training();self.assertEqual(r['status'],'completed')
        self.assertIsNone(r['holdout']['linear_regression']['r2']);self.assertIsNone(r['holdout']['training_mean']['r2'])
        self.assertEqual(len(r['preview']),16);json.dumps(r,allow_nan=False)

    def test_preview_is_bounded_but_scores_use_all_holdout_rows(self):
        self.rows=[(f'2024-01-{i//1440+1:02d}T{i//60%24:02d}:{i%60:02d}',i,2*i+(i%7)) for i in range(900)]
        self.write();r=self.run_training();self.assertEqual(r['status'],'completed')
        self.assertEqual(r['holdout']['rows'],180);self.assertEqual(len(r['preview']),144)
        actual=np.array([row[2] for row in self.rows[720:]])
        x=np.arange(900).reshape(-1,1);pred=LinearRegression().fit(x[:720],np.array([r[2] for r in self.rows[:720]])).predict(x[720:])
        self.assertAlmostEqual(r['holdout']['linear_regression']['mae'],mean_absolute_error(actual,pred))

    def test_fit_failure_produces_no_partial_metrics_or_private_data(self):
        with patch.object(LinearRegression,'fit',side_effect=ValueError('/secret/raw-user-cell')):r=self.run_training()
        self.assertEqual(r['status'],'failed');self.assertTrue(r['training_started']);self.assertNotIn('holdout',r)
        self.assertNotIn('/secret',json.dumps(r))

    def test_fold_local_imputation_and_scaling_ignore_future_extremes(self):
        frame=__import__('pandas').DataFrame({'x':[1.,np.nan,3.,10000.]})
        p=build_preprocessor(frame.iloc[:3],self.options);p.fit(frame.iloc[:3])
        branch=p.named_steps['columns'].named_transformers_['c0']
        self.assertEqual(branch.named_steps['impute'].statistics_[0],2.)
        self.assertEqual(branch.named_steps['scale'].mean_[0],2.)
        self.assertEqual(branch.named_steps['outliers'].upper_[0],4.)


if __name__=='__main__':unittest.main()
