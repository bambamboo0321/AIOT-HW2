import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit
from src.custom_dataset_comparison import compare_custom_dataset
from src.custom_dataset_validation import validate_custom_dataset
from src.custom_dataset_training import metrics
from src.data_splitting import split_dataset, prepare_fold
from src.preprocessing_config import PreprocessingConfig
from src.feature_selection import fit_selected_regression


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);self.root=Path(directory.name)
        rng=np.random.default_rng(10);x=rng.normal(size=(80,3));self.frame=pd.DataFrame(x,columns=['x','z','w']);self.frame['target']=4*x[:,0]+x[:,1]+rng.normal(scale=.2,size=80)
        self.frame['date']=[f'2024-01-{i//24+1:02d}T{i%24:02d}:00' for i in range(80)]
        self.config=PreprocessingConfig(target_column='target',feature_columns=['x','z','w'],column_types={c:'numeric' for c in ['x','z','w','target']},split_strategy='random',cv_splits=3,cv_gap=0)
        self.csv=self.root/'data.csv';self.cfg=self.root/'config.json';self.review=self.root/'review.json';self.exp=self.root/'experiment.json';self.write()

    def write(self,changes=None):
        self.frame.to_csv(self.csv,index=False);self.cfg.write_text(json.dumps(self.config.to_dict()))
        sha=hashlib.sha256(self.csv.read_bytes()).hexdigest();self.review.write_text(json.dumps({'schema_version':1,'artifact_type':'custom_dataset_review','dataset':{'sha256':sha},'preprocessing':self.config.to_dict(),'review':{'confirmed':True,'training_started':False,'scope':'configuration_only'}}))
        v=validate_custom_dataset(self.csv,self.cfg,self.review);self.assertEqual(v['status'],'valid',v)
        e={'schema_version':1,'confirmed':True,'dataset_sha256':sha,'config_sha256':v['config_sha256'],'methods':['f_test','mutual_information','rfe'],'k_values':[1,2,3]};e.update(changes or {});self.exp.write_text(json.dumps(e))

    def run_comparison(self):return compare_custom_dataset(self.csv,self.cfg,self.review,self.exp)

    def test_all_points_match_independent_sklearn_fits_with_shared_splits(self):
        r=self.run_comparison();self.assertEqual(r['status'],'compared',r);self.assertEqual(len(r['points']),9)
        split=split_dataset(self.frame,self.config);partitions=list(KFold(3,shuffle=True,random_state=42).split(split.X_train))
        for point in r['points']:
            for f,(train,valid) in zip(point['folds'],partitions):
                fold=prepare_fold(split.X_train.iloc[train],split.y_train.iloc[train],split.X_train.iloc[valid],split.y_train.iloc[valid],self.config)
                k=point['k'];method=point['method'];idx=np.arange(3)
                if k<3:
                    if method=='rfe':s=RFE(LinearRegression(),n_features_to_select=k,step=1)
                    else:s=SelectKBest(f_regression if method=='f_test' else lambda X,y:mutual_info_regression(X,y,random_state=42,n_jobs=1),k=k)
                    s.fit(fold.X_train,fold.y_train);idx=s.get_support(indices=True)
                self.assertEqual(f['features'],[self.config.feature_columns[int(j)] for j in idx])
                scores=metrics(fold.y_validation,LinearRegression().fit(fold.X_train[:,idx],fold.y_train).predict(fold.X_validation[:,idx]))
                np.testing.assert_allclose(list(f['scores'].values()),list(scores.values()),rtol=1e-9,atol=1e-9)
            self.assertEqual(point['ranking_performed'],point['k']<3)
            for name in ['rmse','mae','r2']:
                self.assertAlmostEqual(point['metrics'][name]['mean'],np.mean([f['scores'][name] for f in point['folds']]))
                self.assertAlmostEqual(point['metrics'][name]['std'],np.std([f['scores'][name] for f in point['folds']],ddof=0))
        self.assertEqual(r,self.run_comparison());self.assertNotIn('holdout',r);self.assertNotIn('prediction_model',r)

    def test_holdout_changes_cannot_change_curves_and_every_fit_is_fold_local(self):
        self.config=replace(self.config,split_strategy='chronological',split_column='date',cv_gap=1);self.write()
        original={p.name:p.read_bytes() for p in self.root.iterdir()};seen=[]
        def capture(X,y,*args,**kwargs):seen.append(list(X.index));return fit_selected_regression(X,y,*args,**kwargs)
        with patch('src.custom_dataset_comparison.fit_selected_regression',capture):a=self.run_comparison()
        self.assertEqual(a['status'],'compared',a);self.assertEqual(original,{p.name:p.read_bytes() for p in self.root.iterdir()})
        split=split_dataset(self.frame,self.config);expected=[list(split.X_train.index[t]) for t,_ in TimeSeriesSplit(3,gap=1).split(split.X_train)]
        self.assertEqual(seen,expected*9)
        self.frame.loc[64:,'target']+=10000;self.frame.loc[64:,'x']-=999;self.write();b=self.run_comparison()
        self.assertEqual(a['points'],b['points'])

    def test_full_k_bypasses_search_and_matches_all_methods(self):
        self.write({'k_values':[3]})
        with patch.object(SelectKBest,'fit',side_effect=AssertionError('no search')),patch.object(RFE,'fit',side_effect=AssertionError('no search')):r=self.run_comparison()
        self.assertEqual(r['status'],'compared');self.assertEqual(r['points'][0]['folds'],r['points'][1]['folds']);self.assertEqual(r['points'][0]['folds'],r['points'][2]['folds'])

    def test_rejects_unconfirmed_stale_auto_methods_invalid_k_extra_parameters_before_fit(self):
        changes=[{'confirmed':False},{'schema_version':True},{'dataset_sha256':'0'*64},{'config_sha256':'0'*64},{'methods':['rfecv']},{'methods':['f_test','f_test']},{'methods':[]},{'k_values':[1.5]},{'k_values':[True]},{'k_values':[0]},{'k_values':[4]},{'k_values':[2,1]},{'k_values':[1,1]},{'command':'/private/path'}]
        for change in changes:
            self.write(change)
            with patch.object(LinearRegression,'fit',side_effect=AssertionError('no fit')):r=self.run_comparison()
            self.assertEqual(r['status'],'rejected',change);self.assertFalse(r['comparison_started'])

    def test_requires_standard_and_enforces_work_budget(self):
        self.config=replace(self.config,numerical_scaling='none');self.write();r=self.run_comparison();self.assertEqual(r['status'],'rejected');self.assertIn('standard',r['errors'][0]['message'])
        self.config=replace(self.config,numerical_scaling='standard');self.write()
        with patch('src.custom_dataset_comparison.MAX_WORK',1),patch.object(LinearRegression,'fit',side_effect=AssertionError('no fit')):r=self.run_comparison()
        self.assertEqual(r['status'],'rejected')

    def test_undefined_r2_is_null_not_imputed(self):
        self.config=replace(self.config,split_strategy='chronological',split_column='date',cv_splits=3)
        self.frame.loc[16:31,'target']=2.;self.write({'methods':['f_test'],'k_values':[1]});r=self.run_comparison();self.assertEqual(r['status'],'compared',r)
        self.assertIsNone(r['points'][0]['folds'][0]['scores']['r2']);self.assertEqual(r['points'][0]['metrics']['r2']['defined_folds'],2)

    def test_failure_does_not_return_partial_curves_or_raw_details(self):
        with patch('src.custom_dataset_comparison.fit_selected_regression',side_effect=ValueError('/private/cell-value')):r=self.run_comparison()
        self.assertEqual(r['status'],'failed');self.assertNotIn('points',r);self.assertNotIn('/private',json.dumps(r))


if __name__=='__main__':unittest.main()
