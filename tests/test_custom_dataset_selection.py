import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from src.custom_dataset_validation import validate_custom_dataset
from src.custom_dataset_training import train_custom_dataset
from src.data_splitting import prepare_fold, split_dataset
from src.feature_selection import make_selector, SelectionSpec
from src.preprocessing_config import PreprocessingConfig


class SelectionTrainingTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);self.root=Path(directory.name)
        rng=np.random.default_rng(42);self.X=rng.normal(size=(120,4));self.y=20+10*self.X[:,0]+2*self.X[:,1]+rng.normal(scale=.1,size=120)
        self.frame=pd.DataFrame(self.X,columns=['x0','x1','x2','x3']);self.frame['target']=self.y
        self.frame['date']=[f'2024-01-{i//24+1:02d}T{i%24:02d}:00' for i in range(120)]
        self.config=PreprocessingConfig(target_column='target',feature_columns=['x0','x1','x2','x3'],column_types={c:'numeric' for c in ['x0','x1','x2','x3','target']},split_strategy='chronological',split_column='date',cv_splits=4,cv_gap=1)
        self.csv=self.root/'data.csv';self.cfg=self.root/'config.json';self.review=self.root/'review.json';self.exp=self.root/'experiment.json';self.write()

    def write(self,selection=None):
        self.frame.to_csv(self.csv,index=False);self.cfg.write_text(json.dumps(self.config.to_dict()))
        sha=hashlib.sha256(self.csv.read_bytes()).hexdigest()
        self.review.write_text(json.dumps({'schema_version':1,'artifact_type':'custom_dataset_review','dataset':{'sha256':sha},'preprocessing':self.config.to_dict(),'review':{'confirmed':True,'training_started':False,'scope':'configuration_only'}}))
        v=validate_custom_dataset(self.csv,self.cfg,self.review);self.assertEqual(v['status'],'valid',v)
        e={'schema_version':1,'method':'linear_regression','confirmed':True,'dataset_sha256':sha,'config_sha256':v['config_sha256']}
        if selection is not None:e['selection']=selection
        self.exp.write_text(json.dumps(e))

    def run_experiment(self):return train_custom_dataset(self.csv,self.cfg,self.review,self.exp)

    def test_both_methods_match_independent_selection_fit_and_predictions(self):
        for method in ['f_test','mutual_information']:
            self.write({'method':method,'k':2});before={p.name:p.read_bytes() for p in self.root.iterdir()};r=self.run_experiment();self.assertEqual(r['status'],'completed',r)
            split=split_dataset(self.frame,self.config);fold=prepare_fold(split.X_train,split.y_train,split.X_test,split.y_test,self.config)
            selector=make_selector(SelectionSpec(method=method,k=2,random_state=42),self.config).fit(fold.X_train,fold.y_train)
            indices=selector.get_support(indices=True);pred=LinearRegression().fit(fold.X_train[:,indices],fold.y_train).predict(fold.X_validation[:,indices])
            self.assertEqual(r['selection']['final_features'],[self.config.feature_columns[j] for j in indices])
            np.testing.assert_allclose([p['prediction'] for p in r['preview']],pred,rtol=1e-10,atol=1e-10)
            self.assertEqual(len(r['selection']['fold_features']),4);self.assertTrue(all(len(f['features'])==2 for f in r['selection']['fold_features']))
            self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})

    def test_selectors_fit_only_each_training_partition_not_full_data(self):
        self.write({'method':'f_test','k':2});seen=[];original=SelectKBest.fit
        def capture(selector,X,y,*args,**kwargs):seen.append(list(y));return original(selector,X,y,*args,**kwargs)
        with patch.object(SelectKBest,'fit',capture):r=self.run_experiment()
        self.assertEqual(r['status'],'completed');self.assertEqual([len(y) for y in seen],[f['training_rows'] for f in r['cv']['folds']]+[96])
        holdout=set(self.y[96:]);self.assertTrue(all(not holdout.intersection(y) for y in seen))

    def test_holdout_targets_cannot_change_cv_selected_columns_or_predictions(self):
        for method in ['f_test','mutual_information']:
            self.frame['target']=self.y;self.write({'method':method,'k':2});a=self.run_experiment()
            self.frame.loc[96:,'target']+=10000;self.write({'method':method,'k':2});b=self.run_experiment()
            self.assertEqual(a['cv'],b['cv']);self.assertEqual(a['selection'],b['selection'])
            self.assertEqual([r['prediction'] for r in a['preview']],[r['prediction'] for r in b['preview']])

    def test_none_and_k_equals_p_reuse_all_features_without_claiming_ranking(self):
        for selection in [None,{'method':'none','k':None},{'method':'f_test','k':4},{'method':'mutual_information','k':4}]:
            self.write(selection)
            with patch.object(SelectKBest,'fit',side_effect=AssertionError('not needed')):r=self.run_experiment()
            self.assertEqual(r['status'],'completed');self.assertFalse(r['selection']['ranking_performed'])
            self.assertEqual(r['selection']['final_features'],self.config.feature_columns)
            self.assertEqual(r['holdout']['linear_regression'],r['holdout']['all_features'])
            self.assertTrue(all(p['prediction']==p['baseline_prediction'] for p in r['preview']))

    def test_invalid_method_k_and_extra_selection_parameters_rejected_before_fit(self):
        invalid=[{'method':'f_test','k':k} for k in [0,5,True,1.5,'2',None]]+[{'method':'none','k':2},{'method':'rfecv','k':2},{'method':'f_test','k':2,'command':'bad'}]
        for selection in invalid:
            self.write(selection)
            with patch.object(LinearRegression,'fit',side_effect=AssertionError('must not fit')):r=self.run_experiment()
            self.assertEqual(r['status'],'rejected');self.assertFalse(r['training_started'])

    def test_all_feature_comparator_is_unchanged_across_selection_methods(self):
        self.write();baseline=self.run_experiment()
        for method in ['f_test','mutual_information']:
            self.write({'method':method,'k':1});r=self.run_experiment()
            self.assertEqual(r['holdout']['all_features'],baseline['holdout']['linear_regression'])
            self.assertEqual([f['all_features'] for f in r['cv']['folds']],[f['linear_regression'] for f in baseline['cv']['folds']])
            again=self.run_experiment();self.assertEqual(r,again)

    def test_tiny_cv_training_is_rejected_before_ranking_or_model_fit(self):
        from dataclasses import replace
        self.frame=self.frame.iloc[:8];self.config=replace(self.config,cv_splits=2,cv_gap=0)
        self.write({'method':'mutual_information','k':1})
        with patch.object(LinearRegression,'fit',side_effect=AssertionError('must not fit')):r=self.run_experiment()
        self.assertEqual(r['status'],'rejected');self.assertFalse(r['training_started'])

    def test_selector_failure_never_returns_partial_scores_or_private_errors(self):
        self.write({'method':'f_test','k':2})
        with patch.object(SelectKBest,'fit',side_effect=ValueError('/private/userdata')):r=self.run_experiment()
        self.assertEqual(r['status'],'failed');self.assertNotIn('holdout',r);self.assertNotIn('/private',json.dumps(r))


if __name__=='__main__':unittest.main()
