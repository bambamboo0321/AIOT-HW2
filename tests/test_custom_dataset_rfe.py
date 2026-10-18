import hashlib
import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.custom_dataset_comparison import compare_custom_dataset
from src.custom_dataset_prediction import predict_saved_model, predict_saved_batch
from src.custom_dataset_training import train_custom_dataset, selection_settings
from src.custom_dataset_validation import validate_custom_dataset
from src.data_splitting import split_dataset, prepare_fold, iter_preprocessed_folds
from src.feature_selection import fit_selection
from src.preprocessing import IQRTransformer
from src.preprocessing_config import PreprocessingConfig


class RfeTrainingTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);self.root=Path(directory.name)
        rng=np.random.default_rng(13);x=rng.normal(size=(120,4))*[1,100,.01,10]
        self.frame=pd.DataFrame(x,columns=['x','z','w','v']);self.frame['target']=10+4*x[:,0]+.02*x[:,1]+rng.normal(scale=.1,size=120)
        self.frame['date']=[f'2024-01-{i//24+1:02d}T{i%24:02d}:00' for i in range(120)]
        self.config=PreprocessingConfig(target_column='target',feature_columns=['x','z','w','v'],column_types={c:'numeric' for c in ['x','z','w','v','target']},split_strategy='chronological',split_column='date',cv_splits=3,cv_gap=1)
        self.csv,self.cfg,self.review,self.exp,self.compare=[self.root/name for name in ['data.csv','config.json','review.json','experiment.json','compare.json']];self.write()

    def write(self,k=2):
        self.frame.to_csv(self.csv,index=False);self.cfg.write_text(json.dumps(self.config.to_dict()));sha=hashlib.sha256(self.csv.read_bytes()).hexdigest()
        self.review.write_text(json.dumps({'schema_version':1,'artifact_type':'custom_dataset_review','dataset':{'sha256':sha},'preprocessing':self.config.to_dict(),'review':{'confirmed':True,'training_started':False,'scope':'configuration_only'}}))
        self.validation=validate_custom_dataset(self.csv,self.cfg,self.review);self.assertEqual(self.validation['status'],'valid',self.validation)
        common={'schema_version':1,'confirmed':True,'dataset_sha256':sha,'config_sha256':self.validation['config_sha256']}
        self.exp.write_text(json.dumps({**common,'method':'linear_regression','selection':{'method':'rfe','k':k}}));self.compare.write_text(json.dumps({**common,'methods':['rfe'],'k_values':[k]}))

    def train(self,model=None):return train_custom_dataset(self.csv,self.cfg,self.review,self.exp,model_output=model)

    def test_training_matches_curve_folds_and_independent_final_rfe_with_missing_clip(self):
        for strategy in ['chronological','random']:
            self.config=replace(self.config,split_strategy=strategy,split_column='date' if strategy=='chronological' else None,cv_gap=1 if strategy=='chronological' else 0,outlier_strategy='clip',missing_numeric='median')
            self.frame.loc[[3,29,77,99],'z']=np.nan;self.frame.loc[4,'v']=1000;self.write()
            before={p.name:p.read_bytes() for p in self.root.iterdir()};curve=compare_custom_dataset(self.csv,self.cfg,self.review,self.compare);r=self.train();self.assertEqual(r['status'],'completed',r);self.assertEqual(curve['status'],'compared',curve)
            for actual,f,features in zip(curve['points'][0]['folds'],r['cv']['folds'],r['selection']['fold_features']):
                self.assertEqual(actual['scores'],f['linear_regression']);self.assertEqual(actual['features'],features['features'])
            split=split_dataset(self.frame,self.config);fold=prepare_fold(split.X_train,split.y_train,split.X_test,split.y_test,self.config)
            selector=RFE(LinearRegression(),n_features_to_select=2,step=1).fit(fold.X_train,fold.y_train);idx=selector.get_support(indices=True)
            pred=LinearRegression().fit(fold.X_train[:,idx],fold.y_train).predict(fold.X_validation[:,idx]);self.assertEqual(r['selection']['final_features'],[self.config.feature_columns[int(i)] for i in idx]);np.testing.assert_allclose([p['prediction'] for p in r['preview']],pred,rtol=1e-9,atol=1e-9)
            self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})

    def test_rfe_receives_only_original_current_training_rows_not_validation_or_holdout(self):
        seen=[]
        def capture(X,y,*args,**kwargs):seen.append((X.copy(),y.copy()));return fit_selection(X,y,*args,**kwargs)
        with patch('src.custom_dataset_training.fit_selection',capture):r=self.train()
        self.assertEqual(r['status'],'completed',r);split=split_dataset(self.frame,self.config)
        expected=[f.y_train.index for f in iter_preprocessed_folds(split.X_train,split.y_train,self.config)]+[split.y_train.index]
        for (X,y),idx in zip(seen,expected):
            self.assertTrue(X.index.equals(idx));self.assertTrue(y.index.equals(idx));self.assertFalse(set(X.index)&set(split.X_test.index));pd.testing.assert_frame_equal(X,self.frame.drop(columns='target').loc[idx]);pd.testing.assert_series_equal(y,split.y_train.loc[idx])
        self.assertEqual(len(seen),4)

    def test_changed_holdout_targets_do_not_change_cv_final_selection_or_predictions(self):
        a=self.train();self.frame.loc[96:,'target']+=10000;self.write();b=self.train()
        self.assertEqual(a['cv'],b['cv']);self.assertEqual(a['selection'],b['selection']);self.assertEqual([r['prediction'] for r in a['preview']],[r['prediction'] for r in b['preview']]);self.assertNotEqual(a['holdout'],b['holdout'])

    def test_rfe_bundle_single_and_batch_inference_use_frozen_selected_branches(self):
        model=self.root/'model.joblib';r=self.train(model);self.assertEqual(r['status'],'completed',r);names=r['selection']['final_features'];raw=model.read_bytes()
        values=[{n:float(self.frame.loc[row['source_row']-1,n]) for n in names} for row in r['preview'][:5]]
        with ExitStack() as stack:
            for cls in [RFE,SelectKBest,LinearRegression,SimpleImputer,StandardScaler,IQRTransformer]:stack.enter_context(patch.object(cls,'fit',side_effect=AssertionError('no fit at inference')))
            batch=predict_saved_batch(model,names,[[v[n] for n in names] for v in values])
            for v,row,actual in zip(values,r['preview'],batch['rows']):
                self.assertAlmostEqual(predict_saved_model(model,v)['prediction'],row['prediction'],places=10);self.assertAlmostEqual(actual['prediction'],row['prediction'],places=10)
            partial=predict_saved_model(model,{names[0]:.5});self.assertEqual([f['feature'] for f in partial['imputed']],names[1:])
        self.assertEqual(raw,model.read_bytes())

    def test_k_equals_p_keeps_all_columns_without_rfe_search(self):
        self.write(4)
        with patch('src.custom_dataset_training.fit_selection',side_effect=AssertionError('no ranking')),patch.object(RFE,'fit',side_effect=AssertionError('no ranking')):r=self.train()
        self.assertEqual(r['status'],'completed',r);self.assertFalse(r['selection']['ranking_performed']);self.assertEqual(r['selection']['final_features'],self.config.feature_columns);self.assertEqual(r['holdout']['linear_regression'],r['holdout']['all_features'])

    def test_rfe_scaling_and_cost_constraints_reject_before_any_fit(self):
        self.config=replace(self.config,numerical_scaling='none');self.write()
        with patch.object(LinearRegression,'fit',side_effect=AssertionError('no fit')):r=self.train()
        self.assertEqual(r['status'],'rejected');self.assertIn('StandardScaler',r['errors'][0]['message'])
        self.config=replace(self.config,numerical_scaling='standard');self.write()
        with patch('src.custom_dataset_training.MAX_RFE_WORK',1),patch.object(LinearRegression,'fit',side_effect=AssertionError('no fit')):r=self.train()
        self.assertEqual(r['status'],'rejected');self.assertFalse(r['training_started'])
        exp=json.loads(self.exp.read_text())
        for changed in [{'dataset':{**self.validation['dataset'],'rows':5001}}]:
            with self.assertRaises(ValueError):selection_settings(exp,self.config,{**self.validation,**changed})

    def test_rfe_failure_returns_no_partial_scores_or_private_paths(self):
        with patch('src.custom_dataset_training.fit_selection',side_effect=ValueError('/private/cell-secret')):r=self.train()
        self.assertEqual(r['status'],'failed');self.assertNotIn('holdout',r);self.assertNotIn('/private',json.dumps(r))


if __name__=='__main__':unittest.main()
