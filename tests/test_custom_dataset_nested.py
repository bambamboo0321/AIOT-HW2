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
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from src.custom_dataset_validation import validate_custom_dataset
from src.custom_dataset_training import train_custom_dataset
from src.custom_dataset_comparison import compare_custom_dataset
from src.custom_dataset_prediction import predict_saved_model, predict_saved_batch
from src.custom_dataset_selectors import fit_custom_selection, inner_partitions, ALPHAS
from src.feature_selection import NumericFoldRegressor, SelectionSpec, fit_preprocessor
from src.data_splitting import split_dataset
from src.preprocessing_config import PreprocessingConfig


class NestedSelectionTests(unittest.TestCase):
    def setUp(self):
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup);self.root=Path(d.name)
        rng=np.random.default_rng(42);x=rng.normal(size=(96,3))*[1,100,.01];self.frame=pd.DataFrame(x,columns=['x','z','w']);self.frame['target']=20+4*x[:,0]+.02*x[:,1]+rng.normal(scale=.1,size=96)
        self.frame['date']=[f'2024-01-{i//24+1:02d}T{i%24:02d}:00' for i in range(96)]
        self.config=PreprocessingConfig(target_column='target',feature_columns=['x','z','w'],column_types={n:'numeric' for n in ['x','z','w','target']},cv_splits=3)
        self.csv,self.cfg,self.review,self.exp,self.cmp=[self.root/n for n in ['data.csv','config.json','review.json','experiment.json','comparison.json']]

    def write(self,method='rfecv',k=None,methods=None,ks=None):
        self.frame.to_csv(self.csv,index=False);self.cfg.write_text(json.dumps(self.config.to_dict()));sha=hashlib.sha256(self.csv.read_bytes()).hexdigest()
        self.review.write_text(json.dumps({'schema_version':1,'artifact_type':'custom_dataset_review','dataset':{'sha256':sha},'preprocessing':self.config.to_dict(),'review':{'confirmed':True,'training_started':False,'scope':'configuration_only'}}))
        v=validate_custom_dataset(self.csv,self.cfg,self.review);self.assertEqual(v['status'],'valid',v);common={'schema_version':1,'confirmed':True,'dataset_sha256':sha,'config_sha256':v['config_sha256']}
        self.exp.write_text(json.dumps({**common,'method':'linear_regression','selection':{'method':method,'k':k}}));self.cmp.write_text(json.dumps({**common,'methods':methods or [method],'k_values':ks if ks is not None else [k] if k is not None else []}))

    def train(self,model=None):return train_custom_dataset(self.csv,self.cfg,self.review,self.exp,model_output=model)
    def compare(self):return compare_custom_dataset(self.csv,self.cfg,self.review,self.cmp)

    def test_sfs_and_rfecv_support_match_independent_library_and_shared_outer_comparison(self):
        for strategy in ['random','chronological']:
            self.config=replace(self.config,split_strategy=strategy,split_column='date' if strategy=='chronological' else None,cv_gap=1 if strategy=='chronological' else 0,outlier_strategy='clip')
            self.frame.loc[[2,31,50],'z']=np.nan;self.frame.loc[3,'w']=1000
            for method,k in [('forward_sfs',2),('rfecv',None)]:
                self.write(method,k);r=self.train();c=self.compare();self.assertEqual(r['status'],'completed',r);self.assertEqual(c['status'],'compared',c)
                point=c['points'][0] if k else c['automatic'][0]
                for f,t,n in zip(point['folds'],r['cv']['folds'],r['selection']['fold_features']):np.testing.assert_allclose(list(f['scores'].values()),list(t['linear_regression'].values()),rtol=1e-10,atol=1e-10);self.assertEqual(f['features'],n['features']);self.assertEqual(f['details'],n['details'])
                split=split_dataset(self.frame,self.config);cv=inner_partitions(len(split.X_train),self.config);est=NumericFoldRegressor(self.config)
                sel=(SequentialFeatureSelector(est,n_features_to_select=2,direction='forward',scoring='neg_root_mean_squared_error',cv=cv,n_jobs=1) if k else RFECV(est,step=1,min_features_to_select=1,scoring='neg_root_mean_squared_error',cv=cv,importance_getter='coef_',n_jobs=1))
                sel.fit(split.X_train[self.config.feature_columns].to_numpy(),split.y_train);idx=sel.get_support(indices=True);self.assertEqual(r['selection']['final_features'],[self.config.feature_columns[int(i)] for i in idx])

    def test_lassocv_alpha_uses_independently_prepared_inner_blocks_and_final_support_refit(self):
        for strategy in ['random','chronological']:
            self.config=replace(self.config,split_strategy=strategy,split_column='date' if strategy=='chronological' else None,cv_gap=1 if strategy=='chronological' else 0,outlier_strategy='clip')
            self.frame.loc[[2,31],'z']=np.nan;self.write('lasso_cv');split=split_dataset(self.frame,self.config);X,y=split.X_train,split.y_train
            seen=[]
            def capture(X,y,c):seen.append(list(X.index));return fit_preprocessor(X,y,c)
            with patch('src.custom_dataset_selectors.fit_preprocessor',capture):selected=fit_custom_selection(X,y,self.config,SelectionSpec('lasso_cv'))
            parts=inner_partitions(len(X),self.config);self.assertEqual(seen,[list(X.index[t]) for t,_ in parts]+[list(X.index)])
            mean_errors=[]
            for alpha in ALPHAS:
                errors=[]
                for train,valid in parts:
                    prep,values=fit_preprocessor(X.iloc[train],y.iloc[train],self.config);pred=Lasso(alpha=alpha,max_iter=10000,tol=1e-5).fit(values,y.iloc[train]).predict(prep.transform(X.iloc[valid]));errors.append(np.mean((pred-y.iloc[valid])**2))
                mean_errors.append(np.mean(errors))
            chosen=selected.metadata['custom_details']['alpha'];best=ALPHAS[int(np.argmin(mean_errors))];self.assertEqual(chosen,best)
            prep,values=fit_preprocessor(X,y,self.config);support=np.flatnonzero(np.abs(Lasso(alpha=chosen,max_iter=10000,tol=1e-5).fit(values,y).coef_)>1e-8);np.testing.assert_array_equal(selected.indices,support)
            r=self.train();c=self.compare();self.assertEqual(r['status'],'completed',r);self.assertEqual(c['status'],'compared',c)
            for f,t in zip(c['automatic'][0]['folds'],r['cv']['folds']):np.testing.assert_allclose(list(f['scores'].values()),list(t['linear_regression'].values()),rtol=1e-10,atol=1e-10)

    def test_inner_estimators_never_fit_outer_validation_or_holdout_targets(self):
        self.config=replace(self.config,split_strategy='chronological',split_column='date',cv_gap=1)
        original=NumericFoldRegressor.fit
        for method,k in [('forward_sfs',1),('rfecv',None)]:
            self.write(method,k);seen=[]
            def capture(est,X,y):seen.append(set(np.asarray(y)));return original(est,X,y)
            with patch.object(NumericFoldRegressor,'fit',capture):r=self.train()
            self.assertEqual(r['status'],'completed',r);split=split_dataset(self.frame,self.config);self.assertTrue(all(not s.intersection(split.y_test) for s in seen))
            # First outer Training is strictly the first chronological block.
            first=set(split.y_train.iloc[:18]);self.assertTrue(seen[0].issubset(first))

    def test_holdout_labels_cannot_change_nested_cv_support_alpha_or_predictions(self):
        self.config=replace(self.config,split_strategy='chronological',split_column='date',cv_gap=1)
        original=self.frame['target'].copy()
        for method,k in [('forward_sfs',2),('rfecv',None),('lasso_cv',None)]:
            self.frame['target']=original;self.write(method,k);a=self.train();self.frame.loc[76:,'target']+=10000;self.write(method,k);b=self.train();self.assertEqual(a['status'],'completed',a);self.assertEqual(b['status'],'completed',b);self.assertEqual(a['selection'],b['selection']);self.assertEqual(a['cv'],b['cv']);self.assertEqual([p['prediction'] for p in a['preview']],[p['prediction'] for p in b['preview']])

    def test_all_three_models_single_and_batch_inference_without_any_selector_or_regressor_fit(self):
        for method,k in [('forward_sfs',2),('rfecv',None),('lasso_cv',None)]:
            self.write(method,k);path=self.root/(method+'.joblib');r=self.train(path);self.assertEqual(r['status'],'completed',r);names=r['selection']['final_features'];inputs=[{n:float(self.frame.loc[row['source_row']-1,n]) for n in names} for row in r['preview'][:3]]
            with ExitStack() as stack:
                for cls in [SequentialFeatureSelector,RFECV,LassoCV,Lasso,LinearRegression]:stack.enter_context(patch.object(cls,'fit',side_effect=AssertionError('must not fit during inference')))
                batch=predict_saved_batch(path,names,[[v[n] for n in names] for v in inputs])
                for row,values,pred in zip(r['preview'],inputs,batch['rows']):self.assertAlmostEqual(predict_saved_model(path,values)['prediction'],row['prediction'],places=9);self.assertAlmostEqual(pred['prediction'],row['prediction'],places=9)

    def test_zero_feature_lasso_is_explicit_no_scores_in_comparison_and_no_model_on_formal_failure(self):
        self.frame['target']=100+np.random.default_rng(8).normal(scale=1e-6,size=len(self.frame));self.write('lasso_cv');c=self.compare();self.assertEqual(c['status'],'compared',c);point=c['automatic'][0];self.assertTrue(all(not f['features'] and f['scores'] is None for f in point['folds']));self.assertIsNone(point['metrics']['rmse']['mean']);self.assertEqual(point['metrics']['rmse']['defined_folds'],0)
        path=self.root/'zero.joblib';r=self.train(path);self.assertEqual(r['status'],'failed');self.assertIn('0 個特徵',r['errors'][0]['message']);self.assertNotIn('holdout',r);self.assertFalse(path.exists())

    def test_full_sfs_and_single_feature_rfecv_bypass_search_without_claiming_ranking(self):
        for method,k in [('forward_sfs',3),('rfecv',None)]:
            if method=='rfecv':self.config=replace(self.config,feature_columns=['x'],column_types={'x':'numeric','target':'numeric'})
            self.write(method,k)
            with patch.object(RFECV,'fit',side_effect=AssertionError('no search')),patch.object(SequentialFeatureSelector,'fit',side_effect=AssertionError('no search')):r=self.train()
            self.assertEqual(r['status'],'completed',r);self.assertFalse(r['selection']['ranking_performed']);self.assertTrue(r['selection']['details']['search_bypassed']);self.assertEqual(r['holdout']['linear_regression'],r['holdout']['all_features'])

    def test_invalid_fixed_k_for_auto_and_small_inner_splits_rejected_before_fit(self):
        for method,k in [('rfecv',2),('lasso_cv',1),('forward_sfs',None)]:
            self.write(method,k)
            with patch.object(LinearRegression,'fit',side_effect=AssertionError('no fit')):r=self.train()
            self.assertEqual(r['status'],'rejected')
        self.config=replace(self.config,split_strategy='chronological',split_column='date',cv_gap=4,cv_splits=5);self.write('rfecv')
        with patch.object(LinearRegression,'fit',side_effect=AssertionError('no fit')):r=self.train()
        self.assertEqual(r['status'],'rejected');self.assertIn('內部',r['errors'][0]['message'])

    def test_mixed_comparison_has_fixed_sfs_points_and_separate_auto_results(self):
        self.write('rfecv',methods=['forward_sfs','rfecv','lasso_cv'],ks=[1,2,3]);before={p.name:p.read_bytes() for p in self.root.iterdir()};r=self.compare();self.assertEqual(r['status'],'compared',r);self.assertEqual(len(r['points']),3);self.assertEqual([p['method'] for p in r['automatic']],['rfecv','lasso_cv']);self.assertTrue(all(p['method']=='forward_sfs' for p in r['points']));self.assertNotIn('holdout',r);self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})

    def test_nested_scaling_and_sfs_column_limits_reject_before_fit(self):
        self.config=replace(self.config,numerical_scaling='none');self.write('forward_sfs',2)
        with patch.object(LinearRegression,'fit',side_effect=AssertionError('no fit')):r=self.train()
        self.assertEqual(r['status'],'rejected');self.assertIn('StandardScaler',r['errors'][0]['message'])
        rng=np.random.default_rng(5);names=[f'f{i}' for i in range(16)]
        for n in names:self.frame[n]=rng.normal(size=len(self.frame))
        self.config=replace(self.config,numerical_scaling='standard',feature_columns=names,column_types={n:'numeric' for n in names+['target']});self.write('forward_sfs',2)
        with patch.object(LinearRegression,'fit',side_effect=AssertionError('no fit')):r=self.train()
        self.assertEqual(r['status'],'rejected');self.assertIn('15',r['errors'][0]['message'])


if __name__=='__main__':unittest.main()
