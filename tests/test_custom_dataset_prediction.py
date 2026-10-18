import hashlib
import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.custom_dataset_prediction import predict_saved_model, predict_saved_batch
from src.custom_dataset_training import train_custom_dataset
from src.custom_dataset_validation import validate_custom_dataset
from src.preprocessing import IQRTransformer
from src.preprocessing_config import PreprocessingConfig


class PredictionTests(unittest.TestCase):
    def setUp(self):
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup);self.root=Path(d.name)
        rng=np.random.default_rng(12);x=rng.normal(size=(120,3))
        self.frame=pd.DataFrame(x,columns=['x','y','z']);self.frame['target']=10+3*x[:,0]+2*x[:,1]+rng.normal(scale=.1,size=120)
        self.config=PreprocessingConfig(target_column='target',feature_columns=['x','y','z'],column_types={n:'numeric' for n in ['x','y','z','target']},cv_splits=3)
        self.model=self.root/'model.joblib'

    def train(self,method='none',k=None):
        paths=[self.root/n for n in ['data.csv','config.json','review.json','experiment.json']]
        csv,cfg,review,exp=paths;self.frame.to_csv(csv,index=False);cfg.write_text(json.dumps(self.config.to_dict()))
        sha=hashlib.sha256(csv.read_bytes()).hexdigest()
        review.write_text(json.dumps({'schema_version':1,'artifact_type':'custom_dataset_review','dataset':{'sha256':sha},'preprocessing':self.config.to_dict(),'review':{'confirmed':True,'training_started':False,'scope':'configuration_only'}}))
        validation=validate_custom_dataset(csv,cfg,review);self.assertEqual(validation['status'],'valid',validation)
        exp.write_text(json.dumps({'schema_version':1,'method':'linear_regression','confirmed':True,'dataset_sha256':sha,'config_sha256':validation['config_sha256'],'selection':{'method':method,'k':k}}))
        before=[p.read_bytes() for p in paths]
        result=train_custom_dataset(*paths,model_output=self.model)
        self.assertEqual([p.read_bytes() for p in paths],before)
        return result

    def test_predictions_match_real_holdout_for_all_methods_without_any_fit(self):
        for method,k in [('none',None),('f_test',2),('mutual_information',2)]:
            self.model.unlink(missing_ok=True);r=self.train(method,k);self.assertEqual(r['status'],'completed',r)
            with ExitStack() as stack:
                for cls in [SelectKBest,SimpleImputer,LinearRegression,StandardScaler,IQRTransformer]:
                    stack.enter_context(patch.object(cls,'fit',side_effect=AssertionError('inference must not fit')))
                for row in r['preview'][:4]:
                    values={n:float(self.frame.iloc[row['source_row']-1][n]) for n in r['selection']['final_features']}
                    pred=predict_saved_model(self.model,values)
                    self.assertAlmostEqual(pred['prediction'],row['prediction'],places=10)
                    self.assertFalse(pred['training_started']);self.assertEqual(pred['imputed'],[])

    def test_partial_inputs_use_frozen_training_mean_or_median_after_clipping(self):
        for policy in ['mean','median']:
            self.model.unlink(missing_ok=True);self.config=replace(self.config,missing_numeric=policy,outlier_strategy='clip')
            r=self.train();self.assertEqual(r['status'],'completed')
            saved=joblib.load(self.model);v={'x':.5}
            pred=predict_saved_model(self.model,v)
            expected=[]
            for b in saved['branches']:
                value=v.get(b['name'],np.nan);expected.append(b['transformer'].transform(pd.DataFrame({b['name']:[value]})))
            self.assertAlmostEqual(pred['prediction'],saved['model'].predict(np.concatenate(expected,axis=1))[0])
            self.assertEqual([row['feature'] for row in pred['imputed']],['y','z'])
            for row in pred['imputed']:
                b=next(b for b in saved['branches'] if b['name']==row['feature']);self.assertEqual(row['value'],b['transformer'].named_steps['impute'].statistics_[0])
            self.assertEqual(saved['model'].n_features_in_,3)

    def test_none_requires_selected_only_not_discarded_candidates(self):
        self.config=replace(self.config,missing_numeric='none');r=self.train('f_test',2);self.assertEqual(r['status'],'completed')
        names=r['selection']['final_features'];pred=predict_saved_model(self.model,{n:1. for n in names});self.assertTrue(np.isfinite(pred['prediction']))
        with self.assertRaises(ValueError):predict_saved_model(self.model,{names[0]:1.})

    def test_unknown_target_nonfinite_boolean_unsafe_and_empty_inputs_rejected(self):
        self.train()
        for values in [{},{'x':None},{'target':1},{'x':1,'other':2},{'x':True},{'x':'1'},{'x':float('inf')},{'x':float('nan')},{'x':2**53}]:
            with self.subTest(values=values),self.assertRaises(ValueError):predict_saved_model(self.model,values)

    def test_extrapolation_is_reported_and_prediction_is_not_clamped(self):
        self.train();result=predict_saved_model(self.model,{'x':-1000.,'y':-1000.,'z':0.})
        self.assertEqual(result['out_of_range'],['x','y']);self.assertLess(result['prediction'],0);self.assertGreaterEqual(len(result['warnings']),2)

    def test_explicit_model_output_never_overwrites_an_existing_file(self):
        self.model.write_bytes(b'existing research artifact');r=self.train();self.assertEqual(r['status'],'failed');self.assertNotIn('holdout',r);self.assertEqual(self.model.read_bytes(),b'existing research artifact')

    def test_bundle_has_no_training_rows_holdout_targets_or_csv(self):
        self.train('f_test',1);b=joblib.load(self.model)
        self.assertEqual(set(b),{'schema_version','target','missing_numeric','branches','model','dataset_sha256','config_sha256'})
        self.assertEqual(len(b['branches']),1);self.assertEqual(set(b['branches'][0]),{'name','transformer','range'})
        self.assertEqual(self.model.stat().st_mode&0o777,0o600)


    def test_batch_matches_holdout_for_all_methods_without_fit_and_loads_once(self):
        for method,k in [('none',None),('f_test',2),('mutual_information',2)]:
            self.model.unlink(missing_ok=True);r=self.train(method,k);names=r['selection']['final_features']
            rows=[[float(self.frame.iloc[p['source_row']-1][name]) for name in names] for p in r['preview']]
            with ExitStack() as stack:
                for cls in [SelectKBest,SimpleImputer,LinearRegression,StandardScaler,IQRTransformer]:
                    stack.enter_context(patch.object(cls,'fit',side_effect=AssertionError('batch must not fit')))
                load=stack.enter_context(patch('src.custom_dataset_prediction.joblib.load',wraps=joblib.load))
                result=predict_saved_batch(self.model,names,rows);self.assertEqual(load.call_count,1)
            np.testing.assert_allclose([p['prediction'] for p in result['rows']],[p['prediction'] for p in r['preview']],rtol=1e-12,atol=1e-12)
            self.assertFalse(result['training_started']);self.assertEqual(result['row_count'],len(rows));self.assertEqual([p['position'] for p in result['rows']],list(range(1,len(rows)+1)))

    def test_batch_frozen_mean_median_clipping_matches_single_without_model_changes(self):
        for policy in ['mean','median']:
            self.model.unlink(missing_ok=True);self.config=replace(self.config,missing_numeric=policy,outlier_strategy='clip');self.train()
            before=self.model.read_bytes();rows=[[None,.3,4.],[1.,None,None],[-100.,1.,0.]];names=['x','y','z']
            result=predict_saved_batch(self.model,names,rows)
            for i,row in enumerate(rows):
                single=predict_saved_model(self.model,dict(zip(names,row)))
                self.assertAlmostEqual(result['rows'][i]['prediction'],single['prediction'],places=10)
                self.assertEqual(result['rows'][i]['imputed_features'],[r['feature'] for r in single['imputed']])
                self.assertEqual(result['rows'][i]['out_of_range_features'],single['out_of_range'])
                fills={r['feature']:r['value'] for r in result['imputation_values']}
                self.assertTrue(all(fills[r['feature']]==r['value'] for r in single['imputed']))
            self.assertEqual(self.model.read_bytes(),before)

    def test_batch_invalid_row_rejects_whole_matrix_and_feature_order(self):
        self.train();names=['x','y','z']
        invalid=[[],[[None,None,None]],[[1.,2.]],[[1.,2.,3.,4.]],[[1.,2.,True]],[[1.,2.,float('inf')]],[[1.,2.,'3']],[[1.,2.,2**53]],[[1.,2.,3.]]*501,[[1.,2.,3.],[1.,2.,float('nan')]]]
        for rows in invalid:
            with self.subTest(rows=str(rows)[:80]),self.assertRaises(ValueError):predict_saved_batch(self.model,names,rows)
        with self.assertRaises(ValueError):predict_saved_batch(self.model,['y','x','z'],[[1.,2.,3.]])

    def test_batch_none_requires_complete_selected_features_only(self):
        self.config=replace(self.config,missing_numeric='none');r=self.train('f_test',2);names=r['selection']['final_features']
        self.assertEqual(predict_saved_batch(self.model,names,[[1.,2.]])['row_count'],1)
        with self.assertRaises(ValueError):predict_saved_batch(self.model,names,[[1.,None]])

    def test_batch_reports_negative_extrapolation_without_clamping(self):
        self.train();r=predict_saved_batch(self.model,['x','y','z'],[[-1000.,-1000.,0.]])
        self.assertLess(r['rows'][0]['prediction'],0);self.assertEqual(r['rows'][0]['out_of_range_features'],['x','y']);self.assertTrue(r['warnings'])


if __name__=='__main__':unittest.main()
