import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from src.custom_dataset_comparison import compare_custom_dataset
from src.custom_dataset_validation import validate_custom_dataset
from src.custom_dataset_selectors import fit_custom_selection, inner_partitions
from src.custom_sfs_comparison import SFSPrefixCache
from src.feature_selection import NumericFoldRegressor, SelectionSpec, count_fits, fit_preprocessor
from src.preprocessing_config import PreprocessingConfig


class SFSPrefixTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rng = np.random.default_rng(42)
        x = rng.normal(size=(150, 4))
        self.frame = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])
        self.frame['target'] = 3*x[:, 0]-2*x[:, 1]+rng.normal(scale=.1, size=150)
        self.config = PreprocessingConfig(target_column='target', feature_columns=['a','b','c','d'],
            column_types={n:'numeric' for n in ['a','b','c','d','target']},
            split_strategy='random', cv_splits=5, cv_gap=0)
        self.ks = [1,2,3,4]

    def run_case(self, optimized):
        self.frame.to_csv(self.root/'data.csv', index=False)
        cfg = self.config.to_dict(); (self.root/'config.json').write_text(json.dumps(cfg))
        sha = hashlib.sha256((self.root/'data.csv').read_bytes()).hexdigest()
        review = dict(schema_version=1,artifact_type='custom_dataset_review',dataset=dict(sha256=sha),
            preprocessing=cfg,review=dict(confirmed=True,training_started=False,scope='configuration_only'))
        (self.root/'review.json').write_text(json.dumps(review))
        v = validate_custom_dataset(self.root/'data.csv',self.root/'config.json',self.root/'review.json')
        self.assertEqual(v['status'],'valid',v)
        e = dict(schema_version=1,confirmed=True,dataset_sha256=sha,config_sha256=v['config_sha256'],
            methods=['forward_sfs'],k_values=self.ks,heavy_confirmed=True)
        (self.root/'experiment.json').write_text(json.dumps(e))
        with count_fits() as counts:
            report = compare_custom_dataset(*(self.root/n for n in ['data.csv','config.json','review.json','experiment.json']), reuse_sfs_prefixes=optimized)
        self.assertEqual(report['status'],'compared',report)
        return report, counts

    def parity(self):
        before, counts_before = self.run_case(False)
        after, counts_after = self.run_case(True)
        self.assertEqual(before,after)  # Every fold/k, score, support, summary and warning.
        return before, counts_before, counts_after

    def test_random_every_fold_and_k_exact_parity_and_reduced_real_fit_count(self):
        _, before, after = self.parity()
        self.assertEqual(before['selector_linear_regression_fits'],5*3*(4+7+9))
        self.assertEqual(after['selector_linear_regression_fits'],5*3*9)
        self.assertEqual(after['preprocessor_fits'],5*3*9+5)
        # A second request recomputes, rather than carrying a cache across datasets/jobs.
        _, second = self.run_case(True)
        self.assertEqual(after,second)

    def test_missing_values_median_clipping_and_noncontiguous_grid(self):
        self.frame.loc[::9,'b'] = np.nan; self.frame.loc[::13,'d'] = np.nan
        self.frame.loc[5,'a'] = 100
        self.config = replace(self.config,outlier_strategy='clip');self.ks=[1,3,4]
        self.parity()

    def test_chronological_missing_mean_and_gap(self):
        self.frame['date'] = pd.date_range('2024-01-01',periods=150,freq='h').strftime('%Y-%m-%dT%H:%M')
        self.frame.loc[::9,'b'] = np.nan
        self.config=replace(self.config,split_strategy='chronological',split_column='date',cv_gap=1,missing_numeric='mean')
        self.parity()

    def test_exact_ties_keep_sklearn_candidate_order(self):
        for n in ['b','c','d']: self.frame[n] = self.frame['a']
        report,_,_ = self.parity()
        for point in report['points']:
            for fold in point['folds']:
                self.assertEqual(fold['features'],self.config.feature_columns[:point['k']])

    def test_nonmonotone_search_path_is_not_a_slice_of_final_support(self):
        self.frame['target'] = 7*self.frame['d']+3*self.frame['b']+.01*self.frame['a']
        report,_,_ = self.parity()
        for fold in report['points'][0]['folds']:
            self.assertEqual(fold['features'],['d'])
        for fold in report['points'][1]['folds']:
            self.assertEqual(fold['features'],['b','d'])

    def test_all_features_still_bypasses_sfs_search(self):
        self.ks=[4]
        with patch.object(SequentialFeatureSelector,'fit',side_effect=AssertionError('all columns must bypass search')):
            self.parity()

    def test_unknown_sklearn_version_falls_back_to_original(self):
        with patch('src.custom_sfs_comparison.sklearn.__version__','999.0.0'):
            _,before,after=self.parity()
        self.assertEqual(before,after)

    def test_cache_rejects_data_settings_and_fold_changes(self):
        X=self.frame[self.config.feature_columns];y=self.frame.target
        cv=inner_partitions(len(X),self.config)
        def selector(config=self.config, partitions=cv):
            return SequentialFeatureSelector(NumericFoldRegressor(config),n_features_to_select=1,
                cv=partitions,scoring='neg_root_mean_squared_error',direction='forward',n_jobs=1)
        cache=SFSPrefixCache();cache.wrap(selector(),X.to_numpy(),y)
        for changed_X,changed_y,s in [(X.iloc[::-1].to_numpy(),y,selector()),
              (X.to_numpy(),y.iloc[::-1],selector()),
              (X.to_numpy(),y,selector(replace(self.config,random_state=7))),
              (X.to_numpy(),y,selector(partitions=cv[::-1]))]:
            with self.assertRaisesRegex(ValueError,'must not cross'):
                cache.wrap(s,changed_X,changed_y)

    def test_outer_preprocessor_only_reuses_identical_training_and_settings(self):
        X=self.frame[self.config.feature_columns];y=self.frame.target;cache=SFSPrefixCache();seen=[]
        def fit(X,y,c):
            seen.append(list(X.index));return fit_preprocessor(X,y,c)
        a=cache.preprocessor(X,y,self.config,X.to_numpy(),fit)
        b=cache.preprocessor(X,y,self.config,X.to_numpy(),fit)
        self.assertIs(a,b);self.assertEqual(seen,[list(X.index)])
        for changed_X,changed_y,c in [(X.iloc[::-1],y.iloc[::-1],self.config),
              (X,y,replace(self.config,missing_numeric='mean'))]:
            with self.assertRaisesRegex(ValueError,'must not cross'):
                cache.preprocessor(changed_X,changed_y,c,changed_X.to_numpy(),fit)

    def test_ordinary_training_selector_does_not_opt_in(self):
        with patch.object(SFSPrefixCache,'wrap',side_effect=AssertionError('training must not use comparison cache')):
            fit_custom_selection(self.frame[['a','b','c','d']],self.frame.target,self.config,SelectionSpec('forward_sfs',k=2,gap=0))

    def test_holdout_changes_cannot_change_any_point(self):
        self.frame['date']=pd.date_range('2024-01-01',periods=150,freq='h').strftime('%Y-%m-%dT%H:%M')
        self.config=replace(self.config,split_strategy='chronological',split_column='date')
        before,_=self.run_case(True)
        self.frame.loc[120:,'target']+=10000;self.frame.loc[120:,'a']-=999
        after,_=self.run_case(True)
        self.assertEqual(before['points'],after['points'])


if __name__=='__main__':unittest.main()
