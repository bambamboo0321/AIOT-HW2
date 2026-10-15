"""Training-only runner、巢狀 CV、timeout 與 Phase 1/2 回歸檢查。"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.feature_selection import METHODS
from src.phase2_common import CSV_PATH, EXPECTED_SHA256, ROOT
from src.preprocessing import ColumnValues
from src.preprocessing_config import PreprocessingConfig
from src.selection_benchmark import BenchmarkCase, estimate_costs, run_benchmark, run_bounded
from src.selection_runner import TrainingPartition, load_energy_training, regression_metrics, run_nested_cv
from src.validate_data import validate


def fixture(n=120, p=3):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    X['date'] = pd.date_range('2025-01-01', periods=n, freq='10min')
    y = pd.Series(4*X.x0 - 2*X.x1 + rng.normal(scale=.1, size=n), name='y')
    cfg = PreprocessingConfig('y', [f'x{i}' for i in range(p)], split_strategy='chronological', split_column='date')
    return TrainingPartition(X, y, tuple(range(n)), tuple(range(n,n+20))), cfg


class SelectionRunnerTests(unittest.TestCase):
    def test_energy_loader_only_parses_existing_training_rows(self):
        original = pd.read_csv
        with patch('src.selection_runner.pd.read_csv', wraps=original) as reading:
            partition, cfg = load_energy_training()
        self.assertEqual(reading.call_count, 1)
        self.assertEqual(reading.call_args.kwargs['nrows'], 15788)
        self.assertEqual((len(partition.X), len(cfg.feature_columns)), (15788,25))
        self.assertEqual(partition.X.index[-1], 15787)
        self.assertEqual(partition.forbidden_indices[0], 15788)
        self.assertNotIn('Appliances', partition.X)
        self.assertNotIn('rv1', cfg.feature_columns)

    def test_actual_five_outer_three_inner_architecture_all_six_methods(self):
        partition, cfg = fixture()
        result = run_nested_cv(partition, cfg, METHODS, (1,2), outer_splits=5, inner_splits=3, gap=0)
        self.assertEqual(len(result['outer_fold_manifest']),5)
        self.assertFalse(result['holdout_used'])
        for method in METHODS:
            folds = result['methods'][method]['folds']
            self.assertEqual(len(folds),5)
            self.assertTrue(all(f['outer_metrics'] is not None for f in folds))
            for f in folds:
                if method in ('lasso_cv','rfecv'):
                    self.assertIsNone(f['chosen_inner_k'])
                else:
                    best=min(f['inner_k_curve'], key=lambda row:(row['mean_inner_rmse'],row['k']))['k']
                    self.assertEqual(best,f['chosen_inner_k'])
        json.dumps(result,allow_nan=False)

    def test_outer_folds_are_disjoint_chronological_with_gap(self):
        partition, cfg=fixture()
        result=run_nested_cv(partition,cfg,['f_test'],(1,),outer_splits=3,inner_splits=2,gap=2)
        for f in result['outer_fold_manifest']:
            train,valid=f['train_indices'],f['validation_indices']
            self.assertFalse(set(train)&set(valid))
            self.assertLess(partition.X.loc[train,'date'].max(),partition.X.loc[valid,'date'].min())
            self.assertGreaterEqual(min(valid)-max(train),3)

    def test_selector_and_predictor_fit_indices_exclude_current_validation(self):
        partition,cfg=fixture()
        audit=[]
        result=run_nested_cv(partition,cfg,['f_test','rfe'],(1,2),outer_splits=3,inner_splits=2,gap=0,audit=audit)
        for event in audit:
            fold=int(event['scope'].split('/')[0].split('_')[1])
            f=result['outer_fold_manifest'][fold]
            self.assertTrue(set(event['fit_indices'])<=set(f['train_indices']))
            self.assertTrue(set(event['fit_indices']).isdisjoint(f['validation_indices']))
            self.assertTrue(set(event['fit_indices']).isdisjoint(partition.forbidden_indices))

    def test_preprocessor_fit_is_not_allowed_to_see_outer_validation(self):
        partition,cfg=fixture()
        # x0 = row index，使直接觀察實際傳給 Phase 2 Transformer 的值即可驗證列。
        partition.X['x0']=np.arange(len(partition.X),dtype=float)
        original=ColumnValues.fit
        seen=[]
        def observed(transformer,X,y=None):
            if X.columns[0]=='x0':
                seen.append(tuple(X.iloc[:,0].astype(int)))
            return original(transformer,X,y)
        with patch.object(ColumnValues,'fit',autospec=True,side_effect=observed):
            result=run_nested_cv(partition,cfg,['f_test'],(1,),outer_splits=3,inner_splits=2,gap=0)
        self.assertEqual(seen,[tuple(f['train_indices']) for f in result['outer_fold_manifest']])

    def test_changing_outer_validation_target_does_not_change_first_selection(self):
        partition,cfg=fixture()
        first=run_nested_cv(partition,cfg,['f_test'],(1,2),outer_splits=2,inner_splits=2,gap=0)
        valid=first['outer_fold_manifest'][0]['validation_indices']
        changed_y=partition.y.copy()
        changed_y.loc[valid]+=100000
        changed=TrainingPartition(partition.X,changed_y,partition.allowed_indices,partition.forbidden_indices)
        second=run_nested_cv(changed,cfg,['f_test'],(1,2),outer_splits=2,inner_splits=2,gap=0)
        a,b=first['methods']['f_test']['folds'][0],second['methods']['f_test']['folds'][0]
        self.assertEqual(a['selection'],b['selection'])
        self.assertEqual(a['inner_k_curve'],b['inner_k_curve'])
        self.assertNotEqual(a['outer_metrics'],b['outer_metrics'])

    def test_holdout_unsorted_and_full_grid_are_rejected(self):
        partition,cfg=fixture()
        with self.assertRaisesRegex(ValueError,'Holdout'):
            TrainingPartition(partition.X,partition.y,tuple(range(100))).validate(cfg)
        with self.assertRaisesRegex(ValueError,'遞增'):
            TrainingPartition(partition.X.iloc[::-1],partition.y.iloc[::-1],partition.allowed_indices).validate(cfg)
        with self.assertRaisesRegex(ValueError,'三個 k'):
            run_nested_cv(partition,cfg,['f_test'],tuple(range(1,26)),gap=0)
        with self.assertRaises(ValueError):
            partition.validate(replace(cfg,split_strategy='random'))

    def test_metrics_match_independent_arithmetic(self):
        metrics=regression_metrics([1,2,4],[1,1,3])
        self.assertAlmostEqual(metrics['rmse'],np.sqrt(2/3))
        self.assertAlmostEqual(metrics['mae'],2/3)
        self.assertAlmostEqual(metrics['r2'],4/7)
        with self.assertRaises(ValueError):
            regression_metrics([1],[1])

    def test_zero_features_never_produces_partial_outer_average(self):
        partition,cfg=fixture()
        result=run_nested_cv(partition,cfg,['lasso_cv'],(),outer_splits=2,inner_splits=2,gap=0,alphas=(1e8,))
        method=result['methods']['lasso_cv']
        self.assertEqual(method['status'],'incomplete_zero_features')
        self.assertEqual(method['zero_feature_folds'],2)
        self.assertIsNone(method['mean_outer_metrics'])
        self.assertEqual(method['fit_accounting'].get('downstream_linear_regression_fits',0),0)

    def test_real_subprocess_timeout_kills_and_reaps_child(self):
        started=monotonic()
        result=run_bounded([sys.executable,'-c','import time; time.sleep(10)'],.15)
        self.assertEqual(result['status'],'timeout')
        self.assertTrue(result['timeout'])
        self.assertTrue(result['child_reaped'])
        self.assertLess(monotonic()-started,3)

    def test_benchmark_total_budget_and_no_retry_after_timeout(self):
        result=run_benchmark([BenchmarkCase('forward_sfs',3),BenchmarkCase('forward_sfs',5)],case_timeout=.02,total_timeout=.1)
        self.assertEqual(result['results'][0]['status'],'timeout')
        self.assertIn(result['results'][1]['status'],['not_run_total_budget','not_run_after_method_timeout'])
        self.assertLess(result['elapsed_seconds'],3)

    def test_phase1_validation_and_raw_csv_remain_unchanged(self):
        before=hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
        result=validate(CSV_PATH)
        self.assertNotEqual(result['overall_status'],'FAIL')
        self.assertEqual(sum(c['status']=='FAIL' for c in result['checks']),0)
        self.assertEqual(hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),before)
        self.assertEqual(before,EXPECTED_SHA256)

    def test_phase2_baseline_models_and_reports_match_accepted_hashes(self):
        evidence=json.loads((ROOT/'reports/phase2/notebook_sync_verification.json').read_text())
        for name,item in evidence['protected_files'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),item['after_sha256'],name)


if __name__=='__main__':
    unittest.main()
