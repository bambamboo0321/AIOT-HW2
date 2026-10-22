import unittest
from dataclasses import replace
from src.custom_dataset_comparison import settings,MAX_WORK,HIGH_WORK,ComparisonRequestError
from src.custom_dataset_selectors import MAX_WORK as SELECTOR_WORK
from src.preprocessing_config import PreprocessingConfig

class ComparisonPolicyTests(unittest.TestCase):
 def setUp(self):
  self.config=PreprocessingConfig(target_column='target',feature_columns=[f'x{i}' for i in range(15)],column_types={**{f'x{i}':'numeric' for i in range(15)},'target':'numeric'},cv_splits=5)
  self.validation={'dataset':{'rows':2000,'sha256':'a'*64},'config_sha256':'b'*64,'split':{'training_rows':1600,'cv_folds':[{'fold':i+1,'training_rows':1280,'validation_rows':320} for i in range(5)]}}
  self.experiment={'schema_version':1,'confirmed':True,'dataset_sha256':'a'*64,'config_sha256':'b'*64,'methods':['f_test','mutual_information','rfe'],'k_values':list(range(1,11))}
 def test_high_work_requires_bound_confirmation_and_preserves_requested_grid(self):
  self.assertEqual(MAX_WORK,100_000_000);self.assertEqual(HIGH_WORK,10_000_000);self.assertEqual(SELECTOR_WORK,100_000_000)
  for flag in [None,False]:
   e={**self.experiment,**({'heavy_confirmed':flag} if flag is not None else {})}
   with self.assertRaisesRegex(ComparisonRequestError,'高負擔'):settings(e,self.config,self.validation)
  e={**self.experiment,'heavy_confirmed':True};m,k=settings(e,self.config,self.validation);self.assertEqual(m,self.experiment['methods']);self.assertEqual(k,list(range(1,11)))
  e['k_values']=list(range(1,12));self.assertEqual(settings(e,self.config,self.validation)[1],list(range(1,12)))
 def test_hard_budget_and_confirmation_type_not_bypassed(self):
  v={**self.validation,'split':{**self.validation['split'],'cv_folds':[{'fold':i+1,'training_rows':1333+(i>=4),'validation_rows':267-(i>=4)} for i in range(6)]}}
  with self.assertRaisesRegex(ComparisonRequestError,'過大'):settings({**self.experiment,'methods':['f_test','mutual_information','rfe','forward_sfs','rfecv','lasso_cv'],'heavy_confirmed':True},replace(self.config,cv_splits=6),v)
  for bad in ['true',1,None]:
   with self.assertRaises(ComparisonRequestError):settings({**self.experiment,'heavy_confirmed':bad},self.config,self.validation)
 def test_sfs_shape_guard_and_automatic_selection_not_changed(self):
  c=replace(self.config,feature_columns=[f'x{i}' for i in range(25)])
  with self.assertRaisesRegex(ValueError,'15'):settings({**self.experiment,'methods':['forward_sfs'],'heavy_confirmed':True},c,self.validation)
  e={**self.experiment,'methods':['rfecv','lasso_cv'],'k_values':[]}
  self.assertEqual(settings(e,self.config,self.validation)[1],[])
  with self.assertRaises(ComparisonRequestError):settings({**e,'k_values':[5]},self.config,self.validation)

if __name__=='__main__':unittest.main()
