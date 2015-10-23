from sklearn.datasets import load_digits
from optim.models import Sklearnopt, XGBopt, VWopt, Kerasopt
from optim.hyperopt import HyperoptParam
import unittest
import pandas as pd


class OptimTestCase(unittest.TestCase):

    def test_xgbopt_tree_auc(self):
        data = load_digits(2)
        cols = ['i'+str(i) for i in range(data.data.shape[1])]
        xgbopt = XGBopt.XGBopt(pd.DataFrame(data.data, columns=cols),
                               pd.Series(data.target))
        param = HyperoptParam.HyperoptParam.param_space_reg_xgb_tree
        param['eval_metric'] = 'auc'
        best = xgbopt.run_hp_xgb(param)
        self.assertIsNotNone(best)
        self.assertLess(xgbopt.score, -0.99)

    def test_xgbopt_tree_logloss(self):
        data = load_digits(2)
        cols = ['i'+str(i) for i in range(data.data.shape[1])]
        xgbopt = XGBopt.XGBopt(pd.DataFrame(data.data, columns=cols),
                               pd.Series(data.target))
        param = HyperoptParam.HyperoptParam.param_space_reg_xgb_tree
        param['max_evals'] = 10
        param['eval_metric'] = 'logloss'
        best = xgbopt.run_hp_xgb(param)
        self.assertIsNotNone(best)
        self.assertLess(xgbopt.score, 0.03)

    def test_xgbopt_linear_auc(self):
        data = load_digits(2)
        cols = ['i'+str(i) for i in range(data.data.shape[1])]
        xgbopt = XGBopt.XGBopt(pd.DataFrame(data.data, columns=cols),
                               pd.Series(data.target))
        param = HyperoptParam.HyperoptParam.param_space_reg_xgb_linear
        param['eval_metric'] = 'auc'
        best = xgbopt.run_hp_xgb(param)
        self.assertIsNotNone(best)
        self.assertLess(xgbopt.score, -0.99)

    def test_rfopt_auc(self):
        data = load_digits(2)
        skopt = Sklearnopt.Sklearnopt(pd.DataFrame(data.data), pd.Series(data.target))
        param = HyperoptParam.HyperoptParam.param_space_reg_skl_rf
        param['eval_metric'] = 'auc'
        best = skopt.run_hp_rf(param)
        self.assertIsNotNone(best)
        self.assertLess(skopt.score, -0.99)

    def test_rfopt_logloss(self):
        data = load_digits(2)
        skopt = Sklearnopt.Sklearnopt(pd.DataFrame(data.data), pd.Series(data.target))
        param = HyperoptParam.HyperoptParam.param_space_reg_skl_rf
        param['eval_metric'] = 'logloss'
        best = skopt.run_hp_rf(param)
        self.assertIsNotNone(best)
        self.assertLess(skopt.score, 0.03)

    def test_lropt_auc(self):
        data = load_digits(2)
        skopt = Sklearnopt.Sklearnopt(pd.DataFrame(data.data), pd.Series(data.target))
        param = HyperoptParam.HyperoptParam.param_space_clf_skl_lr
        param['eval_metric'] = 'auc'
        best = skopt.run_hp_lr(param)
        self.assertIsNotNone(best)
        self.assertLess(skopt.score, -0.99)

    def test_lropt_logloss(self):
        data = load_digits(2)
        skopt = Sklearnopt.Sklearnopt(pd.DataFrame(data.data), pd.Series(data.target))
        param = HyperoptParam.HyperoptParam.param_space_clf_skl_lr
        param['eval_metric'] = 'logloss'
        best = skopt.run_hp_lr(param)
        self.assertIsNotNone(best)
        self.assertLess(skopt.score, 0.01)

    def test_vwopt_auc(self):
        data = load_digits(2)
        vwopt = VWopt.VWopt(pd.DataFrame(data.data), pd.Series(data.target))
        param = HyperoptParam.HyperoptParam.param_space_classification_vw
        param['eval_metric'] = 'auc'
        best = vwopt.run_hp_vw(param)
        self.assertIsNotNone(best)
        self.assertLess(vwopt.score, -0.99)

    def test_vwopt_logloss(self):
        data = load_digits(2)
        vwopt = VWopt.VWopt(pd.DataFrame(data.data), pd.Series(data.target))
        param = HyperoptParam.HyperoptParam.param_space_classification_vw
        param['eval_metric'] = 'logloss'
        best = vwopt.run_hp_vw(param)
        self.assertIsNotNone(best)
        self.assertLess(vwopt.score, 0.3)

    def test_kerasopt_auc(self):
        data = load_digits(2)
        kerasopt = Kerasopt.Kerasopt(pd.DataFrame(data.data), pd.Series(data.target))
        param = HyperoptParam.HyperoptParam.param_space_reg_keras_dnn
        param['eval_metric'] = 'auc'
        best = kerasopt.run_hp_keras(param)
        self.assertIsNotNone(best)
        self.assertLess(kerasopt.score, -0.95)

if __name__ == '__main__':
    unittest.main()