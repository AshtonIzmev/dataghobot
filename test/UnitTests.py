from sklearn.datasets import load_digits
from optim.models import Sklearnopt, XGBopt, VWopt
from optim.hyperopt import HyperoptParam
import unittest
import pandas as pd


class OptimTestCase(unittest.TestCase):

    def test_xgbopt_tree(self):
        data = load_digits(2)
        cols = ['i'+str(i) for i in range(data.data.shape[1])]
        xgbopt = XGBopt.XGBopt(pd.DataFrame(data.data, columns=cols),
                               pd.Series(data.target))

        best = xgbopt.run_hp_xgb(HyperoptParam.HyperoptParam.param_space_reg_xgb_tree)
        self.assertIn('colsample_bytree', best)
        self.assertIn('min_child_weight', best)
        self.assertIn('subsample', best)
        self.assertIn('eta', best)
        self.assertIn('num_round', best)
        self.assertIn('max_depth', best)
        self.assertIn('gamma', best)

    def test_xgbopt_linear(self):
        data = load_digits(2)
        cols = ['i'+str(i) for i in range(data.data.shape[1])]
        xgbopt = XGBopt.XGBopt(pd.DataFrame(data.data, columns=cols),
                               pd.Series(data.target))
        best = xgbopt.run_hp_xgb(HyperoptParam.HyperoptParam.param_space_reg_xgb_linear)
        self.assertIn('lambda_bias', best)
        self.assertIn('alpha', best)
        self.assertIn('eta', best)
        self.assertIn('num_round', best)
        self.assertIn('lambda', best)

    def test_rfopt(self):
        data = load_digits(2)
        skopt = Sklearnopt.Sklearnopt(pd.DataFrame(data.data), pd.Series(data.target), verbose=True)
        best = skopt.run_hp_rf(HyperoptParam.HyperoptParam.param_space_reg_skl_rf)
        self.assertIn('max_features', best)
        self.assertIn('n_estimators', best)
        self.assertIn('max_depth', best)

    def test_vwopt(self):
        data = load_digits(2)
        vwopt = VWopt.VWopt(pd.DataFrame(data.data), pd.Series(data.target), verbose=True)
        best = vwopt.run_hp_vw(HyperoptParam.HyperoptParam.param_space_classification_vw)
        self.assertIn('l1', best)
        self.assertIn('l2', best)
        self.assertIn('decay_learning_rate', best)
        self.assertIn('learning_rate', best)

    def test_lropt(self):
        data = load_digits(2)
        skopt = Sklearnopt.Sklearnopt(pd.DataFrame(data.data), pd.Series(data.target))
        best = skopt.run_hp_lr(HyperoptParam.HyperoptParam.param_space_clf_skl_lr)
        self.assertIn('C', best)

if __name__ == '__main__':
    unittest.main()