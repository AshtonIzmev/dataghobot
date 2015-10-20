from sklearn.datasets import load_digits
from optim.models import RFHopt, XGBopt
from optim.hyperopt import HyperoptParam
import unittest
import pandas as pd


class OptimTestCase(unittest.TestCase):

    def test_XGBopt(self):
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

    def test_RFHopt(self):
        data = load_digits(2)
        rfhopt = RFHopt.RFHopt(pd.DataFrame(data.data), pd.Series(data.target))
        best = rfhopt.run_hp_rf(HyperoptParam.HyperoptParam.param_space_reg_skl_rf)
        self.assertIn('max_features', best)
        self.assertIn('n_estimators', best)
        self.assertIn('max_depth', best)


if __name__ == '__main__':
    unittest.main()