import unittest

from dataghobot.hyperopt import HyperoptParam
from dataghobot.models import SklearnOpt, XGBOpt, KerasOpt
from dataghobot.utils import DataGenerator


class OptimTestCase(unittest.TestCase):

    def test_xgbopt_tree_auc(self):
        x, y = DataGenerator.get_digits_data()
        xgbopt = XGBOpt.XGBOpt(x, y)
        param = HyperoptParam.HyperoptParam.param_space_reg_xgb_tree
        param['eval_metric'] = 'auc'
        best = xgbopt.run_hp(param)
        self.assertIsNotNone(best)
        self.assertLess(xgbopt.score, -0.99)

    def test_xgbopt_tree_logloss(self):
        x, y = DataGenerator.get_digits_data()
        xgbopt = XGBOpt.XGBOpt(x, y)
        param = HyperoptParam.HyperoptParam.param_space_reg_xgb_tree
        param['max_evals'] = 10
        param['eval_metric'] = 'logloss'
        best = xgbopt.run_hp(param)
        self.assertIsNotNone(best)
        self.assertLess(xgbopt.score, 0.04)

    def test_xgbopt_linear_auc(self):
        x, y = DataGenerator.get_digits_data()
        xgbopt = XGBOpt.XGBOpt(x, y)
        param = HyperoptParam.HyperoptParam.param_space_reg_xgb_linear
        param['eval_metric'] = 'auc'
        best = xgbopt.run_hp(param)
        self.assertIsNotNone(best)
        self.assertLess(xgbopt.score, -0.99)

    def test_rfopt_auc(self):
        x, y = DataGenerator.get_digits_data()
        skopt = SklearnOpt.SklearnOpt(x, y)
        param = HyperoptParam.HyperoptParam.param_space_reg_skl_rf
        param['eval_metric'] = 'auc'
        param['type'] = 'random_forest'
        best = skopt.run_hp(param)
        self.assertIsNotNone(best)
        self.assertLess(skopt.score, -0.99)

    def test_rfopt_logloss(self):
        x, y = DataGenerator.get_digits_data()
        skopt = SklearnOpt.SklearnOpt(x, y)
        param = HyperoptParam.HyperoptParam.param_space_reg_skl_rf
        param['eval_metric'] = 'logloss'
        param['type'] = 'random_forest'
        best = skopt.run_hp(param)
        self.assertIsNotNone(best)
        self.assertLess(skopt.score, 0.03)

    def test_etopt_auc(self):
        x, y = DataGenerator.get_digits_data()
        skopt = SklearnOpt.SklearnOpt(x, y)
        param = HyperoptParam.HyperoptParam.param_space_reg_skl_rf
        param['eval_metric'] = 'auc'
        param['type'] = 'extra_trees'
        best = skopt.run_hp(param)
        self.assertIsNotNone(best)
        self.assertLess(skopt.score, -0.99)

    def test_etopt_logloss(self):
        x, y = DataGenerator.get_digits_data()
        skopt = SklearnOpt.SklearnOpt(x, y)
        param = HyperoptParam.HyperoptParam.param_space_reg_skl_rf
        param['eval_metric'] = 'logloss'
        param['type'] = 'extra_trees'
        best = skopt.run_hp(param)
        self.assertIsNotNone(best)
        self.assertLess(skopt.score, 0.03)

    def test_lropt_auc(self):
        x, y = DataGenerator.get_digits_data()
        skopt = SklearnOpt.SklearnOpt(x, y)
        param = HyperoptParam.HyperoptParam.param_space_clf_skl_lr
        param['eval_metric'] = 'auc'
        param['type'] = 'logistic_regression'
        best = skopt.run_hp(param)
        self.assertIsNotNone(best)
        self.assertLess(skopt.score, -0.99)

    def test_lropt_logloss(self):
        x, y = DataGenerator.get_digits_data()
        skopt = SklearnOpt.SklearnOpt(x, y)
        param = HyperoptParam.HyperoptParam.param_space_clf_skl_lr
        param['eval_metric'] = 'logloss'
        param['type'] = 'logistic_regression'
        best = skopt.run_hp(param)
        self.assertIsNotNone(best)
        self.assertLess(skopt.score, 0.011)

    def test_kerasopt_auc(self):
        x, y = DataGenerator.get_digits_data()
        kerasopt = KerasOpt.KerasOpt(x, y)
        param = HyperoptParam.HyperoptParam.param_space_reg_keras_dnn
        param['eval_metric'] = 'auc'
        best = kerasopt.run_hp(param)
        self.assertIsNotNone(best)
        self.assertLess(kerasopt.score, -0.85)

if __name__ == '__main__':
    unittest.main()
