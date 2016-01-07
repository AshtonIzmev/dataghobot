from dataghobot.featengine import MissingValues
from dataghobot.ghobot import Automaton
from dataghobot.utils import DataGenerator
from dataghobot.stacking import CrossValStack as Cvs
from dataghobot.utils import ParamsGenerator, Misc
from sklearn.cross_validation import train_test_split
from dataghobot.models import SklearnOpt
from sklearn import metrics
import unittest
import logging


class OptimTestCase(unittest.TestCase):

    # robot with full pipeline
    def test_full_robot(self):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        # loading
        x, y = DataGenerator.get_adult_data()

        # cleaning
        MissingValues.add_miss_val_indicator(x)

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

        ext_ip, rf_ip, robot_args, xgb_ip = self.get_params()

        res = Automaton.robot(x_train, y_train, x_valid, rf_ip, ext_ip, xgb_ip, **robot_args)

        y_pred_valid = Misc.stacking_res_to_one_pred(res)

        print 'Full Robot'
        print metrics.roc_auc_score(y_valid, y_pred_valid)
        print metrics.log_loss(y_valid, y_pred_valid)

    # Robot without hopt pipelining
    def test_small_robot(self):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        # loading
        x, y = DataGenerator.get_adult_data()

        # cleaning
        MissingValues.add_miss_val_indicator(x)

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

        ext_ip, rf_ip, robot_args, xgb_ip = self.get_params()

        x_train_num, _ = Automaton.numerize(x_train, x_valid, **robot_args)
        rf_rp = Misc.enhance_param(Cvs.get_best_sklopt(x_train_num, y_train, rf_ip), **robot_args)
        ext_rp = Misc.enhance_param(Cvs.get_best_etopt(x_train_num, y_train, ext_ip), **robot_args)
        xgb_rp = Misc.enhance_param(Cvs.get_best_xgbopt(x_train_num, y_train, xgb_ip), **robot_args)

        res = Automaton.small_robot(x_train, y_train, x_valid, rf_rp, ext_rp, xgb_rp, **robot_args)

        y_pred_valid = Misc.stacking_res_to_one_pred(res)

        print 'Small Robot'
        print metrics.roc_auc_score(y_valid, y_pred_valid)
        print metrics.log_loss(y_valid, y_pred_valid)

    # Robot without hopt pipelining nor chaos feature generation
    def test_tiny_robot(self):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        # loading
        x, y = DataGenerator.get_adult_data()

        # cleaning
        MissingValues.add_miss_val_indicator(x)

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

        ext_ip, rf_ip, robot_args, xgb_ip = self.get_params()

        x_train_num, _ = Automaton.numerize(x_train, x_valid, **robot_args)
        rf_rp = Misc.enhance_param(Cvs.get_best_sklopt(x_train_num, y_train, rf_ip), **robot_args)
        ext_rp = Misc.enhance_param(Cvs.get_best_etopt(x_train_num, y_train, ext_ip), **robot_args)
        xgb_rp = Misc.enhance_param(Cvs.get_best_xgbopt(x_train_num, y_train, xgb_ip), **robot_args)

        res = Automaton.tiny_robot(x_train, y_train, x_valid, rf_rp, ext_rp, xgb_rp, **robot_args)

        y_pred_valid = Misc.stacking_res_to_one_pred(res)

        print 'Tiny Robot'
        print metrics.roc_auc_score(y_valid, y_pred_valid)
        print metrics.log_loss(y_valid, y_pred_valid)

    def test_random_forest(self):
        # loading
        x, y = DataGenerator.get_adult_data()

        # cleaning
        MissingValues.add_miss_val_indicator(x)

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

        x_train_1, x_valid_1 = Automaton.numerize(x_train, x_valid)

        sklparam = Cvs.get_best_sklopt(x_train_1, y_train, ParamsGenerator.get_rf_init_param())
        skopt = SklearnOpt.SklearnOpt(x_train_1, y_train)
        y_pred_valid, _ = Cvs.predict_opt_clf(skopt, sklparam, x_valid_1, x_valid_1)

        print 'Random Forest'
        print metrics.roc_auc_score(y_valid, y_pred_valid)
        print metrics.log_loss(y_valid, y_pred_valid)

    def get_params(self):
        robot_args = ParamsGenerator.generate_all_params()
        rf_ip = ParamsGenerator.get_rf_init_param()
        ext_ip = ParamsGenerator.get_ext_init_param()
        xgb_ip = ParamsGenerator.get_xgb_init_param()
        return ext_ip, rf_ip, robot_args, xgb_ip


if __name__ == '__main__':
    unittest.main()
