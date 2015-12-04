from dataghobot.featengine import MissingValues
from dataghobot.ghobot import Automaton
from dataghobot.utils import DataGenerator
from dataghobot.stacking import CrossValStack
from dataghobot.utils import ParamsGenerator
from sklearn.cross_validation import train_test_split
from dataghobot.models import SklearnOpt, XGBOpt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
import unittest


class OptimTestCase(unittest.TestCase):

    def test_full_robot(self):
        # loading
        x, y = DataGenerator.get_adult_data()

        # cleaning
        MissingValues.miss_val_indicator(x)

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

        robot_args = ParamsGenerator.generate_all_params()

        res = Automaton.robot(x_train, y_train, x_valid, **robot_args)

        y_pred_valid = Automaton.stacking_res_to_one_pred(res)

        print 'Full Robot'
        print metrics.roc_auc_score(y_valid, y_pred_valid)
        print metrics.log_loss(y_valid, y_pred_valid)

    def test_random_forest(self):
        # loading
        x, y = DataGenerator.get_adult_data()

        # cleaning
        MissingValues.miss_val_indicator(x)

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

        x_train_1, x_valid_1 = Automaton.numerize(x_train, x_valid)

        sklparam = CrossValStack.get_best_sklopt(x_train_1, y_train)
        skopt = SklearnOpt.SklearnOpt(x_train_1, y_train)
        y_pred_valid, _ = CrossValStack.predict_opt_clf(skopt, sklparam, x_valid_1, x_valid_1)

        #clf = RandomForestClassifier(n_estimators=64, max_depth=15)
        #clf.fit(x_train_1, y_train)
        #y_pred_valid = clf.predict_proba(x_valid_1)[:, 1]

        print 'Random Forest'
        print metrics.roc_auc_score(y_valid, y_pred_valid)
        print metrics.log_loss(y_valid, y_pred_valid)


if __name__ == '__main__':
    unittest.main()
