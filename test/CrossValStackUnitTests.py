from dataghobot.utils import DataGenerator
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import unittest
import pandas as pd
from dataghobot.stacking import CrossValStack
from dataghobot.models import SklearnOpt, XGBOpt


class OptimTestCase(unittest.TestCase):

    def test_cross_val_stack(self):
        x, y = DataGenerator.get_digits_data()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        xgbparam = CrossValStack.get_best_xgbopt(x_train, y_train)
        sklparam = CrossValStack.get_best_sklopt(x_train, y_train)
        extparam = CrossValStack.get_best_etopt(x_train, y_train)

        res = CrossValStack.cross_val_stack(x_train, y_train, x_test, xgbparam, sklparam, extparam)
        dfres = pd.DataFrame([res[0][:, 1], res[1][:, 1], res[2][:, 1]]).transpose()
        dfres.columns = ['p1', 'p2', 'p3']

        y_test_xgb = CrossValStack.predict_opt_clf(XGBOpt.XGBOpt(x_train, y_train), xgbparam, x_test, x_test)[0]
        y_test_skl = CrossValStack.predict_opt_clf(SklearnOpt.SklearnOpt(x_train, y_train), sklparam, x_test, x_test)[0]
        y_test_ext = CrossValStack.predict_opt_clf(SklearnOpt.SklearnOpt(x_train, y_train), extparam, x_test, x_test)[0]

        print metrics.roc_auc_score(y_test, y_test_xgb)
        print metrics.roc_auc_score(y_test, y_test_skl)
        print metrics.roc_auc_score(y_test, y_test_ext)

        print metrics.roc_auc_score(y_test, dfres.p1.values)
        print metrics.roc_auc_score(y_test, dfres.p2.values)
        print metrics.roc_auc_score(y_test, dfres.p3.values)

        print metrics.roc_auc_score(y_test, (dfres.p1+dfres.p2+dfres.p3).values/3)

        print metrics.roc_auc_score(y_test, dfres.p1.values)
        print metrics.roc_auc_score(y_test, (dfres.p1+dfres.p2+dfres.p3).values/3)

        self.assertEqual(len(res), 5)


if __name__ == '__main__':
    unittest.main()
