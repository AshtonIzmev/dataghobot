from dataghobot.utils import DataGenerator
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import unittest
import pandas as pd
from dataghobot.stacking import CrossValStack
from dataghobot.models import SklearnOpt, XGBOpt
from dataghobot.utils import ParamsGenerator


class OptimTestCase(unittest.TestCase):

    def test_cross_val_stack(self):
        x, y = DataGenerator.get_digits_data()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        xgb_initparam = ParamsGenerator.get_xgb_init_param()
        rf_initparam = ParamsGenerator.get_rf_init_param()
        ext_initparam = ParamsGenerator.get_ext_init_param()

        xgb_bestparam = CrossValStack.get_best_xgbopt(x_train, y_train, xgb_initparam)
        rf_bestparam = CrossValStack.get_best_sklopt(x_train, y_train, rf_initparam)
        ext_bestparam = CrossValStack.get_best_etopt(x_train, y_train, ext_initparam)

        res = CrossValStack.cross_val_stack(x_train, y_train, x_test, xgb_bestparam, rf_bestparam, ext_bestparam)
        dfres = pd.DataFrame([res[0][:, 1], res[1][:, 1], res[2][:, 1]]).transpose()
        dfres.columns = ['p1', 'p2', 'p3']

        y_test_xgb = CrossValStack.predict_opt_clf(XGBOpt.XGBOpt(x_train, y_train), xgb_bestparam, x_test, x_test)[0]
        y_test_skl = CrossValStack.predict_opt_clf(SklearnOpt.SklearnOpt(x_train, y_train), rf_bestparam, x_test, x_test)[0]
        y_test_ext = CrossValStack.predict_opt_clf(SklearnOpt.SklearnOpt(x_train, y_train), ext_bestparam, x_test, x_test)[0]

        print metrics.roc_auc_score(y_test, y_test_xgb)
        print metrics.roc_auc_score(y_test, y_test_skl)
        print metrics.roc_auc_score(y_test, y_test_ext)

        print metrics.roc_auc_score(y_test, (dfres.p1+dfres.p2+dfres.p3).values/3)

        self.assertEqual(len(res), 5)

    def test_cross_val_meta_stack(self):
        x, y = DataGenerator.get_digits_data()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        xgb_initparam = ParamsGenerator.get_xgb_init_param()
        rf_initparam = ParamsGenerator.get_rf_init_param()
        ext_initparam = ParamsGenerator.get_ext_init_param()

        xgb_bestparam = CrossValStack.get_best_xgbopt(x_train, y_train, xgb_initparam)
        rf_bestparam = CrossValStack.get_best_sklopt(x_train, y_train, rf_initparam)
        ext_bestparam = CrossValStack.get_best_etopt(x_train, y_train, ext_initparam)

        res = CrossValStack.cross_val_meta_stack(x_train, y_train, x_test, xgb_bestparam, rf_bestparam, ext_bestparam,
                                                 csvstack_cv=3)
        dfres = pd.DataFrame([res[0][:, 1], res[1][:, 1], res[2][:, 1]]).transpose()
        dfres.columns = ['p1', 'p2', 'p3']

        y_test_xgb = CrossValStack.predict_opt_clf(XGBOpt.XGBOpt(x_train, y_train), xgb_bestparam, x_test, x_test)[0]
        y_test_skl = CrossValStack.predict_opt_clf(SklearnOpt.SklearnOpt(x_train, y_train), rf_bestparam, x_test, x_test)[0]
        y_test_ext = CrossValStack.predict_opt_clf(SklearnOpt.SklearnOpt(x_train, y_train), ext_bestparam, x_test, x_test)[0]

        print metrics.roc_auc_score(y_test, y_test_xgb)
        print metrics.roc_auc_score(y_test, y_test_skl)
        print metrics.roc_auc_score(y_test, y_test_ext)

        print metrics.roc_auc_score(y_test, (dfres.p1+dfres.p2+dfres.p3).values/3)

        self.assertEqual(len(res), 3)


if __name__ == '__main__':
    unittest.main()
