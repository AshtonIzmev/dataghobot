from GenericOpt import GenericOpt as Gopt
import xgboost as xgb
import re


class XGBOpt(Gopt):

    @staticmethod
    def straln(s):
        return re.sub(r'[^a-zA-Z0-9]+', 'xx', s)

    @staticmethod
    def predict_hopt(clf_arg, x_test):
        dtest = xgb.DMatrix(x_test.rename(columns=XGBOpt.straln))
        return clf_arg.predict(dtest)

    @staticmethod
    def create_fit_hopt(x_train, y_train, params_arg):
        dtrain = xgb.DMatrix(x_train.rename(columns=XGBOpt.straln), label=y_train)
        plst = params_arg.items()
        evallist = [(dtrain, 'train')]
        return xgb.train(plst, dtrain, 25, evallist, verbose_eval=False)

    @staticmethod
    def assert_params_ok(params_arg):
        Gopt.assert_params_ok(params_arg)
        # xgb params
        assert 'eval_metric' in params_arg
        assert 'booster' in params_arg
        assert 'objective' in params_arg
        assert 'eta' in params_arg
        assert 'num_round' in params_arg
        assert 'silent' in params_arg
        assert 'seed' in params_arg
        assert 'num_boost_round' in params_arg
