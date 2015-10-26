from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from GenericOpt import GenericOpt as Gopt


class SklearnOpt(Gopt):

    @staticmethod
    def build_model(params_arg):
        if params_arg['type'] == 'random_forest':
            return RandomForestClassifier(n_estimators=params_arg['n_estimators'],
                                          max_features=params_arg['max_features'],
                                          random_state=params_arg['random_state'],
                                          max_depth=1 + params_arg['max_depth'],
                                          n_jobs=params_arg['n_jobs'])
        if params_arg['type'] == 'logistic_regression':
            return Pipeline([
                ('scr', StandardScaler()),
                ('lr', LogisticRegression(C=params_arg['C']))])
        raise Exception('Unknown model "type" parameter')

    @staticmethod
    def predict_hopt(clf_arg, preds, test_index, x_test):
        return clf_arg.predict_proba(x_test)[:, 1]

    @staticmethod
    def create_fit_hopt(x_train, y_train, params_arg):
        return SklearnOpt.build_model(params_arg).fit(x_train, y_train)

    @staticmethod
    def assert_params_ok(params_arg):
        Gopt.assert_params_ok(params_arg)
        assert 'type' in params_arg
        if params_arg['type'] == 'random_forest':
            # RF params
            assert 'n_estimators' in params_arg
            assert 'max_features' in params_arg
            assert 'random_state' in params_arg
            assert 'max_depth' in params_arg
            assert 'n_jobs' in params_arg
        if params_arg['type'] == 'logistic_regression':
            # LogReg params
            assert 'C' in params_arg
