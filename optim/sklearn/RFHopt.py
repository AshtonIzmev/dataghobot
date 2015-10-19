from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, cross_validation
from hyperopt import fmin, tpe, STATUS_OK
import numpy as np


class RFHopt:

    def __init__(self, verbose=False):
        self.verbose = verbose
        return

    @staticmethod
    def cross_val_pred_skl(clf_arg, x_arg, y_arg, cv=3):
        kf = cross_validation.KFold(len(x_arg), n_folds=cv, shuffle=True)
        preds = np.zeros(len(x_arg))
        preds_probas = np.zeros((len(x_arg), 2))
        for train_index, test_index in kf:
            x_train, x_test = x_arg.iloc[train_index], x_arg.iloc[test_index]
            y_train = y_arg.iloc[train_index]
            clf_arg.fit(x_train, y_train)
            preds[test_index] = clf_arg.predict(x_test)
            preds_probas[test_index] = clf_arg.predict_proba(x_test)
        return preds_probas, preds

    def get_hp_score_auc_rf(self, params_arg):
        x_arg = params_arg['x_arg']
        y_arg = params_arg['y_arg']
        clf = RandomForestClassifier(n_estimators=params_arg['n_estimators'],
                                     max_features=params_arg['max_features'],
                                     random_state=params_arg['random_state'],
                                     max_depth=1+params_arg['max_depth'],
                                     n_jobs=params_arg['n_jobs'])
        (preds_probas_rf, _) = self.cross_val_pred_skl(clf, x_arg, y_arg, cv=params_arg['cv'])
        return metrics.roc_auc_score(y_arg, preds_probas_rf[:, 1])

    def objective_auc_hp_rf(self, params_arg):
        auc_score = self.get_hp_score_auc_rf(params_arg)
        if self.verbose:
            print "\tScore {0}\tParams{1}".format(auc_score, params_arg)
        return {'loss': -auc_score, 'status': STATUS_OK}

    def run_hp_auc_rf(self, params_arg):
        self.assert_params_ok(params_arg)
        return fmin(self.objective_auc_hp_rf, params_arg, algo=tpe.suggest, max_evals=params_arg['max_evals'])

    @staticmethod
    def assert_params_ok(params_arg):
        # data
        assert 'x_arg' in params_arg
        assert 'y_arg' in params_arg
        # RF params
        assert 'n_estimators' in params_arg
        assert 'max_features' in params_arg
        assert 'random_state' in params_arg
        assert 'max_depth' in params_arg
        assert 'n_jobs' in params_arg
        # hyperopt params
        assert 'max_evals' in params_arg


