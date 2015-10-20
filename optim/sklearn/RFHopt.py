from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, cross_validation
from hyperopt import fmin, tpe, STATUS_OK
import numpy as np


class RFHopt:

    def __init__(self, x_data, y_data, verbose=False,):
        self.verbose = verbose
        self.x_data = x_data
        self.y_data = y_data
        return

    def cross_val_pred_skl(self, clf_arg, cv):
        kf = cross_validation.KFold(len(self.x_data), n_folds=cv, shuffle=True)
        preds = np.zeros(len(self.x_data))
        preds_probas = np.zeros((len(self.x_data), 2))
        for train_index, test_index in kf:
            x_train, x_test = self.x_data.iloc[train_index], self.x_data.iloc[test_index]
            y_train = self.y_data.iloc[train_index]
            clf_arg.fit(x_train, y_train)
            preds[test_index] = clf_arg.predict(x_test)
            preds_probas[test_index] = clf_arg.predict_proba(x_test)
        return preds_probas, preds

    def get_hp_score_auc_rf(self, params_arg):
        clf = RandomForestClassifier(n_estimators=params_arg['n_estimators'],
                                     max_features=params_arg['max_features'],
                                     random_state=params_arg['random_state'],
                                     max_depth=1+params_arg['max_depth'],
                                     n_jobs=params_arg['n_jobs'])
        (preds_probas_rf, _) = self.cross_val_pred_skl(clf, params_arg['cv'])
        return metrics.roc_auc_score(self.y_data, preds_probas_rf[:, 1])

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
        # sklearn params
        assert 'cv' in params_arg
        # RF params
        assert 'n_estimators' in params_arg
        assert 'max_features' in params_arg
        assert 'random_state' in params_arg
        assert 'max_depth' in params_arg
        assert 'n_jobs' in params_arg
        # hyperopt params
        assert 'max_evals' in params_arg


