from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, tpe, STATUS_OK
import numpy as np


class Sklearnopt:

    auc_label = 'auc'
    logloss_label = 'logloss'
    score = np.iinfo(np.int32).max

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

    def get_hp_score(self, clf, params_arg):
        (preds_probas_rf, _) = self.cross_val_pred_skl(clf, params_arg['cv'])
        if params_arg['eval_metric'] == Sklearnopt.auc_label:
            return -metrics.roc_auc_score(self.y_data, preds_probas_rf[:, 1])
        if params_arg['eval_metric'] == Sklearnopt.logloss_label:
            return metrics.log_loss(self.y_data, preds_probas_rf[:, 1])
        raise Exception('Eval metric error : auc or logloss')

    def objective_hp_rf(self, params_arg):
        clf = RandomForestClassifier(n_estimators=params_arg['n_estimators'],
                                     max_features=params_arg['max_features'],
                                     random_state=params_arg['random_state'],
                                     max_depth=1+params_arg['max_depth'],
                                     n_jobs=params_arg['n_jobs'])
        score = self.get_hp_score(clf, params_arg)
        if score < self.score:
            self.score = score
        if self.verbose:
            print "\tScore {0}\tParams{1}".format(score, params_arg)
        return {'loss': score, 'status': STATUS_OK}

    def objective_hp_lr(self, params_arg):
        scaler_lr = Pipeline([
            ('scr', StandardScaler()),
            ('lr', LogisticRegression(C=params_arg['C']))])
        score = self.get_hp_score(scaler_lr, params_arg)
        if score < self.score:
            self.score = score
        if self.verbose:
            print "\tScore {0}\tParams{1}".format(score, params_arg)
        return {'loss': score, 'status': STATUS_OK}

    def run_hp_rf(self, params_arg):
        self.assert_params_rf_ok(params_arg)
        return fmin(self.objective_hp_rf, params_arg, algo=tpe.suggest, max_evals=params_arg['max_evals'])

    def run_hp_lr(self, params_arg):
        self.assert_params_lr_ok(params_arg)
        return fmin(self.objective_hp_lr, params_arg, algo=tpe.suggest, max_evals=params_arg['max_evals'])

    @staticmethod
    def assert_params_rf_ok(params_arg):
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
        # metric params
        assert 'eval_metric' in params_arg
        assert params_arg['eval_metric'] in [Sklearnopt.auc_label, Sklearnopt.logloss_label]

    @staticmethod
    def assert_params_lr_ok(params_arg):
        # sklearn params
        assert 'cv' in params_arg
        # RF params
        assert 'C' in params_arg
        # hyperopt params
        assert 'max_evals' in params_arg
        # metric params
        assert 'eval_metric' in params_arg
        assert params_arg['eval_metric'] in [Sklearnopt.auc_label, Sklearnopt.logloss_label]


