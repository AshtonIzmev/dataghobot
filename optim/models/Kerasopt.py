from sklearn import cross_validation, metrics
import numpy as np
from hyperopt import fmin, tpe, STATUS_OK
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import StandardScaler


class Kerasopt:

    auc_label = 'auc'
    logloss_label = 'logloss'
    scaler = StandardScaler()
    score = np.iinfo(np.int32).max

    def __init__(self, x_data, y_data, verbose=False,):
        self.verbose = verbose
        self.x_data = x_data
        self.y_data = y_data
        return

    @staticmethod
    def build_keras_model(params_arg, x_shape1):
        model = Sequential()
        model.add(Dropout(params_arg["input_dropout"]))
        first = True
        hidden_layers = params_arg['hidden_layers']
        while hidden_layers > 0:
            if first:
                dim = x_shape1
                first = False
            else:
                dim = params_arg["hidden_units"]
            model.add(Dense(dim, params_arg["hidden_units"], init='glorot_uniform'))
            if params_arg["batch_norm"]:
                model.add(BatchNormalization((params_arg["hidden_units"],)))
            if params_arg["hidden_activation"] == "prelu":
                model.add(PReLU((params_arg["hidden_units"],)))
            else:
                model.add(Activation(params_arg['hidden_activation']))
            model.add(Dropout(params_arg["hidden_dropout"]))
            hidden_layers -= 1
        model.add(Dense(params_arg["hidden_units"], 1, init='glorot_uniform'))
        model.add(Activation('linear'))
        model.compile(loss=params_arg['loss_function'], optimizer="adam")
        return model

    def cross_val_pred_keras(self, params_arg):
        kf = cross_validation.KFold(len(self.x_data), n_folds=params_arg['cv'], shuffle=True)
        preds_probas = np.zeros(len(self.x_data))
        for train_index, test_index in kf:
            x_train, x_test = self.x_data.iloc[train_index], self.x_data.iloc[test_index]
            y_train = self.y_data.iloc[train_index]
            model = Kerasopt.build_keras_model(params_arg, x_train.shape[1])
            x_train_scale = self.scaler.fit_transform(x_train)
            x_test_scale = self.scaler.transform(x_test)
            model.fit(x_train_scale, y_train+1, nb_epoch=params_arg['nb_epoch'],
                      batch_size=params_arg['batch_size'], validation_split=0, verbose=params_arg['verbose'])
            preds_probas[test_index] = model.predict_proba(x_test_scale, verbose=0)
        return preds_probas

    def get_score_keras(self, params_arg):
        preds_probas = self.cross_val_pred_keras(params_arg)
        if params_arg['eval_metric'] == Kerasopt.auc_label:
            return -metrics.roc_auc_score(self.y_data, preds_probas)
        if params_arg['eval_metric'] == Kerasopt.logloss_label:
            return metrics.log_loss(self.y_data, preds_probas)
        raise Exception('Eval metric error : auc or logloss')

    def objective_keras(self, params_arg):
        score = self.get_score_keras(params_arg)
        if score < self.score:
            self.score = score
        if self.verbose:
            print "\tScore {0}\tParams{1}".format(score, params_arg)
        return {'loss': score, 'status': STATUS_OK}

    def run_hp_keras(self, params_arg):
        self.assert_params_ok(params_arg)
        return fmin(self.objective_keras, params_arg, algo=tpe.suggest, max_evals=params_arg['max_evals'])

    @staticmethod
    def assert_params_ok(params_arg):
        # models params
        assert 'cv' in params_arg
        # keras params
        assert 'loss_function' in params_arg
        assert 'verbose' in params_arg
        assert 'batch_norm' in params_arg
        assert 'hidden_units' in params_arg
        assert 'hidden_layers' in params_arg
        assert 'input_dropout' in params_arg
        assert 'hidden_dropout' in params_arg
        assert 'hidden_activation' in params_arg
        assert 'batch_size' in params_arg
        assert 'nb_epoch' in params_arg
        # hyperopt params
        assert 'max_evals' in params_arg
        # metric params
        assert 'eval_metric' in params_arg
        assert params_arg['eval_metric'] in [Kerasopt.auc_label, Kerasopt.logloss_label]