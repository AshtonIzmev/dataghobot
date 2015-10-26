from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from GenericOpt import GenericOpt as Gopt


class KerasOpt(Gopt):

    scaler = StandardScaler()

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

    @staticmethod
    def create_fit_hopt(x_train, y_train, params_arg):
        model = KerasOpt.build_keras_model(params_arg, x_train.shape[1])
        x_train_scale = KerasOpt.scaler.fit_transform(x_train)
        model.fit(x_train_scale, y_train+1, nb_epoch=params_arg['nb_epoch'], batch_size=params_arg['batch_size'],
                  validation_split=0, verbose=params_arg['verbose'])
        return model

    @staticmethod
    def predict_hopt(clf_arg, preds, test_index, x_test):
        x_test_scale = KerasOpt.scaler.transform(x_test)
        return clf_arg.predict_proba(x_test_scale, verbose=0)

    @staticmethod
    def assert_params_ok(params_arg):
        Gopt.assert_params_ok(params_arg)
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

