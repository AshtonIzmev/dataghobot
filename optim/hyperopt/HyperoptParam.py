import numpy as np
from hyperopt import hp


class HyperoptParam:

    def __init__(self):
        return

    #######################################
    # XGBoost
    #######################################
    param_space_reg_xgb_linear = {
        'task': 'regression',
        'booster': 'gblinear',
        'objective': 'reg:linear',
        'eta': hp.quniform('eta', 0.01, 1, 0.01),
        'lambda': hp.quniform('lambda', 0, 5, 0.05),
        'alpha': hp.quniform('alpha', 0, 0.5, 0.005),
        'lambda_bias': hp.quniform('lambda_bias', 0, 3, 0.1),
        'num_round': hp.quniform('num_round', 10, 100, 10),
        'num_boost_round': 25,
        'nthread': 14,
        'silent': 1,
        'seed': 42,
        'max_evals': 1,
        'eval_metric': 'auc', #logloss
        'cv': 3
    }

    param_space_reg_xgb_tree = {
        'task': 'regression',
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eta': hp.quniform('eta', 0.01, 1, 0.01),
        'gamma': hp.quniform('gamma', 0, 2, 0.1),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
        'num_round': hp.quniform('num_round', 10, 100, 10),
        'num_boost_round': 25,
        'nthread': 14,
        'silent': 1,
        'seed': 42,
        'max_evals': 1,
        'eval_metric': 'logloss', #auc
        'cv': 3
    }

    #######################################
    # Sklearn
    #######################################
    param_space_reg_skl_rf = {
        'task': 'reg_skl_rf',
        'n_estimators': hp.randint('n_estimators', 100),
        'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
        'n_jobs': -1,
        'max_depth': hp.randint('max_depth', 25),
        'random_state': 42,
        'max_evals': 1,
        'cv': 3,
        'eval_metric': 'logloss' #auc
    }

    param_space_reg_skl_etr = {
        'task': 'reg_skl_etr',
        'n_estimators': hp.randint('n_estimators', 100),
        'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
        'n_jobs': -1,
        'max_depth': hp.randint('max_depth', 25),
        'random_state': 42,
        'max_evals': 1,
        'cv': 3,
        'eval_metric': 'logloss' #auc
    }

    param_space_clf_skl_lr = {
        'task': 'clf_skl_lr',
        'C': hp.loguniform('C', np.log(0.001), np.log(10)),
        'random_state': 42,
        'max_evals': 1,
    }

    #######################################
    # Keras
    #######################################
    param_space_reg_keras_dnn = {
        'task': 'reg_keras_dnn',
        'batch_norm': hp.choice('batch_norm', [True, False]),
        'hidden_units': hp.choice('hidden_units', [64, 128, 256, 512]),
        'hidden_layers': hp.choice('hidden_layers', [1, 2, 3, 4]),
        'input_dropout': hp.quniform('input_dropout', 0, 0.9, 0.1),
        'hidden_dropout': hp.quniform('hidden_dropout', 0, 0.9, 0.1),
        'hidden_activation': hp.choice('hidden_activation', ['relu', 'prelu']),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
        'nb_epoch': hp.choice('nb_epoch', [10, 20, 30, 40]),
        'max_evals': 1,
    }

