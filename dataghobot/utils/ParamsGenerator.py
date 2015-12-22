from dataghobot.hyperopt import HyperoptParam


def generate_robot_params():
    return {
        'robot_cv_feat': 6,
        'robot_cv_hopt': 6,
        'robot_nb_auto_max': 2,
        'robot_rand_state': 42
    }


def generate_chaos_params():
    return {
        'chaos_feat_iter': 10,
        'chaos_n_estimators': 10,
        'chaos_nb_features': 30,
        'chaos_gen_iter': 20,
        'chaos_dummy_max': 20
    }


def generate_cross_val_stack_params():
    return {
        'csvstack_cv': 5
    }


def generate_numerize_params():
    return {
        'numerize_nb_dummy_max': 10,
        'numerize_entropy_max': 3,
        'numerize_ratio': 10,
        'numerize_time_window': [5, 12, 18]
    }


def union(*dicts):
    return dict(sum(map(lambda dct: list(dct.items()), dicts), []))


def generate_all_params():
    return union(generate_numerize_params(),
                 generate_cross_val_stack_params(),
                 generate_chaos_params(),
                 generate_robot_params())


def get_xgb_init_param(cv=5, max_evals=1, num_round=3, num_boost_round=3, nthread=1):
    xgb_initparam = HyperoptParam.HyperoptParam.param_space_reg_xgb_tree
    xgb_initparam['eval_metric'] = 'auc'
    xgb_initparam['cv'] = cv
    xgb_initparam['max_evals'] = max_evals
    xgb_initparam['num_round'] = num_round
    xgb_initparam['num_boost_round'] = num_boost_round
    xgb_initparam['nthread'] = nthread
    return xgb_initparam


def get_rf_init_param(cv=3, max_evals=3, n_estimators=16):
    rf_initparam = HyperoptParam.HyperoptParam.param_space_reg_skl_rf
    rf_initparam['eval_metric'] = 'auc'
    rf_initparam['type'] = 'random_forest'
    rf_initparam['cv'] = cv
    rf_initparam['max_evals'] = max_evals
    rf_initparam['n_estimators'] = n_estimators
    return rf_initparam


def get_ext_init_param(cv=3, max_evals=3, n_estimators=16):
    ext_initparam = HyperoptParam.HyperoptParam.param_space_reg_skl_rf
    ext_initparam['eval_metric'] = 'auc'
    ext_initparam['type'] = 'extra_trees'
    ext_initparam['cv'] = cv
    ext_initparam['max_evals'] = max_evals
    ext_initparam['n_estimators'] = n_estimators
    return ext_initparam
