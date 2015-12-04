

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

