import pandas as pd
import logging
from dataghobot.featengine import ChaosGeneration as Cg
from dataghobot.featengine import Numerisation as Nz
from dataghobot.stacking import CrossValStack as Cvs
from sklearn.cross_validation import KFold


def robot(x_train, y_train, x_valid, xgb_ip, skl_ip, ext_ip, **robot_kwargs):

    robot_cv_feat = robot_kwargs.get('robot_cv_feat', 6)
    robot_cv_hopt = robot_kwargs.get('robot_cv_hopt', 6)
    robot_nb_auto_max = robot_kwargs.get('robot_nb_auto_max', -1)
    robot_rand_state = robot_kwargs.get('robot_rand_state', 42)

    res = []
    nb_auto = 0

    for train1_idx, feat_idx in KFold(len(x_train), n_folds=robot_cv_feat, shuffle=True,
                                      random_state=robot_rand_state):
        x_train1 = x_train.iloc[train1_idx]
        y_train1 = y_train.iloc[train1_idx]
        x_feat = x_train.iloc[feat_idx]
        y_feat = y_train.iloc[feat_idx]

        logging.info("Chaos feature generation")
        x_train1, x_valid = chaosize(x_feat, x_train1, x_valid, y_feat, **robot_kwargs)

        logging.info("Feature cleaning")
        x_train_num, x_valid_num = numerize(x_train1, x_valid, **robot_kwargs)

        for train2_idx, hopt_idx in KFold(len(x_train_num), n_folds=robot_cv_hopt, shuffle=True,
                                          random_state=robot_rand_state):
            x_train2 = x_train_num.iloc[train2_idx]
            y_train2 = y_train1.iloc[train2_idx]
            x_hopt = x_train_num.iloc[hopt_idx]
            y_hopt = y_train1.iloc[hopt_idx]

            logging.info("Looking for hopt parameters")
            xgb_rp = enhance_param(Cvs.get_best_xgbopt(x_hopt, y_hopt, xgb_ip), **robot_kwargs)
            skl_rp = enhance_param(Cvs.get_best_sklopt(x_hopt, y_hopt, skl_ip), **robot_kwargs)
            ext_rp = enhance_param(Cvs.get_best_etopt(x_hopt, y_hopt, ext_ip), **robot_kwargs)
            res.append(
                Cvs.cross_val_stack(x_train2, y_train2, x_valid_num, xgb_rp, skl_rp, ext_rp, **robot_kwargs)
            )

            nb_auto += 1
            if nb_auto == robot_nb_auto_max:
                return res
            print 'nb auto increased: ', nb_auto
    return res


def enhance_param(params, **enhance_args):
    for k, v in params[0].iteritems():
        if k in enhance_args:
            params[0][k] = enhance_args[k]
    return params


def stacking_res_to_one_pred(res):
    s = 0
    nb = 0
    for i in range(len(res)):
        for j in range(len(res[i])):
            s = s + res[i][j][:, 1]
            nb += 1
    return s / nb


def chaosize(x_feat, x_train1, x_valid, y_feat, **chaos_args):
    x_feat.loc[:, 'source'] = 0
    x_train1.loc[:, 'source'] = 1
    x_valid.loc[:, 'source'] = 2
    x_all = pd.concat([x_feat, x_train1, x_valid])
    Cg.chaos_feature_importance(x_all, y_feat, x_all['source'] == 0, **chaos_args)
    x_train_res = x_all[x_all['source'] == 1].drop(['source'], axis=1)
    x_valid_res = x_all[x_all['source'] == 2].drop(['source'], axis=1)
    return x_train_res, x_valid_res


def numerize(x_train1, x_valid, **numerize_args):
    x_train1.loc[:, 'source'] = 0
    x_valid.loc[:, 'source'] = 1
    x_all = pd.concat([x_train1, x_valid])
    num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = Nz.handle_columns(x_all, [], [], **numerize_args)
    x_all_num = Nz.treat_dataframe(x_all, num_cols, [], todummy_cols, tootherisation_cols, **numerize_args)
    x_train_num = x_all_num[x_all_num['source'] == 0].drop(['source'], axis=1)
    x_valid_num = x_all_num[x_all_num['source'] == 1].drop(['source'], axis=1)
    return x_train_num, x_valid_num
