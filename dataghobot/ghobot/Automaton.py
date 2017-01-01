import pandas as pd
import logging
from tqdm import tqdm
from dataghobot.featengine import ChaosGeneration as Cg
from dataghobot.featengine import Numerisation as Nz
from dataghobot.stacking import CrossValStack as Cvs
from dataghobot.utils import Misc
from sklearn.cross_validation import KFold


def robot(x_train, y_train, x_valid, rf_ip, ext_ip, xgb_ip, **robot_kwargs):

    robot_cv_feat = robot_kwargs.get('robot_cv_feat', 6)
    robot_cv_hopt = robot_kwargs.get('robot_cv_hopt', 6)
    robot_cv_stack = robot_kwargs.get('robot_cv_stack', 5)
    robot_nb_auto_max = robot_kwargs.get('robot_nb_auto_max', -1)
    robot_rand_state = robot_kwargs.get('robot_rand_state', 42)

    res = []
    nb_auto = 0
    nb_samples = len(x_train)

    for train1_idx, feat_idx in KFold(nb_samples, n_folds=robot_cv_feat, shuffle=True,
                                      random_state=robot_rand_state):
        x_train1 = x_train.iloc[train1_idx]
        y_train1 = y_train.iloc[train1_idx]
        x_feat = x_train.iloc[feat_idx]
        y_feat = y_train.iloc[feat_idx]

        logging.info(" >>> DGH >>> Chaos feature generation")
        x_train1, x_valid = chaosize(x_feat, x_train1, x_valid, y_feat, **robot_kwargs)

        logging.info(" >>> DGH >>> Feature cleaning")
        x_train_num, x_valid_num = numerize(x_train1, x_valid, **robot_kwargs)

        for train2_idx, hopt_idx in KFold(len(x_train_num), n_folds=robot_cv_hopt, shuffle=True,
                                          random_state=robot_rand_state):
            x_train2 = x_train_num.iloc[train2_idx]
            y_train2 = y_train1.iloc[train2_idx]
            x_hopt = x_train_num.iloc[hopt_idx]
            y_hopt = y_train1.iloc[hopt_idx]

            logging.info(" >>> DGH >>> Looking for hopt parameters")
            rf_rp = Misc.enhance_param(Cvs.get_best_sklopt(x_hopt, y_hopt, rf_ip), **robot_kwargs)
            ext_rp = Misc.enhance_param(Cvs.get_best_etopt(x_hopt, y_hopt, ext_ip), **robot_kwargs)
            xgb_rp = Misc.enhance_param(Cvs.get_best_xgbopt(x_hopt, y_hopt, xgb_ip), **robot_kwargs)

            stack_res = []
            logging.info(" >>> DGH >>> Cross-val-stacking")
            for train3_idx, stack_idx in KFold(len(x_train2), n_folds=robot_cv_stack, shuffle=True):
                y_probas = Cvs.stack_that(x_train2, y_train2, x_valid_num, train3_idx, stack_idx,
                                          rf_rp, ext_rp, xgb_rp)
                stack_res.append(y_probas)
            res.append(stack_res)

            nb_auto += 1
            if nb_auto == robot_nb_auto_max:
                return res
    return res


def small_robot(x_train, y_train, x_valid, rf_rp, ext_rp, xgb_rp, **robot_kwargs):

    robot_cv_feat = robot_kwargs.get('robot_cv_feat', 6)
    robot_cv_stack = robot_kwargs.get('robot_cv_stack', 5)
    robot_nb_auto_max = robot_kwargs.get('robot_nb_auto_max', -1)
    robot_rand_state = robot_kwargs.get('robot_rand_state', 42)

    res = []
    nb_auto = 0

    nb_samples = len(x_train)

    for train1_idx, feat_idx in tqdm(KFold(nb_samples, n_folds=robot_cv_feat, shuffle=True,
                                           random_state=robot_rand_state), desc='cv1'):
        x_train1 = x_train.iloc[train1_idx]
        y_train1 = y_train.iloc[train1_idx]
        x_feat = x_train.iloc[feat_idx]
        y_feat = y_train.iloc[feat_idx]

        logging.info(" >>> DGH >>> Chaos feature generation")
        x_train1, x_valid = chaosize(x_feat, x_train1, x_valid, y_feat, **robot_kwargs)

        logging.info(" >>> DGH >>> Feature cleaning")
        x_train_num, x_valid_num = numerize(x_train1, x_valid, **robot_kwargs)

        stack_res = []
        logging.info(" >>> DGH >>> Cross-val-stacking")
        for train2_idx, stack_idx in tqdm(KFold(len(x_train_num), n_folds=robot_cv_stack, shuffle=True),
                                          nested=True, desc='cv2'):
            y_probas = Cvs.stack_that(x_train_num, y_train1, x_valid_num, train2_idx, stack_idx,
                                      rf_rp, ext_rp, xgb_rp)
            stack_res.append(y_probas)
        res.append(stack_res)

        nb_auto += 1
        if nb_auto == robot_nb_auto_max:
            return res
    return res


def tiny_robot(x_train, y_train, x_valid, rf_rp, ext_rp, xgb_rp, **robot_kwargs):

    robot_cv_stack = robot_kwargs.get('robot_cv_stack', 5)
    robot_nb_auto_max = robot_kwargs.get('robot_nb_auto_max', -1)

    res = []
    nb_auto = 0

    logging.info(" >>> DGH >>> Feature cleaning")
    x_train_num, x_valid_num = numerize(x_train, x_valid, **robot_kwargs)

    stack_res = []
    logging.info(" >>> DGH >>> Cross-val-stacking")
    for train1_idx, stack_idx in tqdm(KFold(len(x_train_num), n_folds=robot_cv_stack, shuffle=True),
                                      nested=True, desc='cv2'):
        y_probas = Cvs.stack_that(x_train_num, y_train, x_valid_num, train1_idx, stack_idx,
                                  rf_rp, ext_rp, xgb_rp)
        stack_res.append(y_probas)
        nb_auto += 1
        if nb_auto == robot_nb_auto_max:
            return res
        res.append(stack_res)
    return res


def chaosize(x_feat, x_mirror, x_valid, y_feat, **chaos_args):
    x_feat.loc[:, 'source'] = 0
    x_mirror.loc[:, 'source'] = 1
    x_valid.loc[:, 'source'] = 2
    x_all = pd.concat([x_feat, x_mirror, x_valid])
    Cg.chaos_feature_importance(x_all, y_feat, x_all['source'] == 0, **chaos_args)
    x_train_res = x_all[x_all['source'] == 1].drop(['source'], axis=1)
    x_valid_res = x_all[x_all['source'] == 2].drop(['source'], axis=1)
    return x_train_res, x_valid_res


def numerize(x_train1, x_valid, **numerize_args):
    x_train1.loc[:, 'source'] = 0
    x_valid.loc[:, 'source'] = 1
    x_all = pd.concat([x_train1, x_valid])
    num_cols, todummy_cols, tootherisation_cols, tobedefined_cols = Nz.handle_columns(x_all, [], [], **numerize_args)
    date_cols = []
    for c in x_train1.columns:
        if 'date' in x_all[c].dtype.name:
            date_cols.append(c)
    x_all_num = Nz.treat_dataframe(x_all, num_cols, date_cols, todummy_cols, tootherisation_cols, **numerize_args)
    x_train_num = x_all_num[x_all_num['source'] == 0].drop(['source'], axis=1)
    x_valid_num = x_all_num[x_all_num['source'] == 1].drop(['source'], axis=1)
    return x_train_num, x_valid_num
