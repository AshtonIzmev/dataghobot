from scipy.stats import entropy
import numpy as np
import pandas as pd


def handle_columns(df, key_cols, date_cols, **numerize_args):
    """
    Generate the columns that has to be treated for dummysation
    :param df:
    :param key_cols: columns that should not be modified
    :param date_cols: columns that will be used for date extraction
    :param numerize_args: additionnal arguments
    :return:
    """

    numerize_nb_dummy_max = numerize_args.get('numerize_nb_dummy_max', 10)
    numerize_entropy_max = numerize_args.get('numerize_entropy_max', 3)

    num_cols = [col for col in df.columns if df[col].dtype in [np.dtype('int64'), np.dtype('float64')]]
    object_cols = [col for col in df.columns if df[col].dtype == np.dtype('object') and
                   col not in key_cols and col not in date_cols]
    todummy_cols = [col for col in object_cols if len(df[col].unique()) <= numerize_nb_dummy_max]
    tootherisation_cols = [col for col in object_cols if len(df[col].unique()) > numerize_nb_dummy_max and
                           entropy(df[col].value_counts()/sum(df[col].value_counts())) < numerize_entropy_max]
    tobedefined_cols = list(set(object_cols) - (set(todummy_cols) | set(num_cols) | set(tootherisation_cols)))
    return num_cols, todummy_cols, tootherisation_cols, tobedefined_cols


def treat_dataframe(df_arg, num_cols, date_cols, todummy_cols, toother_cols, **numerize_args):
    """
    Convert a dataset into a full numeric matrix
    :param df_arg:
    :param num_cols: columns to keep as they are
    :param date_cols: columns for date feature extraction
    :param todummy_cols: columns to extract dummies
    :param toother_cols: columns to extract dummies after otherisation
    :param numerize_args: additionnal arguments
    :return:
    """

    numerize_ratio = numerize_args.get('numerize_ratio', 10)
    numerize_time_window = numerize_args.get('numerize_time_window', [5, 12, 18])

    x_result = pd.DataFrame()
    for col in todummy_cols:
        dummies = pd.get_dummies(df_arg[col], prefix='split_'+col, dummy_na=True)
        x_result = pd.concat([x_result, dummies], axis=1)
    for col in toother_cols:
        value_counts = df_arg[col].fillna('').value_counts().to_dict()

        def otherisation(s):
            return 'other' if value_counts[s] < df_arg.shape[0]/numerize_ratio else s
        dummies = pd.get_dummies(df_arg[col].fillna('').map(otherisation), prefix='splot_'+col, dummy_na=True)
        x_result = pd.concat([x_result, dummies], axis=1)
    for col in date_cols:
        x_result[col+'_ho'] = df_arg[col].map(lambda d: d.hour)
        x_result[col+'_ho_sl'] = df_arg[col].map(lambda d: int(d.hour <= numerize_time_window[0]))
        x_result[col+'_ho_mo'] = df_arg[col].map(lambda d: int((d.hour <= numerize_time_window[1]) &
                                                               (d.hour > numerize_time_window[0])))
        x_result[col+'_ho_af'] = df_arg[col].map(lambda d: int((d.hour <= numerize_time_window[2]) &
                                                               (d.hour > numerize_time_window[1])))
        x_result[col+'_ho_ev'] = df_arg[col].map(lambda d: int(d.hour > numerize_time_window[2]))
        x_result[col+'_wd'] = df_arg[col].map(lambda d: d.weekday())
        x_result[col+'_mt'] = df_arg[col].map(lambda d: d.month)
        x_result[col+'_yr'] = df_arg[col].map(lambda d: d.year)
    for c1 in date_cols:
        x_result[c1+'_timestamp'] = x_result[c1+'_timestamp'] = \
            (df_arg[c1].map(lambda d: (d - np.datetime64('2000-01-01T00:00:00Z')) / np.timedelta64(1, 's')))
        for c2 in date_cols:
            if c1 > c2:
                x_result[c1+'_'+c2+'_timediff'] = (df_arg[c1] - df_arg[c2])/np.timedelta64(1, 'm')
    return pd.concat([df_arg[num_cols], x_result], axis=1)
