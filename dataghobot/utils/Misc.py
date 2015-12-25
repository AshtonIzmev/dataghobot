

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

