import numpy as np
from collections import OrderedDict
from microstate_analysis.microstate_base.tanova_calc import tanova

def bootstrap_sample(data, n_subjects):
    resampled_data = OrderedDict()
    random_subjects = np.random.choice(np.arange(n_subjects), n_subjects, replace=True)
    for condition, condition_data in data.items():
        resampled_data[condition] = condition_data[random_subjects]
    return resampled_data


def bootstrap_test(data, condition, n_subjects, n_ch):
    observed = tanova(data, condition, n_subjects, n_ch)
    res = []
    n_iters = 1000
    for i in range(n_iters):
        resampled_data = bootstrap_sample(data, n_subjects)
        p_val = tanova(resampled_data, condition, n_subjects, n_ch)
        res.append(p_val)
    p_val = np.sum(np.greater(res, observed)) / n_iters
    return p_val