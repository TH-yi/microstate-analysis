import numpy as np
from microstate_analysis.microstate_base.diss_calc import diss_interaction
from microstate_analysis.microstate_base.data_handler import shuffle_data
# from eeg_code.microstate_base.bootstrap import bootstrap_test
def tanova(data, condition, n_subjects, n_ch):
    n_condition = len(condition)
    grand_mean_across_subjects_across_conditions = np.mean(data, axis=(0, 1))

    s_total = 0
    n_iters = 1000
    for i in range(n_iters):
        data_subject = shuffle_data(np.copy(data), n_condition)
        for j in range(n_condition):
            grand_mean_across_subjects = np.mean(data_subject[:, j, :], axis=0)
            s_total += diss_interaction(grand_mean_across_subjects, grand_mean_across_subjects_across_conditions)

    observed = diss_interaction(np.mean(data, axis=0), grand_mean_across_subjects_across_conditions)
    return observed / s_total


def tanova_perm_test(data, condition, n_subjects, n_ch, task_name, condition_name):
    grand_mean_across_subjects_across_conditions = np.mean(data, axis=(0, 1))

    s_total = 0
    n_iters = 1000
    for i in range(n_iters):
        data_subject = shuffle_data(np.copy(data), len(condition))
        for j in range(len(condition)):
            grand_mean_across_subjects = np.mean(data_subject[:, j, :], axis=0)
            s_total += diss_interaction(grand_mean_across_subjects, grand_mean_across_subjects_across_conditions)

    observed = diss_interaction(np.mean(data, axis=0), grand_mean_across_subjects_across_conditions)
    res = {}
    p_val = np.sum(np.greater(s_total, observed)) / n_iters
    res[condition_name] = p_val
    return res


def tanova_test(data, condition, n_subjects, n_ch, n_bootstrap=1000):
    p_val = bootstrap_test(data, condition, n_subjects, n_ch)
    p_val = np.sum(np.greater(p_val, 0)) / n_bootstrap
    return p_val