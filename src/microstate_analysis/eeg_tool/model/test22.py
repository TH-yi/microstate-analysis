from itertools import combinations
from eeg_tool.utilis import load_data, read_subject_info, write_info, read_xlsx, to_string
from scipy.io import loadmat, savemat
import numpy as np
from scipy import stats
from eeg_tool.math_utilis import ceil_decimal
from statsmodels.stats.multitest import multipletests

if __name__ == '__main__':
    n_microstates = 7
    n_paritions = 3
    n_condition = 6
    alpha = 0.05
    path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\subjects_formulation\condition_'
    save_path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\mean\mean.xlsx'
    comb = [item for item in combinations([i for i in range(n_microstates)], n_paritions)]
    data = []
    for item in comb:
        read_path = path + to_string(item) + ".mat"
        temp = loadmat(read_path)['EEG'].T
        data.append(temp.tolist())
    np_data = np.asarray(data)
    mean_np_data = np_data.mean(axis=0)
    for i in range(n_condition):
        print(round(mean_np_data[:,i].mean(),3), round(stats.sem(mean_np_data[:,i]),3))
    # p_list = []
    # for comb in combinations([i for i in range(n_condition)], 2):
    #     p = stats.ttest_rel(mean_np_data[:, comb[0]], mean_np_data[:, comb[1]])
    #     p_list.append(p[1])
    # bonferroni_corrected = multipletests(p_list, alpha, 'bonferroni')[1]
    # write_info(save_path, 'p-values', [p_list])
    # write_info(save_path, 'corrected-p-values', [bonferroni_corrected.tolist()])
    # write_info(save_path, 'mean', mean_np_data)