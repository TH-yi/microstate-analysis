from itertools import combinations
from eeg_tool.utilis import load_data, read_subject_info, write_info, read_xlsx, to_string
from scipy.io import loadmat, savemat
import numpy as np
from scipy import stats
from eeg_tool.math_utilis import ceil_decimal
from statsmodels.stats.multitest import multipletests


if __name__ == '__main__':
    data = loadmat(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\entropy_rate\history_6\condition.mat')['EEG'].T
    save_path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\entropy_rate\history_6\entropy_rate.xlsx'
    alpha = 0.05
    # for i in range(6):
    #     print(round(data[:,i].mean(),3), round(stats.sem(data[:,i]),3))
    p_list = []
    for comb in combinations([i for i in range(6)], 2):
        p = stats.ttest_rel(data[:, comb[0]], data[:, comb[1]])
        p_list.append(p[1])
    bonferroni_corrected = multipletests(p_list, alpha, 'bonferroni')[1]
    write_info(save_path, 'p-values', [p_list])
    write_info(save_path, 'corrected-p-values', [bonferroni_corrected.tolist()])
