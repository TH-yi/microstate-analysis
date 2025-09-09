import numpy as np
from multiprocessing import Pool
import codecs, json
from collections import OrderedDict
from scipy import signal
import os
import mne
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, create_info, to_string
from scipy.io import loadmat, savemat
from scipy.stats import sem
from scipy import trapz, argmax
from itertools import combinations
import matplotlib.pyplot as plt
import itertools
import scipy.stats as st


if __name__ == '__main__':
    data = np.asarray(read_xlsx(r'C:\Users\Zeng\Desktop\1_30_duration_occurrence_coverage.xlsx','coverage'))
    data_dic = {0:data[:,0:6],1:data[:,6:12],2:data[:,12:18],3:data[:,18:24]}
    order = [5, 1, 4, 3, 0, 2]
    title = ['rest','idea generation','idea evolution','evaluation']
    alpha = 0.95
    sample_size = 28
    res = []

    for comb in itertools.combinations([i for i in range(4)], 2):
        temp = []
        for i in range(6):
            t, p = st.ttest_rel(data_dic[comb[0]][:, i], data_dic[comb[1]][:, i])
            diff = data_dic[comb[0]][:, i] - data_dic[comb[1]][:, i]
            diff_mean = np.mean(diff)
            diff_std = st.sem(diff)
            ci = st.t.interval(alpha, sample_size-1, loc=diff_mean, scale=diff_std)
            temp.append((title[comb[0]]+"_"+title[comb[1]], 't-value:', t, 'p-value', p, 'low-ci',ci[0], 'high-ci',ci[1]))
        order_temp = [temp[i] for i in order]
        res.append(order_temp)
    for item in res:
        for item_i in item:
            print(item_i)
