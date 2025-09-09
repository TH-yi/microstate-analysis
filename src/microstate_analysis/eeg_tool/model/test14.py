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
from scipy.stats import  ttest_rel

def merge_task_name(conditions, tasks):
    res = {}
    for condition in conditions:
        res[condition] = []
        for task in tasks:
            if task.startswith(condition):
                res[condition].append(task)
    return res



if __name__ == '__main__':
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    tasks_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered_name')
    conditions = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered')
    conditions_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered_name')
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    # path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\entropy_rate\history_3'
    path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\subjects_formulation'
    name = merge_task_name(conditions, tasks)
    # for condition in conditions:
    #     res = [[], []]
    #     data = np.asarray(loadmat(path + "\\" +condition+".mat")['EEG'])
    #     n = data.shape[0]
    #     comb = [item for item in combinations([i for i in range(n)], 2)]
    #     for item in comb:
    #         t, p = ttest_rel(data[item[0],:], data[item[1],:])
    #         temp = name[condition][item[0]] + "Vs." + name[condition][item[1]]
    #         res[0].append(temp)
    #         res[1].append(p)
    #     write_info(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\entropy_rate\history_3\entropy_rate.xlsx', condition, res)

    # data = np.asarray(loadmat(path+"\\"+'condition.mat')['EEG'])
    # comb = [item for item in combinations([i for i in range(len(conditions))], 2)]
    # res = [[],[]]
    # for item in comb:
    #     t, p = ttest_rel(data[item[0], :], data[item[1], :])
    #     temp = conditions[item[0]] +"Vs." +conditions[item[1]]
    #     res[0].append(temp)
    #     res[1].append(p)
    # write_info(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\entropy_rate\history_3\entropy_rate.xlsx', 'condition', res)

    for condition in conditions:
        print(condition)
        comb = combinations([i for i in range(7)], 3)
        for index, item in enumerate(comb):
            data = np.asarray(loadmat(path + "\\" + condition + "_" + to_string(item) +".mat")['EEG'])
            comb1 = combinations([i for i in range(data.shape[0])], 2)
            res = [[],[]]
            for item1 in comb1:
                t, p = ttest_rel(data[item1[0],:], data[item1[1],:])
                temp = name[condition][item1[0]] + "Vs." + name[condition][item1[1]]
                res[0].append(temp)
                res[1].append(p)
            write_info(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\subjects_formulation\dfa.xlsx', condition + "_" + to_string(item), res)

    comb = combinations([i for i in range(7)], 3)
    for index, item in enumerate(comb):
        data = np.asarray(loadmat(path + "\\" + 'condition' + "_" + to_string(item) +".mat")['EEG'])
        comb1 = combinations([i for i in range(data.shape[0])], 2)
        res = [[],[]]
        for item1 in comb1:
            t, p = ttest_rel(data[item1[0],:], data[item1[1],:])
            temp = conditions[item1[0]] +"Vs." +conditions[item1[1]]
            res[0].append(temp)
            res[1].append(p)
        write_info(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\subjects_formulation\dfa.xlsx', 'condition' + "_" + to_string(item), res)
