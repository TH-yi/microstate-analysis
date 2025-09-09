import numpy as np
from multiprocessing import Pool
import codecs, json
from collections import OrderedDict
from scipy import signal
import os
import mne
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx
from scipy.io import loadmat, savemat
from scipy.stats import sem
import itertools

if __name__ == '__main__':
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    conditions = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered')
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_set\info'
    for condition in conditions:
        temp = []
        for task in tasks:
            if task.startswith(condition):
                for subject in subjects:
                    bad_chs = loadmat(path + "\\" +subject +"\\" + task +".mat")['info']['bad_chs'][0][0][0]
                    n = len(bad_chs)
                    temp.append(n)
        print(condition, np.mean(temp), sem(temp))