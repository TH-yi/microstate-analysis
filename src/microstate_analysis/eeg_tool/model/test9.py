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
    conditions = ['RE','PU', 'IG', 'RIG', 'IE', 'RIE']
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    maps = np.asarray(load_data(
        r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=7\across_conditions\reorder_microstate_across_runs_across_participants_across_conditions.json'))
    paras = ['coverage', 'duration', 'frequency']
    n_microstates = 7
    path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=7\parameters\condition_m_t_test'
    str_temp = []
    for comb in itertools.combinations([i for i in range(len(conditions))], 2):
        temp = conditions[comb[0]] +"&Vs.&" + conditions[comb[1]] +" & "
        str_temp.append(temp)

    for para in paras:
        res = []
        for m in range(n_microstates):
            temp = []
            # res.append(temp)
            read_path = path + "\\" + 'condition_' + para + "_m" + str(m) + ".mat"
            file_name = 'condition_' + para + "_m" + str(m)
            data = loadmat(read_path)['EEG'].flatten().tolist()
            for item in data:
                item = round(item, 3)
                if item > 0.05:
                    item_temp = str(item)
                elif 0.05 >= item >0.01:
                    item_temp = str(item)+'\\' + 'tnote{*}'
                elif 0.01 >= item >0.005:
                    item_temp = str(item)+'\\' + 'tnote{**}'
                else:
                    item_temp = str(item)+'\\' + 'tnote{***}'
                temp.append(item_temp + " & ")
            res.append(temp)
            # write_info(path + "\\" + 'condition_t_test.xlsx', file_name, res)
        print(para)
        for i in range(15):
            print(str_temp[i], res[0][i], res[1][i], res[2][i], res[3][i],res[4][i],res[5][i],res[6][i])
