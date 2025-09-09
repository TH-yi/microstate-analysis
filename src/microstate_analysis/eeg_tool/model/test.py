from eeg_tool.utilis import read_subject_info
import mne
from eeg_tool.utilis import create_info, load_data
import numpy as np
from collections import OrderedDict
import os
import codecs, json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import savemat, loadmat
from multiprocessing import Pool


def combine_task_data(data, combined_task):
    data_temp = OrderedDict()
    for task in combined_task:
        a_task = task.split("#")
        if len(a_task) > 1:
            temp = np.asarray(data[a_task[0]]['task_data'])
            for i in range(1, len(a_task)):
                temp = np.concatenate((temp, np.asarray(data[a_task[i]]['task_data'])),axis=1)
            data_temp[task] = temp.tolist()
        else:
            data_temp[task] = data[task]['task_data']
    return data_temp

if __name__ == '__main__':
    info = create_info(500)
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'all')
    tasks_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all')
    path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data'
    bad_chs_path = r'D:\EEGdata\clean_data_six_problem\bad_chs_1_40.xlsx'
    for subject in subjects[23::]:
        print(subject)
        read_path = path + "\\" + subject +'.json'
        data = load_data(read_path)
        data = combine_task_data(data, tasks)
        res = {}
        save_path = r'D:\EEGdata\clean_data_six_problem\1_40\downsample_data' + "\\" + subject +'.mat'
        bad_chs = read_subject_info(bad_chs_path, subject)
        for index, task in enumerate(tasks):
            print(task, tasks_name[index])
            raw = mne.io.RawArray(data[task], info)
            raw.preload = True
            raw = raw.set_eeg_reference()
            raw = raw.resample(250)
            raw.info['bads'] = bad_chs
            raw = raw.interpolate_bads()
            res[tasks_name[index]] = raw.get_data()
        # json.dump(res, codecs.open(save_path, 'w', encoding='utf-8'), separators=(',', ':'))
        # savemat(save_path,{'EEG':res})

