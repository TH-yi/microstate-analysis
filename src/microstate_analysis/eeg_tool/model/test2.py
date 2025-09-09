import codecs, json
from eeg_tool.utilis import read_subject_info, create_info
from eeg_tool.algorithm.clustering.microstate import Microstate
import matplotlib.pyplot as plt
import mne
import numpy as np
import itertools
from openpyxl import Workbook, load_workbook
from scipy.io import loadmat, savemat
import os


if __name__ == '__main__':
    # path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_matlab_yyx\data\april_2(1)\rest_2.set'
    # raw = mne.io.read_epochs_eeglab(path)
    # raw.info = create_info(250)
    # raw.plot_psd(1,40)

    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_matlab\merged_epochs_data'
    for subject in subjects:
        save_path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_mat' + "\\" + subject
        # os.mkdir(save_path)
        for task in tasks:
            read_path = path + "\\" + subject + "\\" + task +".set"
            raw = mne.io.read_epochs_eeglab(read_path)
            data = raw.get_data()
            savemat(save_path + "\\" + task +".mat", {'EEG':data})