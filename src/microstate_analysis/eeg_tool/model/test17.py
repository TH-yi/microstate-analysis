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
from PIL import Image, EpsImagePlugin

if __name__ == '__main__':
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    conditions = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered')
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_matlab\fig'
    EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.53.3\bin\gswin64c'
    for subject in subjects[22::]:
        print(subject)
        save_path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_matlab\png'+"\\"+subject
        os.mkdir(save_path)
        for task in tasks:
            for i in range(1,7):
                read_path = path + "\\" + subject + "\\" + task + "_" + str(i) +".eps"
                try:
                    im = Image.open(read_path)
                    im.load(scale=10)
                    im.save(save_path + "\\" + task + "_" + str(i) +".png")
                except Exception as e:
                    print(e)
                    pass
