import numpy as np
from multiprocessing import Pool
import codecs, json
from collections import OrderedDict
from scipy import signal
import os
import mne
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, create_info, to_string
from scipy.io import loadmat, savemat
from mne.preprocessing import ICA
from scipy.stats import sem
from scipy import trapz, argmax
from itertools import combinations
import matplotlib.pyplot as plt
from time import sleep

def read_mat(path):
    info = create_info(250)
    return mne.io.RawArray(loadmat(path)['EEG'], info)

if __name__ == '__main__':
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    read_path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_mat'
    for subject in subjects:
        subject = 'april_22'
        save_path = r'D:\EEGdata\clean_data_six_problem\1_40\ica' + "\\" + subject
        # os.mkdir(save_path)
        for task in tasks[12::]:
            print(task)
            save_path_task = r'D:\EEGdata\clean_data_six_problem\1_40\ica' + "\\" + subject +"\\" +task
            os.mkdir(save_path_task)
            raw = read_mat(read_path+"\\"+subject+"\\"+task+".mat")
            ica = ICA(n_components=63, random_state=97, method='infomax')
            ica = ica.fit(raw)
            fig_list = ica.plot_properties(raw, picks=[i for i in range(63)], show=False)
            for i in range(len(fig_list)):
                fig_list[i].savefig(save_path_task+ "\\" + "IC"+str(i)+'.png')
            plt.close('all')
        break
