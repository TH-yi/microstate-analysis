import numpy as np
from scipy import signal
import mne
from scipy.io import loadmat, savemat
from eeg_tool.utilis import create_info, load_data, read_subject_info
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    info = create_info(250)
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all')
    read_path = r'D:\EEGdata\clean_data_six_problem\1_40\downsample_data'
    save_path_psd = r'D:\EEGdata\clean_data_six_problem\1_40\psd\pic'
    for subject in subjects:
        print(subject)
        read_path_mat = read_path + "\\" +subject + '.mat'
        data = loadmat(read_path_mat)['EEG']
        os.mkdir(save_path_psd+"\\" +subject)
        for task in tasks:
            raw = mne.io.RawArray(data[task][0][0], info)
            fig = raw.plot_psd(1, 40, dB=True, estimate='power', show=False)
            save_path = save_path_psd + "\\" +subject + "\\" +task +'.png'
            fig.savefig(save_path, dpi=200)
            plt.close()