from eeg_tool.utilis import read_subject_info, create_info, read_xlsx
import numpy as np
from scipy.io import loadmat, savemat
import os
import codecs, json
import mne
import matplotlib.pyplot as plt


if __name__ == '__main__':
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    for subject in subjects:
        # os.mkdir(r'D:\EEGdata\clean_data_six_problem\1_40\set_psd' + '\\' + subject)

        # bad_chs_file = read_xlsx(r'D:\EEGdata\clean_data_six_problem\bad_chs_1_40.xlsx', subject, False)
        # bad_chs = dict(zip(bad_chs_file[0],bad_chs_file[1])) if len(bad_chs_file) > 0 else {}
        # os.mkdir(r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_matlab_psd' + '\\' + subject)

        # os.mkdir(r'D:\EEGdata\clean_data_six_problem\1_40\mat_psd' + '\\' + subject)
        for task in tasks:
            print(subject, task)
            # read_path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_matlab\data' + '\\' +subject + '\\' +task +'.set'
            save_path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_mat' + '\\' +subject + '\\' +task +'.mat'
            # raw = mne.io.read_raw_eeglab(read_path)
            info = create_info(250)
            raw = mne.io.RawArray(loadmat(save_path)['EEG'], info)
            # raw.info = info
            # if task in bad_chs:
            #     raw.info['bads'] = bad_chs[task].split(',')
            #     raw.interpolate_bads()
            # data = raw.get_data()
            # savemat(save_path, {'EEG':data})
            fig = raw.plot_psd(1, 40, n_fft=500, n_overlap=250, dB=False, estimate='power')
            # fig.savefig(r'D:\EEGdata\clean_data_six_problem\1_40\mat_psd' + '\\' + subject + '\\' + task +'.png', dpi=200)
            plt.show()
            plt.close()