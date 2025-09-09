from eeg_tool.utilis import read_subject_info
from eeg_tool.model.raw_data import RawData
from eeg_tool.algorithm.clustering import microstate
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
import json

class ProjectData:
    def __init__(self, subject_info=None):
        self.subject_info = subject_info

    def read_subject_info(self, input_fname):
        self.subject_info = read_subject_info(input_fname=input_fname)

    @staticmethod
    def get_full_path(input_fname, subject_name):
        return input_fname + subject_name + "\\" +subject_name + ".vhdr"


if __name__ == '__main__':
    project = ProjectData()
    project.read_subject_info('C:\\Users\\umroot\\Desktop\\creativity\\subject.xlsx')
    k = 4
    freq = [2, 20]
    for freq_index in range(0, len(freq), 2):
        l_freq = freq[freq_index]
        h_freq = freq[freq_index+1]
        for subject in project.subject_info:
            raw = RawData()
            raw.read_raw_data(input_fname=ProjectData.get_full_path('D:\\EEGdata\\raw_data_creativity\\', subject), montage='D:\\workspace\\eeg_tool\\Cap63.locs',preload=True)
            raw.read_trial_info(input_fname='C:\\Users\\umroot\\Desktop\\creativity\\notebook_res.xlsx',sheet_name=subject)
            raw.bandpass_filter(l_freq=1, h_freq=30)
            raw.split_trial()
            keys = ['1_rest', '3_rest']
            n_keys = len(keys)
            ax = []
            plt.figure()
            plt.ion()
            k = 4
            n_run = 10
            fig, ax = plt.subplots(n_keys, k)
            for i in range(n_keys):
                maps, gev, L_ = microstate.kmeans_modified(raw.trial_data[keys[i]], k, n_run, n_std=1)
                for j in range(k):
                    ax[i, j].set_title(gev[j], fontsize=5)
                    norm_eegmap = microstate.norm_0_1(maps.T[:, j])
                    im, _ = mne.viz.plot_topomap(maps.T[:, j], raw.raw_data.info, show=False, axes=ax[i, j],
                                                 image_interp='spline36', contours=6)

            plt.ioff()
            plt.show()
    # project = ProjectData()
    # project.read_subject_info('C:\\Users\\umroot\\Desktop\\creativity\\subject.xlsx')
    # k = 4
    # n_run = 10
    # n_each_trial = 6
    # n_trial = 6
    # l_freq = 8
    # h_freq = 13
    # freq = [1, 30, 1, 4, 4, 8, 8, 13, 13, 30]
    # cmap = 'RdBu_r'
    # for freq_index in range(0, len(freq), 2):
    #     l_freq = freq[freq_index]
    #     h_freq = freq[freq_index+1]
    #     for subject in project.subject_info:
    #         raw = RawData()
    #         raw.read_raw_data(input_fname=ProjectData.get_full_path('D:\\EEGdata\\raw_data_six_problem\\', subject), montage='D:\\workspace\\eeg_tool\\Cap63.locs',preload=True)
    #         raw.read_trial_info(input_fname='C:\\Users\\umroot\\Desktop\\six_problem\\notebook_res.xlsx',sheet_name=subject)
    #         raw.bandpass_filter(l_freq=l_freq, h_freq=h_freq)
    #         raw.split_trial()
    #         trial_name = []
    #         for key in raw.trial_info.keys():
    #             temp = key.split('_')
    #             if temp[1] not in trial_name and temp[1] != 'rest':
    #                 trial_name.append(temp[1])
    #         for activity in trial_name:
    #             plt.ion()
    #             fig, ax = plt.subplots(n_trial, k)
    #             data_save = {}
    #             for trial in range(1, 7):
    #                 activity_name = str(trial) + '_' + activity
    #                 maps, gev, L_ = microstate.kmeans_modified(raw.trial_data[activity_name], k, n_run)
    #                 ordered_gev = np.argsort(gev)[::-1]
    #                 data_save['activity_name'] = maps.copy().tolist()
    #                 for index, value in enumerate(ordered_gev):
    #                     ax[trial-1, index].set_title(gev[value], fontsize=5)
    #                     im, _ = mne.viz.plot_topomap(maps.T[:, value], raw.raw_data.info, show=False, vmin=-0.3, vmax=0.3, axes=ax[trial-1, index], cmap=cmap)
    #                     if index == k-1:
    #                         mne.viz.topomap._add_colorbar(ax=ax[trial-1, index], im=im, cmap=cmap, pad=0.2, size="10%")
    #                     # print('test')
    #             plt.subplots_adjust(wspace=0.5, hspace=0.5)
    #             plt.ioff()
    #             path = 'D:\\EEGdata\\six_problem_res\\'+subject+'\\' + "without_peaks_"+str(l_freq) + '_' + str(h_freq)+'\\'
    #             if not os.path.isdir(path):
    #                 os.makedirs(path)
    #             plt.savefig(path + activity+'.png', dpi=(1000))
    #             plt.show()
    #             with open(path + activity + ".json", 'w', encoding='utf-8') as f:
    #                 json.dump(data_save, f)

