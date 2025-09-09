import mne
import os, sys
import numpy as np
from collections import OrderedDict
from sklearn.decomposition import PCA
from mne.preprocessing import  compute_proj_ecg, compute_proj_eog, ICA
from eeg_tool import utilis
from eeg_tool.algorithm.dimensionality_reduction.multitask_principal_component_analysis import multitask_pca, zero_mean
from eeg_tool.algorithm.clustering.microstate import Microstate, batch_microstate
import matplotlib.pyplot as plt
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from eeg_tool.utilis import load_data

class RawData:
    def __init__(self):
        self.raw_data = None
        self.tasks_data = OrderedDict()
        self.epochs_data = OrderedDict()


    def read_raw_data(self, fname, montage=None, eog=None, misc='auto', scale=1., preload=False, verbose=False):
        fname = os.path.abspath(fname)
        ext = os.path.splitext(fname)[1][1:].lower()
        if ext == 'bdf':
            pass
        elif ext == 'edf':
            pass
        elif ext == 'vhdr':
            eog = ('HEOGL', 'HEOGR', 'VEOGb')
            self.raw_data = mne.io.read_raw_brainvision(vhdr_fname=fname, eog=eog, misc=misc, scale=scale, preload=preload, verbose=verbose)
            dig_montage = mne.channels.read_custom_montage(fname=montage)
            self.raw_data.set_montage(dig_montage, verbose=verbose)

    def read_montage(self, montage):
        self.raw_data.set_montage(montage)

    def read_trial_info(self, fname, sheet_name=None):
        if sheet_name is None:
            sheet_name = self.raw_data.filenames[0].split('\\')[-1].split('.')[0]
        self.trial_info = utilis.read_trial_info(fname, sheet_name)

    def remove_bad_channel(self, reset_bads=True):
        self.raw_data = self.raw_data.interpolate_bads(reset_bads=reset_bads)

    def split_tasks(self):
        data = self.raw_data.get_data()
        for key, value in self.trial_info.items():
            if ',' in str(value[0]):
                start = value[0].split(',')
                end = value[1].split(',')
                temp = data[:, int(float(start[0])):int(float(end[0]))]
                for i in range(1, len(start)):
                    temp = np.concatenate((temp, data[:, int(float(start[i])):int(float(end[i]))]), axis=1)
            else:
                temp = data[:, int(float(value[0])):int(float(value[1]))]
            self.tasks_data[key] = temp


    def split_epochs(self, epoch_sample):
        data = self.raw_data.get_data()
        for key, value in self.trial_info.items():
            if ',' in str(value[0]):
                start = value[0].split(',')
                end = value[1].split(',')

        n_t = data.shape[1]
        n_channel = data.shape[0]
        n_epochs = int(n_t / epoch_sample)
        return data[:, 0:n_epochs*epoch_sample].reshape(n_epochs, n_channel, epoch_sample)

    def concatenate_trials(self, trials_name, exclude_trials_name):
        trials_data = []
        for trial in trials_name:
            if trial not in exclude_trials_name:
                trials_data.append(self.tasks_data[trial])
        data = trials_data[0]
        for i in range(1, len(trials_name)-len(exclude_trials_name)):
            data = np.concatenate((data, trials_data[i]), axis=1)
        return trials_data, data


    def bandpass_filter(self, l_freq=1., h_freq=40., ):
        self.raw_data = self.raw_data.filter(l_freq, h_freq)

    def resample(self, sfreq):
        self.raw_data = self.raw_data.resample(sfreq=sfreq)


if __name__ == '__main__':
    pass
    # data_fname = r'D:\EEGdata\raw_data_six_problem\april_02(3)\april_02(3).vhdr'
    # subject_fname = r'C:\Users\umroot\Desktop\six_problem\notebook_res.xlsx'
    # bad_channel = ['T8', 'Fp1', 'FT8']
    # subject = 'april_02(3)'
    # resample_freq = 125

    # data_fname = r'D:\EEGdata\raw_data_creativity\eeg_jan_29_2014\eeg_jan_29_2014.vhdr'
    # subject_fname = r'C:\Users\umroot\Desktop\creativity\notebook_res.xlsx'
    # bad_channel = ['FCz', 'TP9', 'Fz']
    # subject = 'eeg_jan_29_2014'


    raw_data = RawData()
    raw_data.read_raw_data(fname=r'C:\Users\yaoya\OneDrive\Desktop\eeg\april_02(3)\april_02(3).vhdr', montage=r'C:\Users\yaoya\OneDrive\Documents\eeg_tool\Cap63.locs', preload=True)
    raw_data.read_trial_info(fname=r'C:\Users\yaoya\OneDrive\Desktop\eeg\notebook.xlsx', sheet_name='april_02(3)')
    raw_data.raw_data.plot()
    # raw_data = RawData()
    # raw_data.read_raw_data(fname=data_fname, montage='D:\\workspace\\eeg_tool\\Cap63.locs', preload=True)
    # raw_data.read_trial_info(fname=subject_fname, sheet_name=subject)
    # # raw_data.split_tasks()
    # bads = raw_data.raw_data.ch_names
    # print(bads)
    # raw_data.raw_data.info['bads'] = bads[0:62]
    # raw_data.remove_bad_channel()



    # raw_data.resample(sfreq=resample_freq)
    # raw_data.bandpass_filter(l_freq=1, h_freq=30)

    # raw_data.raw_data.plot()


    # trial_name = [*raw_data.trial_info]
    # trial_name_merged = ['idea generation', 'idea evolution', 'idea rating']
    # raw_data.raw_data.info['bads'] = ['TP8']
    # raw_data.remove_bad_channel()
    # raw_data.raw_data.interpolate_bads(reset_bads=False)

    # raw_data.bandpass_filter(l_freq=1, h_freq=30)
    # raw_data.raw_data.plot()
    # raw_data.split_trial()
    # all_data, segment = raw_data.concatenate_trial(raw_data.merge_trial_name(trial_name, trial_name_merged))

    # all_data, segment = raw_data.concatenate_trial(trial_name)
    # all_data = raw_data.trial_data['1_rest']
    # maps_min = 4
    # maps_max = 15
    # microstate = Microstate(data=all_data)
    # kmeans_modified
    # microstate.opt_microstate(min_maps=maps_min, max_maps=maps_max, n_std=3, polarity=False, smooth_threshold=5, method='kmeans_modified')
    # n_maps = len(microstate.maps)
    # duration, coverage, occurrence = microstate.microstate_statistic_individual(microstate.label, n_maps, segment)
    #
    # fig, ax = plt.subplots(1, len(microstate.maps))
    # for j in range(n_maps):
    #     ax[j].set_title(round(microstate.gev[j], 4), fontsize=5)
    #     mne.viz.plot_topomap(microstate.maps.T[:, j], raw_data.raw_data.info, show=False, axes=ax[j],
    #                          image_interp='spline36', contours=6)
    # plt.show()
    #

    # print(microstate.duration, np.sum(microstate.gev))


    # for j in range(n_maps):
    #     fig, ax = plt.subplots()
    #     ax.errorbar(np.arange(len(trial_name)), [task[j]['mean'] for task in duration],
    #                 yerr=[task[j]['std'] for task in duration], fmt='o', color='r', ecolor='grey', elinewidth=2,
    #                 capsize=4)
    #     ax.set_xticks(np.arange(len(trial_name)))
    #     ax.set_xticklabels(trial_name, fontdict={'fontsize': 6}, rotation=45)
    #     ax.set_title('Micorstate_' + str(j))
    #     plt.show()



    # maps, gevs, cvs = microstate.k_means_microstate(all_data, n_maps_min=n_maps_min, n_maps_max=n_maps_max)
    # opt_k = sorted(cvs.items(), key=lambda x: x[1])[0][0]
    # opt_map = maps[opt_k]
    # opt_gev = gevs[opt_k]
    # plt.figure()
    # plt.plot([i for i in range(n_maps_min, n_maps_max+1)], [cvs[i] for i in range(n_maps_min, n_maps_max+1)], 'bo')
    # plt.plot(opt_k, cvs[opt_k], 'ro')
    #
    # plt.ion()
    # fig, ax = plt.subplots(1, len(opt_map))
    # for j in range(len(opt_map)):
    #     ax[j].set_title(opt_gev[j], fontsize=5)
    #     im, _ = mne.viz.plot_topomap(opt_map.T[:, j], raw_data.raw_data.info, show=False, axes=ax[j], image_interp='spline36', contours=6)
    # plt.ioff()
    # plt.show()



    # n_keys = len(trial_name)
    # ax = []
    # plt.ion()
    # k = 4
    # n_run = 1
    # fig, ax = plt.subplots(n_keys, k)
    # for i in range(n_keys):
    #     # maps, gev, L_ = microstate.kmeans_modified(raw_data.trial_data[keys[i]], k, n_run, n_std=1)
    #     maps, gev, L_ = microstate.kmeans_modified(all_data, k, n_run, n_std=1)
    #     for j in range(k):
    #         ax[i, j].set_title(gev[j], fontsize=5)
    #         norm_eegmap = microstate.norm_0_1(maps.T[:, j])
    #         im, _ = mne.viz.plot_topomap(maps.T[:, j], raw_data.raw_data.info, show=False, axes=ax[i, j], image_interp='spline36', contours=6)
    #
    # plt.ioff()
    # plt.show()

    # data_temp = []
    # for key in keys:
    #     data_temp.append(zero_mean(raw_data.trial_data[key]))
    # lamda = [5]
    # u_t, score = multitask_pca(data_temp, lamda, 5, False)
    # for i in u_t:
    #     print(i)
