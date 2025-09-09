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
from eeg_tool.algorithm.preprocessing.prep_pipeline import PreProcessPipeline
from multiprocessing import Pool
from collections import OrderedDict, Counter


class Eeg:
    def __init__(self):
        self.n_ch = None
        self.trial_duration = []
        self.tasks_cleaned = None
        self.raw_data = None
        self.tasks_data = OrderedDict()
        self.tasks_merged = None
        self.segmented_raw_data = None
        self.segmented_trial_info = None
        self.new_raw_data = None
        self.bad_array = None
        self.global_good_index_list = []
        self.ica_data_array = None
        self.pca_list = None
        self.ica_list = None
        self.source_list = None
        self.mixing_list = None
        self.rename_channel_dict = None
        self.bad_ica_components = None
        self.artifacts_removal_eeg = None
        self.artifacts_removal_eeg_list = []
        self.new_tasks_data = OrderedDict()
        self.mark_epochs = None
        self.bad_epochs_list = None

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

    def read_trial_info(self, fname, sheet_name=None):
        if sheet_name is None:
            sheet_name = self.raw_data.filenames[0].split('\\')[-1].split('.')[0]
        self.trial_info = utilis.read_trial_info(fname, sheet_name)

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

    def data_segment(self):
        data = self.raw_data.get_data()
        tasks_values = list(self.trial_info.values())
        np_tasks_values = np.array(tasks_values)
        self.tasks_start = list(np_tasks_values[:, 0])
        self.tasks_end = list(np_tasks_values[:, 1])

        self.mk_start = list(np_tasks_values[:, 2])
        self.mk_end = list(np_tasks_values[:, 3])

        self.tasks_start = list(map(int, self.tasks_start))
        self.tasks_end = list(map(int, self.tasks_end))
        self.mk_start = list(map(int, self.mk_start))
        self.mk_end = list(map(int, self.mk_end))

        segment_list = []
        self.raw_array = []
        # self.raw_mne_array = []
        self.task_array = []
        self.trial_array = []
        sum_end = 0
        sum_start = 0
        new_trial_start = 0
        new_trial_end = 0
        self.new_trial = []
        for i in range(len(self.tasks_start)):
            # raw分段
            start = int(self.tasks_start[i])
            end = int(self.tasks_end[i])
            self.segmented_raw_data = np.hsplit(data, [start, end])[1]

            self.raw_array.append(self.segmented_raw_data)
            # self.raw_mne_task_array = mne.io.RawArray(self.raw_array[i], self.raw_data.info)
            # self.raw_mne_array.append(self.raw_mne_task_array)

            # task分段
            task_data = list(self.tasks_data.items())
            task_data_list = []
            task_data_list.append(task_data[i])
            self.segmented_tasks = dict(task_data_list)

            self.task_array.append(self.segmented_tasks)

            # trial分段
            trial_data = list(self.trial_info.items())
            trial_data_list = []
            trial_data_list.append(trial_data[i])
            self.segmented_trial_info = dict(trial_data_list)

            self.trial_array.append(self.segmented_trial_info)

            # sidebar trial
            if i > 0 and self.tasks_start[i] == self.tasks_start[i-1]:
                new_trial_start = new_trial_start
                new_trial_end = new_trial_end
                new_trial_list = []
                new_trial_list.append(new_trial_start)
                new_trial_list.append(new_trial_end)
                new_trial_list.append(self.mk_start[i])
                new_trial_list.append(self.mk_end[i])
                trial_data_i = list(trial_data[i])
                trial_data_i[1] = new_trial_list
                self.new_trial.append(trial_data_i)
            else:
                sum_end += end
                sum_start += start
                new_trial_start = new_trial_end
                new_trial_end = sum_end - sum_start
                new_trial_list = []
                new_trial_list.append(new_trial_start)
                new_trial_list.append(new_trial_end)
                new_trial_list.append(self.mk_start[i])
                new_trial_list.append(self.mk_end[i])
                trial_data_i = list(trial_data[i])
                trial_data_i[1] = new_trial_list
                self.new_trial.append(trial_data_i)


            segmented_raw_data_T = self.segmented_raw_data.T
            segmented_raw_data_T_tolist = segmented_raw_data_T.tolist()
            segment_list += segmented_raw_data_T_tolist
        self.new = np.array(segment_list)
        self.new_raw_data = mne.io.RawArray(self.new.T, self.raw_data.info)# 可以画图的raw
        self.new_trial_info = dict(self.new_trial)
        self.new_trial_array = []
        for i in range(len(self.tasks_start)):
            new_trial_data = list(self.new_trial_info.items())
            new_trial_data_list = []
            new_trial_data_list.append(new_trial_data[i])
            self.segmented_new_trial_info = dict(new_trial_data_list)
            self.new_trial_array.append(self.segmented_new_trial_info)

    def task_segmentation(self):
        data = self.new_raw_data.get_data()
        for key, value in self.new_trial_info.items():
            if ',' in str(value[0]):
                start = value[0].split(',')
                end = value[1].split(',')
                temp = data[:, int(float(start[0])):int(float(end[0]))]
                for i in range(1, len(start)):
                    temp = np.concatenate((temp, data[:, int(float(start[i])):int(float(end[i]))]), axis=1)
            else:
                temp = data[:, int(float(value[0])):int(float(value[1]))]
            self.new_tasks_data[key] = temp


    def init_preprocessing(self, raw, tasks, trial_info, tasks_cleaned):
        self.preprocessing = PreProcessPipeline(raw, tasks, trial_info, tasks_cleaned)

