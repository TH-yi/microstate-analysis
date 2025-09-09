#!/usr/bin/env python
# -*- coding: utf-8 -*-
# EEG signals respond differently to idea generation, idea evolution, and evaluation in a loosely controlled creativity experiment
# Time    : 2019-10-10
# Author  : Wenjun Jia
# File    : microstate.py

import numpy as np
from scipy import signal
from microstate_analysis.eeg_tool.math_utilis import zero_mean, condensed_to_square
import matplotlib.pyplot as plt
from scipy import stats
from scipy import spatial
from microstate_analysis.eeg_tool.utilis import read_subject_info, read_info, get_root_path
import codecs, json
from collections import OrderedDict
import mne
from multiprocessing import Pool
import matplotlib.gridspec as gridspec
from microstate_analysis.eeg_tool.utilis import write_info
import itertools
from operator import itemgetter, attrgetter
from statsmodels.stats.multitest import multipletests
from sklearn import preprocessing
import os
from scipy.io import loadmat

class Microstate:
    def __init__(self, data):
        self.data = data.T
        self.data = zero_mean(self.data, 1)
        self.n_t = self.data.shape[0]
        self.n_ch = self.data.shape[1]
        self.gfp = None
        self.peaks = None

        self.cv = None
        self.maps = None
        self.label = None

        # All data
        self.label = None
        self.gev = None
        self.correlation = None
        self.duration = None
        self.coverage = None
        self.opt_k = -1
        self.opt_k_index = -1

        #All data from min_maps to max_maps
        self.cv_list = []
        self.gev_list = []
        self.maps_list = []
        self.label_list = []

    @staticmethod
    def orthogonal_dist(v, eeg_map):
        return np.sum(v ** 2) - np.sum(np.sum((eeg_map * v), axis=1) ** 2)

    @staticmethod
    def normalization(v, axis=1):
        return v / np.linalg.norm(v, axis=axis, keepdims=True)

    @staticmethod
    def global_dissimilarity(spatial_correlation):
        return np.sqrt(2 * (1 - spatial_correlation))

    def global_explained_variance(self, n_maps, correlation, label, data_std):
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            gev[k] = np.sum(data_std[label == k] ** 2 * correlation[label == k, k]**2) / np.sum(data_std[label != -1]**2)
        return gev

    @staticmethod
    def global_explained_variance_sum(data, maps, distance=10, n_std=3, polarity=False):
        n_maps = maps.shape[0]
        n_ch = maps.shape[1]
        gev_peaks = np.zeros(n_maps)
        gev_raw = np.zeros(n_maps)
        gfp = data.std(axis=1)
        peaks, _ = signal.find_peaks(gfp, distance=distance, height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
        correlation_peaks = Microstate.spatial_correlation(data[peaks], zero_mean(maps, 1).T, gfp[peaks], maps.std(axis=1), n_ch)
        correlation_raw = Microstate.spatial_correlation(data, zero_mean(maps, 1).T, gfp, maps.std(axis=1), n_ch)
        correlation_peaks = correlation_peaks if polarity else abs(correlation_peaks)
        correlation_raw = correlation_raw if polarity else abs(correlation_raw)
        label_peaks = np.argmax(correlation_peaks, axis=1)
        label_raw = np.argmax(correlation_raw, axis=1)
        for k in range(n_maps):
            gev_peaks[k] = np.sum(gfp[peaks][label_peaks == k] ** 2 * correlation_peaks[label_peaks == k, k]**2) / np.sum(gfp[peaks]**2)
            gev_raw[k] = np.sum(gfp[label_raw == k] ** 2 * correlation_raw[label_raw == k, k]**2) / np.sum(gfp**2)
        return np.sum(gev_peaks), np.sum(gev_raw)


    @staticmethod
    def opt_microstate_criteria(cvs):
        k_opt = np.argmin(cvs)
        return k_opt

    @staticmethod
    def max_evec(v, axis):
        data = np.dot(v.T, v)
        evals, evecs = np.linalg.eig(data)
        c = evecs[:, np.argmax(np.abs(evals))]
        c = np.real(c)
        return Microstate.normalization(c, axis)

    @staticmethod
    def spatial_correlation(averaged_v1, averaged_v2, std_v1, std_v2, n_ch):
        correlation = np.dot(averaged_v1, averaged_v2) / (n_ch * np.outer(std_v1, std_v2))
        return correlation

    @staticmethod
    def fit_back(data, maps, distance=10, n_std=3, polarity=False, instantaneous_eeg=False):
        if instantaneous_eeg:
            correlation = Microstate.spatial_correlation(data, zero_mean(maps, 1).T, data.std(axis=1),
                                                         maps.std(axis=1), data.shape[1])
            correlation = correlation if polarity else abs(correlation)
            label = np.argmax(correlation,axis=1)
            return label

        gfp = data.std(axis=1)
        peaks, _ = signal.find_peaks(gfp, distance=distance, height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
        label = np.full(data.shape[0], -1)
        correlation = Microstate.spatial_correlation(data[peaks], zero_mean(maps, 1).T, gfp[peaks], maps.std(axis=1), data.shape[1])
        correlation = correlation if polarity else abs(correlation)
        label_peaks = np.argmax(correlation, axis=1)
        for i in range(len(peaks)):
            if i == 0:
                previous_middle = 0
                next_middle = int((peaks[i] + peaks[i + 1]) / 2)
            elif i == len(peaks) - 1:
                previous_middle = int((peaks[i] + peaks[i - 1]) / 2)
                next_middle = len(label)
            else:
                previous_middle = int((peaks[i] + peaks[i - 1]) / 2)
                next_middle = int((peaks[i] + peaks[i + 1]) / 2)
            label[previous_middle:next_middle] = label_peaks[i]
        return label

    @staticmethod
    def microstates_duration(label, n_maps):
        duration = [[] for i in range(n_maps)]
        j = label[0]
        count = 1
        for i in range(1, len(label)):
            if j != label[i]:
                duration[j].append(count)
                j = label[i]
                count = 1
            elif j == label[i]:
                count += 1
        for i, temp in enumerate(duration):
            if i == label[0]:
                duration[i] = np.asarray(temp[1::]).mean() if len(temp[1::]) > 0 else 0
            elif i == label[-1]:
                duration[i] = np.asarray(temp[0:-1]).mean() if len(temp[0:-1]) > 0 else 0
            else:
                duration[i] = np.asarray(temp).mean() if len(temp) > 0 else 0

        return duration

    @staticmethod
    def microstates_occurrence(label, n_maps, sfreq):
        occurence = [[] for i in range(n_maps)]
        for i in range(0, len(label), sfreq):
            temp = np.asarray(label[i:i + sfreq])
            for j in range(n_maps):
                occurence[j].append(np.argwhere(temp == j).shape[0])
        for i, temp in enumerate(occurence):
            occurence[i] = np.mean(temp) if len(temp) > 0 else 0
        return occurence

    @staticmethod
    def microstates_coverage(label, n_maps):
        coverage = []
        n_label = len(label)
        for i in range(n_maps):
            coverage.append(np.argwhere(label == i).shape[0] / n_label)
        return coverage

    @staticmethod
    def microstates_parameters(data, maps, distance=10, n_std=3, polarity=False, sfreq=500, epoch=2):
        n_maps = len(maps)
        res = {'duration': [], 'occurrence': [], 'coverage': []}
        data = zero_mean(data.T, 1)
        for i in range(0, data.shape[0], sfreq * epoch):
            data_epoch = data[i: i + sfreq * epoch]
            label = Microstate.fit_back(data_epoch, maps, distance, n_std, polarity)
            res['duration'].append(Microstate.microstates_duration(label, n_maps))
            res['occurrence'].append(Microstate.microstates_occurrence(label, n_maps, sfreq))
            res['coverage'].append(Microstate.microstates_coverage(label, n_maps))
        return res


    def cross_validation(self, var, n_maps):
        return var * (self.n_ch - 1) ** 2 / (self.n_ch - n_maps - 1.) ** 2

    def variance(self, label, maps, data):
        var = np.sum(data ** 2) - np.sum(np.sum(maps[label, :] * data, axis=1) ** 2)
        var /= (self.n_t * (self.n_ch-1))
        return var

    def gfp_peaks(self, distance=10, n_std=3):
        gfp = self.data.std(axis=1)
        peaks, _ = signal.find_peaks(gfp, distance=distance, height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
        self.peaks = peaks
        self.gfp = gfp

    def kmeans_modified(self, data, data_std, n_runs=10, n_maps=4, maxerr=1e-6, maxiter=1000, polarity=False):
        n_gfp = data_std.shape[0]
        cv_list = []
        gev_list = []
        maps_list = []
        label_list = []
        for run in range(n_runs):
            rndi = np.random.permutation(n_gfp)[:n_maps]
            maps = Microstate.normalization(data[rndi, :], axis=1)
            n_iter = 0
            var0 = 1.0
            var1 = 0.0
            while ((np.abs(var0 - var1) / var0 > maxerr) & (n_iter < maxiter)):
                label = np.argmax(np.dot(data, maps.T) ** 2, axis=1)
                for k in range(n_maps):
                    data_k = data[label == k, :]
                    maps[k, :] = Microstate.max_evec(data_k, 0)
                var1 = var0
                var0 = Microstate.orthogonal_dist(data, maps[label, :])
                var0 /= (n_gfp * (self.n_ch - 1))

            label, correlation, cv, gev = self.optimize_k(maps=maps, data=data, data_std=data_std, polarity=polarity)
            cv_list.append(cv)
            gev_list.append(gev)
            maps_list.append(maps)
            label_list.append(label)
            opt = Microstate.opt_microstate_criteria(cv_list)
        return cv_list[opt], gev_list[opt], maps_list[opt], label_list[opt]

    def opt_microstate(self, min_maps=2, max_maps=10, distance=10, n_std=3, n_runs=100, maxerr=1e-6, maxiter=1000, polarity=False, peaks_only=True, method='kmeans_modified'):
        self.gfp_peaks(distance=distance, n_std=n_std)
        if peaks_only:
            temp_data = self.data[self.peaks]
            temp_data_std = self.gfp[self.peaks]
            temp_max_maps = min(temp_data.shape[0], max_maps)
        else:
            temp_data = self.data
            temp_data_std = self.gfp
            temp_max_maps = min(temp_data.shape[0], max_maps)

        if method == 'kmeans_modified':
            for n_maps in range(min_maps, temp_max_maps+1):
                if n_maps == min_maps or n_maps == max_maps:
                    print("kmeans_number:{number}".format(number=n_maps))
                cv, gev, maps, label = self.kmeans_modified(data=temp_data, data_std=temp_data_std, n_runs=n_runs, n_maps=n_maps, maxerr=maxerr, maxiter=maxiter, polarity=polarity)
                self.cv_list.append(cv)
                self.gev_list.append(gev)
                self.maps_list.append(maps)
                self.label_list.append(label)

        elif method == 'aahc':
            self.cv_list, self.gev_list, self.maps_list, self.label_list = self.aahc(data=temp_data, data_std=temp_data_std, min_maps=min_maps, max_maps=temp_max_maps, polarity=polarity)

        self.opt_k_index = Microstate.opt_microstate_criteria(self.cv_list)

        self.cv = self.cv_list[self.opt_k_index]
        self.gev = self.gev_list[self.opt_k_index]
        self.maps = self.maps_list[self.opt_k_index]
        self.label = self.label_list[self.opt_k_index]

        self.opt_k = len(self.gev)

        self.gev_list = [temp.tolist() for temp in self.gev_list]
        self.maps_list = [temp.tolist() for temp in self.maps_list]
        self.label_list = [temp.tolist() for temp in self.label_list]


    def optimize_k(self, maps, data=None, data_std=None, polarity=False):
        if data is None and data_std is None:
            data = self.data
            data_std = self.gfp
        n_maps = len(maps)
        correlation = Microstate.spatial_correlation(data, zero_mean(maps, 1).T, data_std, maps.std(axis=1),self.n_ch)
        correlation = correlation if polarity else abs(correlation)
        label = np.argmax(correlation, axis=1)
        var = self.variance(label=label, maps=maps, data=data)
        cv = self.cross_validation(var, n_maps)
        gev = self.global_explained_variance(n_maps=n_maps, correlation=correlation, label=label, data_std=data_std)
        return label, correlation, cv, gev



    def aahc(self, data, data_std, min_maps, max_maps, polarity=False):
        maps = data
        n_maps = len(maps)
        label_list = [[k] for k in range(len(maps))]
        cv_list = []
        gev_list = []
        maps_list = []
        res_label_list = []
        while n_maps > (min_maps - 1):
            print("n_maps:%d" % n_maps)
            correlation = Microstate.spatial_correlation(data, zero_mean(maps, 1).T, data_std, maps.std(axis=1),self.n_ch)
            correlation = correlation if polarity else abs(correlation)
            label = np.argmax(correlation, axis=1)
            gev = self.global_explained_variance(n_maps, correlation, label, data_std)
            if max_maps >= n_maps >= min_maps:
                var = self.variance(label, maps, data=data)
                cv = self.cross_validation(var, n_maps)
                cv_list.append(cv)
                maps_list.append(maps)
                gev_list.append(gev)
                res_label_list.append(label)
                if n_maps == min_maps:
                    break
            excluded_k = np.argmin(gev)
            maps = np.vstack((maps[:excluded_k, :], maps[excluded_k+1:, :]))
            re_label = label_list.pop(excluded_k)
            re_cluster = []
            for k in re_label:
                correlation = Microstate.spatial_correlation(data[k, :], zero_mean(maps, 1).T, data_std[k], maps.std(axis=1),self.n_ch)
                correlation = correlation if polarity else abs(correlation)
                new_label = np.argmax(correlation)
                re_cluster.append(new_label)
                label_list[new_label].append(k)

            for i in re_cluster:
                idx = label_list[i]
                maps[i] = Microstate.max_evec(data[idx, :], 0)
            n_maps = len(maps)

        return cv_list[::-1], gev_list[::-1], maps_list[::-1], res_label_list[::-1]


class MeanMicrostate:
    def __init__(self, data, n_k, n_ch, n_condition):
        self.data = data
        self.n_k = n_k
        self.n_ch = n_ch
        self.n_condition = n_condition
        data_concatenate = np.zeros((1, n_ch))
        for i in range(n_condition):
            data_concatenate = np.concatenate((data_concatenate, data[i]), axis=0)

        self.data_concatenate = data_concatenate[1::, :]

    def label_two_microstates(self, microstates, mean_microstates, polarity=False):
        similarity_matrix = np.zeros((self.n_k, self.n_k))
        for i in range(self.n_k):
            for j in range(self.n_k):
                similarity_matrix[i, j] = stats.pearsonr(microstates[i], mean_microstates[j])[0]

        comb = [zip(perm, [i for i in range(self.n_k)]) for perm in itertools.permutations([j for j in range(self.n_k)], self.n_k)]
        res = []
        for i, item in enumerate(comb):
            s = 0
            comb_list = []
            sign = []
            for item_j in item:
                comb_list.append(item_j)
                similarity = similarity_matrix[item_j[0], item_j[1]]
                if similarity < 0:
                    sign.append(-1)
                else:
                    sign.append(1)
                s = s + similarity if polarity else s + abs(similarity)
            res.append((s/len(comb_list), comb_list, sign))
        sorted_res = sorted(res, key=itemgetter(0), reverse=True)
        return sorted_res[0]

    def label_microstates(self, mul_microstates, mean_microstates, polarity=False):
        label = []
        sign = []
        similarity = []
        for microstates in mul_microstates:
            s = self.label_two_microstates(microstates, mean_microstates, polarity)
            for index, item in enumerate(s[1]):
                label.append(item[0])
            sign.extend(s[2])
            similarity.append(s[0])
        return label, sign, np.mean(similarity), np.std(similarity)

    def reorder_microstates(self, mul_microstates, mean_microstates, polarity=False):
        res = []
        for microstates in mul_microstates:
            s = self.label_two_microstates(microstates, mean_microstates, polarity)
            sorted_index = [i[0] for i in s[1]]
            sign = np.repeat(s[2], self.n_ch).reshape(-1,self.n_ch)
            microstate_updated = np.asarray(microstates)[sorted_index] * sign
            res.append(microstate_updated.tolist())

        return res

    def update_mean_microstates(self, label, sign, polarity=False):
        label = np.asarray(label)
        mean_microsate_updated = np.zeros((self.n_k, self.n_ch))
        for i in range(self.n_k):
            index = np.argwhere(label == i).reshape(-1)
            maps = self.data_concatenate[index, :]
            # if not polarity:
            temp = np.asarray(sign)[index].reshape(self.n_condition, 1)
            temp = np.repeat(temp, self.n_ch, axis=1)
            maps = maps * temp
            mean_microsate_updated[i, :] = np.mean(maps, axis=0)
        return mean_microsate_updated

    def mean_microstates(self, n_runs=100, maxiter=100):
        n_data_concatenate = len(self.data_concatenate)
        maps_list = []
        label_list = []
        mean_similarity_list = []
        std_similarity_list = []
        for run in range(n_runs):
            print("current run: ", run, "total: ", n_runs)
            mean_similarity_run = []
            std_similarity_run = []
            label_run = []
            maps_run = []
            rndi = np.random.permutation(n_data_concatenate)[:self.n_k]
            # maps = Microstate.normalization(self.data_concatenate[rndi, :], axis=1)

            maps = self.data_concatenate[rndi, :]
            label, sign, mean_similarity, std_similarity = self.label_microstates(self.data, maps, False)
            iter_num = 0
            while iter_num < maxiter:
                iter_num += 1
                maps_updated = self.update_mean_microstates(label, sign, False)
                label_updated, sign, mean_similarity, std_similarity = self.label_microstates(self.data, maps_updated, False)
                if label == label_updated:
                    maps_list.append(maps_updated)
                    label_list.append(label)
                    mean_similarity_list.append(mean_similarity)
                    std_similarity_list.append(std_similarity)
                    break
                else:
                    mean_similarity_run.append(mean_similarity)
                    std_similarity_run.append(std_similarity)
                    label_run.append(label_updated)
                    maps_run.append(maps_updated)
                    label = label_updated
            else:
                index = np.argmax(np.asarray(mean_similarity_run))
                mean_similarity_list.append(mean_similarity_run[index])
                std_similarity_list.append(std_similarity_run[index])
                label_list.append(label_run[index])
                maps_list.append(maps_run[index])

        index = np.argmax(mean_similarity_list)
        return maps_list[index], label_list[index], mean_similarity_list[index], std_similarity_list[index]


def batch_microstate_parameters(para):
    data = para[0]
    maps = para[1]
    distance = para[2]
    n_std = para[3]
    polarity = para[4]
    sfreq = para[5]
    epoch = para[6]
    _ = Microstate(np.asarray(data))
    return Microstate.microstates_parameters(np.asarray(data), np.asarray(maps), distance, n_std, polarity, sfreq, epoch)

def batch_microstate_state(data, topographies, task_name):
    for a_task_name in task_name:
        microstate = Microstate(data[a_task_name])
        microstate.gfp_peaks()
        microstate.maps = np.asarray(topographies[a_task_name]["maps_list"][topographies[a_task_name]["opt_k_index"]])
        microstate.microstate_state()

def batch_order_mean_microstate(para):
    data = para[0]
    n_k = para[1]
    n_ch = para[2]
    n_condition = para[3]
    mean_microstate = para[4]
    microstate = MeanMicrostate(data, n_k, n_ch, n_condition)
    res = microstate.reorder_microstates(microstate.data, mean_microstate, polarity=False)
    return res

def batch_mean_microstate(para):
    data = para[0]
    n_k = para[1]
    n_ch = para[2]
    n_condition = para[3]
    microstates = MeanMicrostate(data, n_k, n_ch, n_condition)
    eegmaps, label, mean_similarity, std_similarity = microstates.mean_microstates()
    return {"maps": eegmaps, "label": label, "mean_similarity": mean_similarity, "std_similarity": std_similarity}


def batch_microstate(para):
    data = para[0]
    peaks_only = para[1]
    min_maps = para[2]
    max_maps = para[3]
    microstate = Microstate(data)
    microstate.opt_microstate(min_maps, max_maps, n_std=3, n_runs=100, peaks_only=peaks_only, method='kmeans_modified')
    return microstate


def filter_data(data):
    return mne.filter.filter_data(data, 500, 1., 30.)


def batch_filter(clean_data_fname, subjects, data_fname, data_fname_save_dict, data_fname_save, task_name):
    for subject in subjects:
        print(subject)
        res = OrderedDict()
        data_text = codecs.open(clean_data_fname + "\\" + subject + data_fname, 'r', encoding='utf-8').read()
        data = json.loads(data_text)
        pool = Pool(8)
        multi_res = [pool.apply_async(filter_data, (data[a_task_name]['task_data'],)) for a_task_name in task_name]
        pool.close()
        pool.join()
        for i in range(len(task_name)):
            res[task_name[i]] = multi_res[i].get().tolist()
        json.dump(res, codecs.open(data_fname_save_dict + "\\" + subject + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def eegmaps_parameters(clean_data_fname, subjects, data_fname, eegmaps_fname, clean_data_fname_save, data_fname_save, task_name):
    eegmaps = load_data(eegmaps_fname)
    eegmaps = eegmaps['maps'] if 'maps' in eegmaps else eegmaps
    for subject in subjects:
        print(subject)
        temp_res = []
        res = {}
        pool = Pool(len(task_name))
        data = load_data(clean_data_fname + "\\" + subject + data_fname)
        for a_task_name in task_name:
            temp_res.append(pool.apply_async(batch_microstate_parameters, ([data[a_task_name], eegmaps, 10, 3, False, 500, 2],)))
        pool.close()
        pool.join()
        for i, a_task_name in enumerate(task_name):
            res[a_task_name] = temp_res[i].get()
        json.dump(res, codecs.open(clean_data_fname_save + "\\" + subject + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def eegmaps_parameters_across_runs(clean_data_fname, subjects, data_fname, data_fname_save, condition_name, tasks_name):
    for subject in subjects:
        print(subject)
        data = load_data(clean_data_fname + "\\" + subject + data_fname)
        res = {}
        for i, task_name in enumerate(tasks_name):
            duration = []
            coverage = []
            occurrence = []
            res[condition_name[i]] = {}
            for a_task_name in task_name:
                duration.extend(data[a_task_name]["duration"])
                coverage.extend(data[a_task_name]["coverage"])
                occurrence.extend(data[a_task_name]["occurrence"])
            res[condition_name[i]]['duration'] = exclude_zero_mean(np.asarray(duration)).tolist()
            res[condition_name[i]]['coverage'] = np.asarray(coverage).mean(axis=0).tolist()
            res[condition_name[i]]['occurrence'] = exclude_zero_mean(np.asarray(occurrence)).tolist()
        json.dump(res, codecs.open(clean_data_fname + "\\" + subject + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def exclude_zero_mean(data):
    sum = np.sum(data, axis=0)
    col = (data != 0).sum(0)
    return sum/col

def eegmaps_individual_run(clean_data_fname, subjects, data_fname, data_fname_save_dict, data_fname_save, task_name, min_maps, max_maps, peaks_only=True):
    for subject in subjects:
        print(subject)
        res = OrderedDict()
        pool = Pool(11)
        multi_res = []
        data = load_data(clean_data_fname + "\\" + subject + data_fname)
        data = combine_task_data(data, task_name)
        for i in range(len(task_name)):
            multi_res.append(pool.apply_async(batch_microstate, ([np.asarray(data[task_name[i]]), peaks_only, min_maps[i], max_maps[i]],)))
        for i in range(len(task_name)):
            temp = multi_res[i].get()
            res[task_name[i]] = {'cv_list': temp.cv_list, 'gev_list': temp.gev_list, 'maps_list': temp.maps_list,
                                 'opt_k': temp.opt_k, 'opt_k_index': int(temp.opt_k_index), 'min_maps': min_maps[i],
                                 'max_maps': max_maps[i]}
        json.dump(res, codecs.open(data_fname_save_dict + "\\" + subject + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'))

def eegmaps_across_runs(clean_data_fname, subjects, data_fname, data_fname_save_dict, data_fname_save, task_name, condition_name, n_k, n_k_index, n_ch):
    conditions = {}
    res = OrderedDict()
    pool = Pool(len(condition_name))
    for condition in condition_name:
        conditions[condition] = []
        for task in task_name:
            if task.startswith(condition):
                conditions[condition].append(task)
    for subject in subjects:
        print(subject)
        data = load_data(clean_data_fname + "\\" + subject + data_fname)
        condition_res = []
        for condition in condition_name:
            maps = [data[a_task_name]['maps'][n_k_index] for a_task_name in conditions[condition]]
            condition_res.append(pool.apply_async(batch_mean_microstate, ([maps, n_k, n_ch, len(conditions[condition])],)))
        for index, condition in enumerate(condition_name):
            temp = condition_res[index].get()
            res[condition] = {'maps': temp["maps"].tolist(),'mean_similarity': temp['mean_similarity'],'std_similarity': temp['std_similarity']}
        json.dump(res, codecs.open(data_fname_save_dict + "\\" + subject + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)


def eegmpas_across_subjects(clean_data_fname, subjects, data_fname, data_fname_save_dict, data_fname_save, condition_name, n_k, n_ch):
    res = OrderedDict()
    condition_res = []
    n_condition = len(condition_name)
    pool = Pool(n_condition)
    for a_condition_name in condition_name:
        maps = []
        for subject in subjects:
            data = load_data(clean_data_fname + "\\" + subject + data_fname)
            maps.append(data[a_condition_name]['maps'])
        condition_res.append(pool.apply_async(batch_mean_microstate, ([maps, n_k, n_ch, len(subjects)],)))
    for i, a_condition_name in enumerate(condition_name):
        temp = condition_res[i].get()
        res[a_condition_name] = {'maps': temp["maps"].tolist(), 'label': temp["label"], 'mean_similarity': temp['mean_similarity'], 'std_similarity': temp['std_similarity']}
    json.dump(res, codecs.open(data_fname_save_dict + "\\" + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def eegmaps_across_conditions(clean_data_fname, data_fname, data_fname_save_dict, data_fname_save, condition_name, n_k, n_ch):
    maps = []
    for a_condition_name in condition_name:
        data = load_data(clean_data_fname + "\\" + data_fname)
        maps.append(data[a_condition_name]['maps'])
    microstate = MeanMicrostate(maps, n_k, n_ch, len(condition_name))
    temp = microstate.mean_microstates()
    res = {'maps': temp[0].tolist(), 'label': temp[1], 'mean_similarity': temp[2], 'std_similarity': temp[3]}
    json.dump(res, codecs.open(data_fname_save_dict + "\\" + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def create_info():
    ch = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
    montage = 'D:\\workspace\\eeg_tool\\Cap63.locs'
    info = mne.create_info(ch_names=ch, sfreq=500, ch_types='eeg', montage=os.path.join(get_root_path(),'Cap63.locs'))
    return info

def plot_eeg(data, condition_name, n_k):
    info = create_info()
    for k in range(n_k):
        ax_all = gridspec.GridSpec(10, 5, figure=plt.figure(figsize=(10, 100)))
        for a_condition_name in condition_name:
            maps = data[a_condition_name]
            for row in range(10):
                for col in range(5):
                    ax = plt.subplot(ax_all[row, col])
                    mne.viz.plot_topomap(np.asarray(maps[row][k]), info, show=False, axes=ax, image_interp='spline36', contours=6)
        plt.show()

def plot_eegmap_conditions(eegmaps, condition_name, n_k, order=None, sign=None, title=None, ylabel=None, savepath=None):
    info = create_info()
    n_condition = len(condition_name)
    row = n_condition
    col = n_k
    ax_all = gridspec.GridSpec(row, col, figure=plt.figure(figsize=(25, 10)))
    order = order if order else [i for i in range(col)]
    sign = sign if sign else [1 for i in range(col)]
    for i in range(row):
        maps = np.asarray(eegmaps[condition_name[i]])[order]
        for j in range(n_k):
            ax = plt.subplot(ax_all[i, j])
            temp = (maps[j] - maps[j].mean()) * sign[j]
            mne.viz.plot_topomap(temp, info, show=False, axes=ax, image_interp='spline36', contours=6)
            if i == 0:
                ax.set_title(title[j], fontsize=30,)
            if j == 0:
                x = ax.get_xlim()[0] - 2.1
                y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2 -0.15
                ax.text(x, y, ylabel[i], fontsize=30,)
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


def plot_eegmap_one_row(maps, order=None, sign=None, savepath=None, title=None):
    info = create_info()
    col = maps.shape[0]
    ax_all = gridspec.GridSpec(1, col, figure=plt.figure(figsize=(16, 10)))
    order = order if order else [i for i in range(col)]
    sign = sign if sign else [1 for i in range(col)]
    for index, j in enumerate(order):
        ax = plt.subplot(ax_all[index])
        # temp = (maps[j] - maps[j].mean()) * sign[index]
        temp = maps[j] * sign[index]
        mne.viz.plot_topomap(temp, info, show=False, axes=ax, image_interp='spline36', contours=6)
        if title:
            ax.set_title(title[index], fontsize=30)
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()



def plot_eegmaps(maps, task_name):
    info = create_info()
    min_value = maps[0].min()
    max_value = maps[0].max()
    if len(task_name) == 1:
        ax_all = gridspec.GridSpec(1, len(task_name), figure=plt.figure(figsize=(10, 100)))
        for j in range(len(maps[0])):
            ax = plt.subplot(ax_all[j])
            mne.viz.plot_topomap(maps[0].T[:, j], info, show=False, axes=ax, image_interp='spline36', contours=6, vmin=min_value, vmax=max_value)
        plt.show()
    else:
        similarity = []
        row = len(task_name)
        col = 0
        for i in range(1, len(task_name)):
            temp = eegmaps_similarity(maps[0], maps[i])
            similarity.append(temp)
            col = max(col, max([item['index'] for item in [*temp.values()]])+1)
        ax_all = gridspec.GridSpec(row, col, figure=plt.figure(figsize=(10, 100)))
        ax_all.update(hspace=0.5)
        for i in range(row):
            for j in range(len(maps[i])):
                if i == 0:
                    ax = plt.subplot(ax_all[i, j])
                else:
                    ax = plt.subplot(ax_all[i, similarity[i-1][j]['index']])
                    ax.set_title(round(similarity[i-1][j]['similarity'], 4), fontsize=5)
                mne.viz.plot_topomap(maps[i].T[:, j], info, show=False, axes=ax, image_interp='spline36', contours=6, vmin=min_value, vmax=max_value)
        plt.show()


def eegmaps_similarity_across_runs(clean_data_fname, subjects, data_fname, task_name, opt_k_index):
    res = []
    for subject in subjects:
        data_text = codecs.open(clean_data_fname + "\\" + subject + data_fname, 'r', encoding='utf-8').read()
        data = json.loads(data_text)
        res_temp = []
        for a_task_name in task_name:
            for comb in itertools.combinations([i for i in range(1, len(task_name)+1)], 2):
                similarity = eegmaps_similarity(np.asarray(data[str(comb[0]) + "_" + a_task_name]["maps_list"][opt_k_index]), np.asarray(data[str(comb[1]) + "_" + a_task_name]["maps_list"][opt_k_index]), 0.0)
                averaged_similarity = eegmaps_averaged_similarity(similarity)
                res_temp.append(averaged_similarity)
        res.append(res_temp)
    return res



def eegmaps_similarity_highest_comb(data):
    data = sorted(data, key=itemgetter(1), reverse=True)
    res = []
    for item in data:
        for item_res in res:
            if len(set(item[0]).intersection(set(item_res[0]))) != 0:
                break
        else:
            res.append(item)
    return res

def eegmaps_similarity_highest_merged(comb, eegmaps, n_runs, opt_k):
    res_eegmaps = []
    for item in comb:
        res = []
        for i in range(len(item[0])):
            temp = 0
            for j in range(len(item[0])):
                temp += abs(stats.pearsonr(eegmaps[i][item[0][i] % n_runs], eegmaps[j][item[0][j] % n_runs])[0])
            res.append(temp/len(item[0]))
        index = np.argmax(np.asarray(res))
        res_eegmaps.append(eegmaps[task_index_runs(item[0][index], opt_k)-1][item[0][index]%opt_k].tolist())
    return res_eegmaps


def task_index_runs(index, opt_k):
    if index < opt_k:
        return 1
    elif opt_k <= index < 2*opt_k:
        return 2
    elif 2*opt_k <= index < 3*opt_k:
        return 3

def eegmaps_averaged_similarity(similarity):
    res = 0
    for key, value in similarity.items():
        res += value['similarity']
    return res / len(similarity)


def eegmaps_similarity(baseline, task, threshold=0.0):
    res = {}
    similarity = np.zeros((baseline.shape[0], task.shape[0]))
    for i in range(baseline.shape[0]):
        for j in range(task.shape[0]):
            similarity[i][j] = abs(stats.pearsonr(baseline[i, :], task[j, :])[0])
    iter = min(baseline.shape[0], task.shape[0])
    while iter != 0:
        max_similarity = np.max(similarity)
        if max_similarity < threshold:
            iter -= 1
            continue
        temp = np.argwhere(similarity == max_similarity)
        row = temp[0][0]
        col = temp[0][1]
        res[col] = {'index': row, 'similarity': max_similarity}
        iter -= 1
        similarity[row, ::] = -3
        similarity[::, col] = -3
    if len(res) != task.shape[0]:
        task_index = [*res]
        diff = list(set([i for i in range(task.shape[0])]) - set(task_index))
        temp = baseline.shape[0]
        for i in diff:
            res[i] = {'index': temp, 'similarity': -3}
            temp += 1
    return res


def concatenate_data_by_condition(data, task_condition, exclude_task):
    res = OrderedDict()
    for condition in task_condition:
        for task_name, task_data in data.items():
            if task_name not in exclude_task:
                if condition == task_name.split("_")[1]:
                    res[condition] = np.asarray(task_data) if condition not in res else np.concatenate((res[condition], np.asarray(task_data)), axis=1)
    return res

def opt_microstate_tasks(data, task_name, min_maps, max_maps, peaks_only=True):
    res = OrderedDict()
    pool = Pool(11)
    multi_res = []
    for i in range(len(task_name)):
        multi_res.append(pool.apply_async(batch_microstate, ([np.asarray(data[task_name[i]]), peaks_only, min_maps[i], max_maps[i]],)))
    pool.close()
    pool.join()
    for i in range(len(task_name)):
        temp = multi_res[i].get()
        res[task_name[i]] = {'cv_list': temp.cv_list, 'gev_list': temp.gev_list, 'maps_list': temp.maps_list,
                             'opt_k': temp.opt_k, 'opt_k_index': int(temp.opt_k_index), 'min_maps': min_maps[i],
                             'max_maps': max_maps[i]}
    return res

def opt_microstate_across_subjects(clean_data_fname, subjects, data_fname, data_fname_save, task_name, peaks_only=False):
    mean_opt_k = microstate_stat(clean_data_fname + "\\" + "_1_30_microstate_stat.json")
    task_data = OrderedDict()
    for a_task_name in task_name:
        for subject in subjects:
            data_text = codecs.open(clean_data_fname + "\\" + subject + data_fname, 'r', encoding='utf-8').read()
            data = json.loads(data_text)
            opt_k_index = data[a_task_name]["opt_k_index"]
            temp = np.asarray(data[a_task_name]["maps_list"][opt_k_index]).T
            task_data[a_task_name] = temp if a_task_name not in task_data else np.concatenate((task_data[a_task_name], temp), axis=1)
    # res = opt_microstate_tasks(task_data, task_name, [1 for i in range(len(task_name))], [10 for i in range(len(task_name))], peaks_only)

    res = opt_microstate_tasks(task_data, task_name, [mean_opt_k[a_task_name]["mean_opt_k"] for a_task_name in task_name], [mean_opt_k[a_task_name]["mean_opt_k"] for a_task_name in task_name], peaks_only)
    json.dump(res, codecs.open(clean_data_fname + "\\" + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def opt_microstate_subjects_across_tasks(data, task_name, min_maps, max_maps, peaks_only=False):
    temp_data = np.zeros((63, 1))
    for a_task_name in task_name:
        opt_k_index = data[a_task_name]["opt_k_index"]
        temp = np.asarray(data[a_task_name]["maps_list"][opt_k_index]).T
        temp_data = np.concatenate((temp_data, temp), axis=1)
    temp = batch_microstate([temp_data[:, 1::], peaks_only, min_maps, max_maps])
    return {'cv_list': temp.cv_list, 'gev_list': temp.gev_list, 'maps_list': temp.maps_list,
            'opt_k': temp.opt_k, 'opt_k_index': int(temp.opt_k_index), 'min_maps': min_maps,
            'max_maps': max_maps}


def opt_microstate_across_subjects_across_tasks_across_runs(clean_data_fname, data_fname, data_fname_save, task_name, peaks_only=False):
    data_text = codecs.open(clean_data_fname + "\\" + data_fname, 'r', encoding='utf-8').read()
    data = json.loads(data_text)
    res = np.zeros((63, 1))
    for a_task_name in task_name:
        opt_k_index = data[a_task_name]["opt_k_index"]
        res = np.concatenate((res, np.asarray(data[a_task_name]["maps_list"][opt_k_index]).T), axis=1)

    temp = opt_microstate_tasks({'all_tasks': res[::, 1:]}, ['all_tasks'], [6], [6], peaks_only)
    json.dump(temp, codecs.open(clean_data_fname + "\\" + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def opt_microstate_subjects(clean_data_fname, subjects, data_fname, data_fname_save_dict, data_fname_save, task_name, min_maps, max_maps, across_tasks=False, peaks_only=False):
    for subject in subjects:
        print(subject)
        # data_text = codecs.open(clean_data_fname + "\\" + subject + data_fname, 'r', encoding='utf-8').read()
        data = load_data(clean_data_fname + "\\" + subject + data_fname)
        data = combine_task_data(data, task_name)
        res = opt_microstate_subjects_across_tasks(data, task_name, min_maps, max_maps, peaks_only) if across_tasks else opt_microstate_tasks(data, task_name, min_maps, max_maps, peaks_only)
        json.dump(res, codecs.open(data_fname_save_dict + "\\" + subject + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)


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


def save_eegmpas(clean_data_fname, data_fname, task_name):
    data_text = codecs.open(clean_data_fname + "\\" + data_fname, 'r', encoding='utf-8').read()
    data = json.loads(data_text)
    microstate_map = []
    for a_task_name in task_name:
        microstate_map.append(np.asarray(data[a_task_name]["maps_list"][data[a_task_name]["opt_k_index"]]))
    plot_eegmaps(microstate_map, task_name)

def save_eegmaps_subjects(clean_data_fname, data_fname, task_name, subjects):
    for subject in subjects:
        save_eegmpas(clean_data_fname, subject+data_fname, task_name)


def microstate_stat(clean_data_fname, subjects, data_fname, data_fname_save, task_name):
    res = OrderedDict()
    for subject in subjects:
        data_text = codecs.open(clean_data_fname + "\\" + subject + data_fname, 'r', encoding='utf-8').read()
        data = json.loads(data_text)
        for a_task_name in task_name:
            if a_task_name not in res:
                res[a_task_name] = {'opt_k': [], 'min_maps': data[a_task_name]["min_maps"], 'max_maps': data[a_task_name]["max_maps"]}
            res[a_task_name]['opt_k'].append(data[a_task_name]["opt_k"])
    for a_task_name in task_name:
        res[a_task_name]['mean_opt_k'] = int(np.asarray(res[a_task_name]['opt_k']).mean())
        res[a_task_name]['mean_opt_k_index'] = res[a_task_name]['mean_opt_k'] - int(res[a_task_name]['min_maps'])
    json.dump(res, codecs.open(clean_data_fname + "\\" + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'),sort_keys=True, indent=4)
    return res


def load_data(path):
    data_text = codecs.open(path, 'r', encoding='utf-8').read()
    data = json.loads(data_text)
    return data

def read_microstate_stat(data_fname):
    res = {}
    data_text = codecs.open(data_fname, 'r', encoding='utf-8').read()
    data = json.loads(data_text)
    for task_name, task_data in data.items():
        res[task_name] = {'mean_opt_k': task_data['mean_opt_k']}
    return res


def normalized_data(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # data[i][j] = (data[i][j]-np.mean(data[i][j])) / np.std(data[i][j])
            data[i][j] = data[i][j]-np.mean(data[i][j])
    return data

def generalized_dissimilarity(data):
    grand_mean_across_subjects = np.mean(data, axis=0)
    grand_mean_across_subjects_across_conditions = np.mean(data, axis=(0, 1))

    # grand_mean_across_subjects = (grand_mean_across_subjects - np.mean(grand_mean_across_subjects, axis=0)) / np.std(grand_mean_across_subjects, axis=0)
    # grand_mean_across_subjects_across_conditions = (grand_mean_across_subjects_across_conditions - np.mean(grand_mean_across_subjects_across_conditions)) / np.std(grand_mean_across_subjects_across_conditions)

    # s = 0
    # for i in range(grand_mean_across_subjects.shape[0]):
    #     s += (stats.pearsonr(grand_mean_across_subjects[i], grand_mean_across_subjects_across_conditions)[0])
    residual = np.power(grand_mean_across_subjects - grand_mean_across_subjects_across_conditions, 2)
    s = np.sum(np.sqrt(np.mean(residual, axis=1)))
    return s


def format_data(data, condition, n_subject, n_ch, ith_class):
    n_condition = len(condition)
    data_temp = np.zeros((n_subject, n_condition, n_ch))
    for i in range(n_subject):
        for j in range(n_condition):
            data_temp[i,j] = data[condition[j]][i][ith_class]
    return data_temp

def shuffle_data(data, n_condition):
    for i in range(data.shape[0]):
        random_index = np.random.permutation(n_condition)
        data[i] = data[i][random_index]
    return data

def diss(v_ij, v_j):
    return np.sum(np.sqrt(np.mean(np.power(v_ij - v_j, 2),1)))

def diss_interaction(level_mean, grand_mean):
    residual = np.power(level_mean -grand_mean, 2)
    return np.sum(np.sqrt(np.mean(residual, 2)))


def tanova(data, condition, n_subjects, n_c, n_k, n_ch, n_times):
    data_concatenation = np.zeros((n_c, n_subjects, n_k, n_ch))
    for i, a_condition_name in enumerate(condition):
        temp = np.asarray(data[a_condition_name])
        temp = temp - temp.mean(axis=2).reshape(n_subjects, n_k, 1)
        data_concatenation[i] = temp
    condition_mean = data_concatenation.mean(axis=2)
    class_mean = data_concatenation.mean(axis=0)
    grand_mean = data_concatenation.mean(axis=(0, 1, 2))
    grand_mean = (grand_mean - grand_mean.mean()) / np.std(grand_mean)

    condition_mean_subject = condition_mean.mean(1)
    class_mean_subject = class_mean.mean(0)
    condition_mean_subject = (condition_mean_subject - condition_mean_subject.mean(1, keepdims=True)) / np.std(condition_mean_subject, axis=1,keepdims=True)
    class_mean_subject = (class_mean_subject - class_mean_subject.mean(1, keepdims=True)) / np.std(class_mean_subject, axis=1,keepdims=True)

    # condition_class_mean = data_concatenation - class_mean_subject - np.tile(condition_mean_subject, n_k).reshape(n_c,1, n_k, n_ch)
    condition_class_mean = data_concatenation - class_mean_subject
    condition_class_mean_subject = condition_class_mean.mean(1, keepdims=True)
    condition_class_mean_subject = (condition_class_mean_subject - condition_class_mean_subject.mean(axis=(0, 1), keepdims=True)) / np.std(condition_class_mean_subject, axis=(0, 1), keepdims=True)

    count = 0
    condition_s = []
    class_s = []
    condition_class_s = []
    while count != n_times:
        condition_s.append(diss(condition_mean_subject, grand_mean))
        class_s.append(diss(class_mean_subject, grand_mean))
        condition_class_s.append(diss_interaction(condition_class_mean_subject, grand_mean))

        for i in range(n_subjects):
            condition_order = np.random.permutation(n_c)
            class_order = np.random.permutation(n_k)
            condition_mean[:, i, :] = condition_mean[condition_order, i, :]
            class_mean[i, :, :] = class_mean[i, class_order, :]
            data_concatenation[:, i] = data_concatenation[:, i, class_order]

        condition_mean_subject = condition_mean.mean(1)
        class_mean_subject = class_mean.mean(0)
        condition_mean_subject = (condition_mean_subject - condition_mean_subject.mean(1, keepdims=True)) / np.std(condition_mean_subject, axis=1, keepdims=True)
        class_mean_subject = (class_mean_subject - class_mean_subject.mean(1, keepdims=True)) / np.std(class_mean_subject, axis=1, keepdims=True)

        # condition_class_mean = data_concatenation - class_mean_subject - np.tile(condition_mean_subject, n_k).reshape(n_c, 1, n_k, n_ch)
        condition_class_mean = data_concatenation - class_mean_subject
        condition_class_mean_subject = condition_class_mean.mean(1, keepdims=True)
        condition_class_mean_subject = (condition_class_mean_subject - condition_class_mean_subject.mean(axis=(0, 1),keepdims=True)) / np.std(condition_class_mean_subject, axis=(0, 1), keepdims=True)
        count += 1

    res = [len(np.where(condition_s >= condition_s[0])[0]) / n_times, len(np.where(class_s >= class_s[0])[0]) / n_times,
           len(np.where(condition_class_s >= condition_class_s[0])[0]) / n_times]

    return res





def permutation_test_2_conditions(data, task_name, n_subjects, n_times):
    p_list = []
    for comb in itertools.combinations(task_name, 2):
        print(comb)
        res = []
        temp_task_name = [comb[0], comb[1]]
        for i in range(6):
            data_format = format_data(data, temp_task_name, n_subjects, 63, i)
            data_normalized = normalized_data(data_format)
            count = 0
            temp = []
            while count != n_times:
                temp.append(generalized_dissimilarity(data_normalized))
                data_normalized = shuffle_data(data_normalized, len(temp_task_name))
                count += 1
            temp = np.asarray(temp)
            res.append(len(np.where(temp >= temp[0])[0]) / n_times)
        p_list.append(res)
    return p_list

def permutation_test_3_conditions(data, task_name, n_subjects, n_times):
    res = []
    for i in range(6):
        data_format = format_data(data, task_name, n_subjects, 63, i)
        data_normalized = normalized_data(data_format)
        count = 0
        temp = []
        while count != n_times:
            temp.append(generalized_dissimilarity(data_normalized))
            data_normalized = shuffle_data(data_normalized, len(task_name))
            count += 1
        temp = np.asarray(temp)

        res.append(len(np.where(temp >= temp[0])[0]) / n_times)
    return res

def outliers_index(data, outlierConstant=1.5):
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    index = np.where((data >= quartileSet[0]) & (data <= quartileSet[1]))[0]
    return index.tolist()

def reorder_individual_eegmaps(global_maps, subjects, subjects_fname, subjects_sfname, task, k, k_index):
    for subject in subjects:
        res = {}
        maps = []
        for a_task in task:
            maps.append(load_data(subjects_fname+"\\"+subject+"_1_30_microstate.json")[a_task]["maps_list"][k_index])
        m = MeanMicrostate(maps, k, 63, len(task))
        temp = m.reorder_microstates(np.asarray(maps), global_maps)
        for index, a_task in enumerate(task):
            res[a_task] = temp[index]
        json.dump(res, codecs.open(subjects_sfname+"\\"+subject+"_1_30_microstate_reorder.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def fit_back(maps_fname, subjects, subjects_fname, subjects_sfname, task, is_global_maps, is_instantaneous=False):
    for subject in subjects:
        print(subject)
        pool = Pool(11)
        temp_res = []
        res = {}
        data = load_data(subjects_fname + "\\" + subject + '.json')
        data = combine_task_data(data, task)
        if is_global_maps:
            template = load_data(maps_fname)['maps']
            # template = load_data(maps_fname)
            for a_task in task:
                data_temp = zero_mean(np.asarray(data[a_task]).T, 1)
                temp_res.append(pool.apply_async(batch_fit_back, ([data_temp, np.asarray(template), is_instantaneous],)))
            pool.close()
            pool.join()
            for i, a_task in enumerate(task):
                res[a_task] = temp_res[i].get()
            json.dump(res, codecs.open(subjects_sfname + "\\" + subject + "_microstate_labels.json", 'w', encoding='utf-8'), separators=(',', ':'))
        # else:
        #     for a_task in task:
        #         template = load_data(maps_fname + "\\" +subject +"_microstate.json")[a_task]
        #         index = template['opt_k_index']
        #         template_map = template['maps_list'][index]
        #         Microstate.fit_back(zero_mean(np.asarray(data[a_task]).T, 1), np.asarray(template_map), instantaneous_eeg=is_instantaneous)


def batch_fit_back(para):
    data = para[0]
    template = para[1]
    is_instantaneous = para[2]
    return Microstate.fit_back(data, template, instantaneous_eeg=is_instantaneous).tolist()


# def fit_back(maps_fname, subjects, subjects_fname, subjects_sfname, task, is_global_maps):
#     for subject in subjects:
#         print(subject)
#         res = {}
#         data = load_data(subjects_fname + "\\" + subject + '.json')
#         data = combine_task_data(data, task)
#
#         if is_global_maps:
#             template = load_data(maps_fname)['maps']
#             for a_task in task:
#                 data_temp = zero_mean(np.asarray(data[a_task]).T, 1)
#                 res[a_task] = Microstate.fit_back(data_temp, np.asarray(template), instantaneous_eeg=False).tolist()
#         else:
#             template = load_data(maps_fname + "\\" + subject +"_1_30_microstate_reorder.json")
#             for a_task in task:
#                 data_temp = zero_mean(np.asarray(data[a_task]).T, 1)
#                 res[a_task] = Microstate.fit_back(data_temp, np.asarray(template[a_task]),instantaneous_eeg=True).tolist()
#         json.dump(res, codecs.open(subjects_sfname + "\\" + subject + "_microstate_labels.json", 'w',
#                                    encoding='utf-8'), separators=(',', ':'), sort_keys=True)



if __name__ == '__main__':
    # path = r'D:\EEGdata\clean_data_six_problem\new_1_30\individual_run\2-20\april_02(3)_1_30_microstate.json'
    # data = load_data(path)
    # task_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx','all')
    # for task in task_name:
    #     print(task)
    #     k = data[task]['opt_k_index']
    #     plot_eegmap_one_row(np.asarray(data[task]['maps_list'][3]))

    # subjects = read_subject_info(r'D:\EEGdata\clean_data_creativity\clean_subjects_28.xlsx', 'subjects')
    # path = r'D:\EEGdata\clean_data_creativity\1_30\1_30_clean_data'
    # task = read_subject_info(r'D:\EEGdata\clean_data_creativity\task_name.xlsx', 'task')
    # res = []
    # for subject in subjects:
    #     data = load_data(path + "\\" +subject +"_1_30.json")
    #     print(subject)
    #     temp = []
    #     for a_task in task:
    #         temp.append(np.asarray(data[a_task]).shape[1]/500)
    #     np_temp = np.asarray(temp)
    #     np_temp = (np_temp - np_temp.min()) / (np_temp.max()-np_temp.min())
    #     res.append(np_temp.tolist())
    # write_info(r'D:\EEGdata\clean_data_creativity\clean_subjects_28_behaviour.xlsx','norm_task_completion_time',res)

    # data = np.asarray(read_info(r'D:\EEGdata\clean_data_creativity\clean_subjects_28_behaviour.xlsx', 'task_completion_time'), dtype=np.float64)
    # res = np.zeros((28,3))
    # j=0
    # max_data = data.max(axis=0)
    # min_data = data.min(axis=0
    # data = (data - data.min()) / (data.max() - data.min())
    # for i in range(0,9,3):
    #     # mean_data = np.repeat(data[i:i+3].mean(axis=1), 9, axis=0).reshape(28,9)
    #     # std_data = np.repeat(data[i:i+3].std(axis=1), 9, axis=0).reshape(28,9)
    #     # res = (data[:,i:i+3] - np.mean(data[:,i:i+3]))/np.std(data[:,i:i+3])
    #     # res = (data[:,i:i+1] - np.min(data[:,i:i+1])) / (np.max(data[:,i:i+1]) - np.min(data[:,i:i+1]))
    #     # data[:,i:i+1] = res
    #     res[:,j] = data[:,i:i+3].mean(axis=1)
    #     j = j + 1

    # for j in range(0,3):
    #     res[:,j] = (res[:,j] - res[:,j].min())/(res[:,j].max() - res[:,j].min())
    # data = (data - min_data) / (max_data-min_data)

    # variation_ratio = stats.variation(data,axis=0)
    # variation_ratio = variation_ratio.reshape(1,9)
    # write_info(r'D:\EEGdata\clean_data_creativity\clean_subjects_28_behaviour.xlsx','task_all_max_min',res.tolist())
    # write_info(r'D:\EEGdata\clean_data_creativity\clean_subjects_28_behaviour.xlsx', 'variation_ratio', variation_ratio.tolist())

    # clean_data_fname = r'D:\EEGdata\clean_data_creativity\1_30\1_30_clean_data'

    parameters =[
                  # {'data_fname_save_dict':r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\labels\global_eegmaps_all_conditions',
                  #  'global_eegmaps_fname':r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\across_conditions\_1_30_microstate_across_runs_across_participants_across_conditions.json',
                  #  'is_global_maps': True,
                  #  'task_name':read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx','all')},
                  {
                      'clean_data_fname': r'D:\EEGdata\clean_data_six_problem\2_20\clean_data\downsample250',
                      'data_fname_save_dict': r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\k=7\labels_peaks\global_eegmaps_pu_ig_ie',
                      'global_eegmaps_fname': r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\k=7\across_conditions\_microstate_across_runs_across_participants_across_conditions_pu_ig_ie.json',
                      # 'global_eegmaps_fname': r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\k=7\individual_run',
                      'is_global_maps': True,
                      'task_name':read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx','pu_ig_ie')},
                  # {
                  #     'data_fname_save_dict': r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\labels\individual_eegmaps_all_conditions',
                  #     'global_eegmaps_fname': r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\individual_run_reordered\all_conditions',
                  #     'is_global_maps': False,
                  #     'task_name':read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx','all')},
                  # {
                  #     'data_fname_save_dict': r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\labels\individual_eegmaps_pu_ig_ie',
                  #     'global_eegmaps_fname': r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\individual_run_reordered\pu_ig_ie',
                  #     'is_global_maps': False,
                  #     'task_name':read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx','pu_ig_ie')
                  # },

                  ]
    #
    # clean_data_fname = r'D:\EEGdata\clean_data_six_problem\2_20\clean_data\downsample250'
    # data_fname_save_dict = r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\k=7\individual_run'
    # data_fname_save_dict = r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\across_runs'
    # clean_data_fname = r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\individual_run'
    # data_fname_save_dict = r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\across_participants'
    # clean_data_fname = r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\across_runs'
    # data_fname_save_dict = r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\across_conditions'
    # clean_data_fname = r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\across_participants'

    # subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_2_20')
    # task_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name.xlsx', 'pu_ig_ie')
    # for subject in subjects:
    #     print(subject)
    #     path = r'D:\EEGdata\clean_data_six_problem\2_20\clean_data'
    #     save_path = r'D:\EEGdata\clean_data_six_problem\2_20\clean_data\downsample125' + "\\" +subject +".json"
    #     data = load_data(path +"\\" +subject+".json")
    #     res = {}
    #     for task in task_name:
    #         print(task)
    #         secs = np.asarray(data[task]['task_data']).shape[1] / 500
    #         resample = signal.resample(data[task]['task_data'], int(secs*125), axis=1)
    #         res[task] = {'task_data':resample.tolist()}
    #     json.dump(res, codecs.open(save_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)

    # clean_data_fname = r'D:\EEGdata\clean_data_six_problem\1_30\microstate_across_runs_across_participants\k=6'
    # clean_data_fname = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\individual_run'

    # clean_data_fname = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\across_participants'
    # clean_data_fname = r'D:\EEGdata\clean_data_creativity\clean_subject_28\individual_parameters_new'
    # data_fname_save_dict = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\individual_run_reordered\pu_ig_ie'
    # data_fname_save_dict = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\labels\global_eegmaps_all_conditions'
    # data_fname_save_dict = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6'

    # global_eegmaps_fname = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\individual_run_reordered\all_conditions'
    # individual_eegmaps_fname = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\individual_run'
    # condition_eegmaps_fname = r'D:\EEGdata\clean_data_six_problem\1_30\microstate_across_runs_across_participants\k=7\_1_30_microstate_across_runs_across_subjects_k=7.json'
    # condition_eegmaps_fname = r'D:\EEGdata\clean_data_six_problem\1_30\microstate_across_runs_across_participants\k=7\_1_30_microstate_across_runs_across_subjects__k=7_reordered.json'


    # task_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name.xlsx','all')
    # condition_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx','pu_ig_ie')
    # task_name_temp = [[task_name[0]]]
    # for i in range(1, len(condition_name)):
    #     temp = []
    #     for index in range(1, len(task_name)):
    #         if condition_name[i] in task_name[index]:
    #             temp.append(task_name[index])
    #     task_name_temp.append(temp)
    # task_name = task_name_temp

    # subjects = read_subject_info(r'D:\EEGdata\clean_data_creativity\clean_subjects_28.xlsx')
    # subjects = read_subject_info(r'D:\EEGdata\clean_data_creativity\clean_subjects.xlsx')
    # condition_name = ['rest', 'idea generation', 'idea evolution', 'idea rating']

    # task_name = read_subject_info(r'D:\EEGdata\clean_data_creativity\clean_subject_28\task_name.xlsx')
    # task_name = [['1_rest'], ['1_idea generation', '2_idea generation', '3_idea generation'],
    #              ['1_idea evolution', '2_idea evolution', '3_idea evolution'],
    #              ['1_idea rating', '2_idea rating', '3_idea rating']]

    # individual microstates
    # eegmaps_individual_run(clean_data_fname, subjects, ".json", data_fname_save_dict, "_microstate.json", task_name, [1 for i in range(len(task_name))], [10 for i in range(len(task_name))])

    # individual microstates in terms of number
    # microstate_stat(data_fname_save_dict, subjects, "_microstate.json", "_microstate_stat.json", task_name)

    # write individual microstates statistic into a excel
    # data = load_data(r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\individual_run\_microstate_stat.json')
    # for a_condition_name in condition_name:
    #     res_data = []
    #     for a_task_name in task_name:
    #         if a_condition_name in a_task_name:
    #             res_data.append(data[a_task_name]['opt_k'])
    #     write_info(r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING_downsample250\parameters\microstate_num.xlsx', a_condition_name, res_data, False)

    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    conditions = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered')
    clean_data_fname = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_participants'
    data_fname_save_dict = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_conditions'

    # generate eegmaps across runs
    # eegmaps_across_runs(clean_data_fname, subjects, '_microstate.json', data_fname_save_dict, '_microstate_across_runs.json', tasks, conditions, 7, 6, 63)

    # generate eegmaps across runs across participants.
    # eegmpas_across_subjects(clean_data_fname, subjects, '_microstate_across_runs.json', data_fname_save_dict, '_microstate_across_runs_across_participants.json', conditions, 7, 63)

    # generate eegmaps across runs across participants across conditions
    # eegmaps_across_conditions(clean_data_fname, '_microstate_across_runs_across_participants.json', data_fname_save_dict, '_microstate_across_runs_across_participants_across_conditions.json', conditions, 7, 63)

    # for subject in subjects:
    #     print(subject)
    #     for task in tasks:
    #         task = 'rest_1'
    #         print(task)
    #         data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\individual_run'+ "\\" + subject+'_microstate.json')[task]
    #         cv = np.argmin(np.asarray(data['cv']))
    #         maps = np.asarray(data['maps'][3])
    #         plot_eegmap_one_row(maps)

    # plot global eegmaps
    # order = [0,1,2,3,4,5,6]
    # order = [6,0,2,5,3,4,1]
    # order = [2,3,0,1]
    # sign = [1,1,1,1,1,1,1]
    # data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_conditions\_microstate_across_runs_across_participants_across_conditions.json')["maps"]
    # plot_eegmap_one_row(np.asarray(data)[order],sign=sign)

    # reorder global eegmaps
    # data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_conditions\_microstate_across_runs_across_participants_across_conditions.json')["maps"]
    # # order = [2,3,0,1]
    # order = [3,0,2,5,6,4,1]
    # data = np.asarray(data)[order]
    # json.dump(data.tolist(), codecs.open(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_conditions\reorder_microstate_across_runs_across_participants_across_conditions.json', 'w', encoding='utf-8'), separators=(',', ':'),sort_keys=True)

    # plot reordered global eegmpas
    # data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_conditions\reorder_microstate_across_runs_across_participants_across_conditions.json')
    # # order = [4,5,6,0,3,2,1]
    # order = [0, 1, 2, 3, 4, 5, 6]
    # sign = [1,1,1,1,1,1,1]
    # order = [2, 1, 3, 4, 5, 0]
    # sign = [-1, -1, -1, 1, 1, -1]
    # order = [0,1,2,3]
    # sign = [1,1,-1,-1]
    # data = load_data(
    #     r'D:\EEGdata\clean_data_creativity\clean_subject_28\_1_30_microstate_across_runs_across_subjects_across_conditions_reordered.json')
    # order = [5, 1, 4, 3, 0, 2]
    # sign = [-1, 1, -1, 1, -1, -1]
    # savepath = r'D:\EEGdata\clean_data_creativity\clean_subject_28\reorder\global.eps'
    # plot_eegmap_one_row(np.asarray(data),order=order, sign=sign, title=['A','B','C','D','E','F','G'],savepath=None)

    # reoder condition eegmaps based on global eegmaps
    condition_maps = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_participants\_microstate_across_runs_across_participants.json')
    global_maps = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_conditions\reorder_microstate_across_runs_across_participants_across_conditions.json')
    maps = []
    res = {}
    for condition in conditions:
        maps.append(condition_maps[condition]["maps"])
    m = MeanMicrostate(maps, 7, 63, len(conditions))
    temp = m.reorder_microstates(np.asarray(maps), global_maps)
    for index, a_task_name in enumerate(conditions):
        res[a_task_name] = temp[index]
    json.dump(res, codecs.open(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_participants\reorder_microstate_across_runs_across_participants.json', 'w', encoding='utf-8'), separators=(',', ':'),sort_keys=True)

    #reorder individual condition eegmaps based on global eegmaps
    # global_maps = load_data(global_eegmaps_fname)['maps']
    # res_temp = []
    # res = {}
    # pool = Pool(len(condition_name))
    # for a_condition_name in condition_name:
    #     maps = []
    #     for subject in subjects:
    #         data = load_data(clean_data_fname + "\\" + subject + "_1_30_microstate_across_runs_k=6.json")[a_condition_name]["maps"]
    #         maps.append(data)
    #     res_temp.append(pool.apply_async(batch_order_mean_microstate, ([maps, 6, 63, len(subjects), global_maps],)))
    # pool.close()
    # pool.join()
    # for i, task_name in enumerate(condition_name):
    #     res[task_name] = res_temp[i].get()
    # json.dump(res, codecs.open(r'D:\EEGdata\clean_data_six_problem\1_30\microstate_across_runs\k=6\_1_30_microstate_across_runs_reordered.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    # global_maps = load_data(global_eegmaps_fname)['maps']
    # reorder_individual_eegmaps(global_maps, subjects, clean_data_fname, data_fname_save_dict, task_name, 6, 5)

    # plot condition eegmpas
    # condition_maps = load_data(r'D:\EEGdata\clean_data_creativity\clean_subject_28\_1_30_microstate_across_runs_across_subjects_reordered.json')
    # order = [5, 1, 4, 3, 0, 2]
    # sign = [-1, 1, -1, 1, -1, -1]
    # conditions = ['1_rest', 'idea generation', 'idea evolution', 'idea rating']
    # ylabel = ['Rest', 'Idea generation', 'Idea evolution', 'Evaluation']
    # savepath = r'D:\EEGdata\clean_data_creativity\clean_subject_28\reorder\condition.eps'
    # plot_eegmap_conditions(condition_maps, conditions, 6, sign=sign, order=order, title=['A','B','C','D','E','F'], ylabel=ylabel,savepath=savepath)

    # topographic parameter
    # data = load_data(r'D:\EEGdata\clean_data_creativity\clean_subject_28\_1_30_microstate_across_runs_reordered.json')
    # res = tanova(data, condition_name, len(subjects), len(condition_name), 6, 63, 5000)
    # res = permutation_test_3_conditions(data, condition_name, len(subjects), 5000)
    # res = permutation_test_2_conditions(data, condition_name, len(subjects), 5000)
    # print(res)

    #labels

    # for item in parameters:
    #     fit_back(item['global_eegmaps_fname'], subjects, item['clean_data_fname'], item['data_fname_save_dict'], item['task_name'], item['is_global_maps'], False)

    # fit_back(r'D:\EEGdata\clean_data_creativity\clean_subject_28\_1_30_microstate_across_runs_across_subjects_across_conditions_reordered.json', read_subject_info(r'D:\EEGdata\clean_data_creativity\clean_subjects_28.xlsx','subjects'),
    #          r'D:\EEGdata\clean_data_creativity\1_30\1_30_clean_data', r'D:\EEGdata\clean_data_creativity\clean_subject_28\labels_peaks',
    #          read_subject_info(r'D:\EEGdata\clean_data_creativity\task_name.xlsx','task'), True)

    # duration, occurrence, coverage
    # eegmaps_parameters(raw_clean_data_fname, subjects, "_1_30.json", global_eegmaps_fname, data_fname_save_dict, '_1_30_microstates_paraetmers', task_name)
    # eegmaps_parameters_across_runs(data_fname_save_dict, subjects, '_1_30_microstates_paraetmers', '_1_30_microstates_paraetmers_across_runs.json', condition_name, task_name)
    # para_name = ['duration', 'occurrence', 'coverage']
    # res = []
    # for subject in subjects:
    #     data = load_data(data_fname_save_dict + "\\" +subject+'_1_30_microstates_paraetmers_across_runs.json')
    #     temp = []
    #     for a_para_name in para_name:
    #         for a_task_name in condition_name:
    #             temp.extend(data[a_task_name][a_para_name])
    #     res.append(temp)
    # write_info(r'D:\EEGdata\clean_data_six_problem\1_30\statistical analysis\k=6\duration_occurrence_coverage.xlsx', 'across_runs', res, row_first=True)

    # paird-t test
    # 0-4: idea generation, 4-8: idea evolution, 8-12: evaluation
    # order = [5, 1, 4, 3, 0, 2]
    # order = [i for i in range]
    # res_temp = []
    # for comb in itertools.combinations([i for i in range(0, 4)], 2):
    #     print(comb)
    #     a = []
    #     b = []
    #     temp = []
    #     is_norm = []
    #     for item in res:
    #         a.append(item[comb[0]*6:(comb[0]+1)*6])
    #         b.append(item[comb[1] * 6:(comb[1] + 1) * 6])
    #     a = np.asarray(a)
    #     b = np.asarray(b)
    #     for i in range(6):
    #         temp.append(stats.ttest_rel(a[:,i],b[:,i])[1])
    #
    #     p_value = [temp[i] for i in order]
    #     # p_value = multipletests(temp, method='bonferroni')[1]
    #     res_temp.append(p_value)
    # write_info(r'D:\EEGdata\clean_data_creativity\clean_subject_28\1_30_duration_occurrence_coverage_paird_t_test.xlsx', 'duration_reordered', res_temp, row_first=True)


    # gev
    # gev_peaks = []
    # gev_raw = []
    # eegmaps = np.asarray(load_data(r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING\across_conditions\_microstate_across_runs_across_participants_across_conditions_pu_ig_ie.json')['maps'])
    # for subject in subjects:
    #     print(subject)
    #     data = load_data(r'D:\EEGdata\clean_data_six_problem\2_20\clean_data' + "\\" + subject +".json")
    #     data = combine_task_data(data, task_name)
    #     gev_peaks_list = []
    #     gev_raw_list = []
    #     for a_task_name in task_name:
    #         eegmaps_data = load_data(r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING\individual_run' + "\\" + subject + "_microstate.json")
    #         eegmaps = np.asarray(eegmaps_data[a_task_name]['maps_list'][6])
    #         microstate = Microstate(np.asarray(data[a_task_name]))
    #         temp_gev = Microstate.global_explained_variance_sum(microstate.data, eegmaps)
    #         gev_peaks_list.append(temp_gev[0])
    #         gev_raw_list.append(temp_gev[1])
    #
    #     gev_peaks.append(gev_peaks_list)
    #     gev_raw.append(gev_raw_list)
    # write_info(r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING\microstate_stat\microstate_gev.xlsx', 'gev_peaks', gev_peaks)
    # write_info(r'D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING\microstate_stat\microstate_gev.xlsx', 'gev_raw', gev_raw)