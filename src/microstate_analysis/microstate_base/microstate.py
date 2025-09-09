
from scipy import signal
import numpy as np

from microstate_analysis.eeg_tool.math_utilis import zero_mean


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

        # All data from min_maps to max_maps
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
            gev[k] = np.sum(data_std[label == k] ** 2 * correlation[label == k, k] ** 2) / np.sum(
                data_std[label != -1] ** 2)
        return gev

    @staticmethod
    def global_explained_variance_sum(data, maps, distance=10, n_std=3, polarity=False):
        n_maps = maps.shape[0]
        n_ch = maps.shape[1]
        gev_peaks = np.zeros(n_maps)
        gev_raw = np.zeros(n_maps)
        gfp = data.std(axis=1)
        peaks, _ = signal.find_peaks(gfp, distance=distance,
                                     height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
        correlation_peaks = Microstate.spatial_correlation(data[peaks], zero_mean(maps, 1).T, gfp[peaks],
                                                           maps.std(axis=1), n_ch)
        correlation_raw = Microstate.spatial_correlation(data, zero_mean(maps, 1).T, gfp, maps.std(axis=1), n_ch)
        correlation_peaks = correlation_peaks if polarity else abs(correlation_peaks)
        correlation_raw = correlation_raw if polarity else abs(correlation_raw)
        label_peaks = np.argmax(correlation_peaks, axis=1)
        label_raw = np.argmax(correlation_raw, axis=1)
        for k in range(n_maps):
            gev_peaks[k] = np.sum(
                gfp[peaks][label_peaks == k] ** 2 * correlation_peaks[label_peaks == k, k] ** 2) / np.sum(
                gfp[peaks] ** 2)
            gev_raw[k] = np.sum(gfp[label_raw == k] ** 2 * correlation_raw[label_raw == k, k] ** 2) / np.sum(gfp ** 2)
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
    def fit_back(data, maps, distance=10, n_std=3, polarity=False):
        gfp = data.std(axis=1)
        peaks, _ = signal.find_peaks(gfp, distance=distance,
                                     height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
        label = np.full(data.shape[0], -1)
        correlation = Microstate.spatial_correlation(data[peaks], zero_mean(maps, 1).T, gfp[peaks], maps.std(axis=1),
                                                     data.shape[0])
        correlation = correlation if polarity else abs(correlation)
        label_peaks = np.argmax(correlation, axis=1)
        for i in range(len(peaks)):
            if i == 0:
                previous_middle = 0
                next_middle = int((peaks[i] + peaks[i + 1]) / 2)
            elif i == len(peaks) - 1:
                previous_middle = int((peaks[i] + peaks[i - 1]) / 2)
                next_middle = len(peaks) - 1
            else:
                previous_middle = int((peaks[i] + peaks[i - 1]) / 2)
                next_middle = int((peaks[i] + peaks[i + 1]) / 2)
            label[previous_middle:next_middle] = label_peaks[i]
        return label

    @staticmethod
    def microstates_duration(label, n_maps):
        duration = [[] for _ in range(n_maps)]
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
    def microstates_coverage(label, n_maps):
        coverage = []
        n_label = len(label)
        for i in range(n_maps):
            coverage.append(np.argwhere(label == i).shape[0] / n_label)
        return coverage

    @staticmethod
    def microstates_parameters(data, maps, distance=10, n_std=3, polarity=False, sfreq=500, epoch=2):
        n_maps = len(maps)
        res = {'duration': [], 'coverage': []}
        data = zero_mean(data.T, 1)
        for i in range(0, data.shape[0], sfreq * epoch):
            data_epoch = data[i: i + sfreq * epoch]
            label = Microstate.fit_back(data_epoch, maps, distance, n_std, polarity)
            res['duration'].append(Microstate.microstates_duration(label, n_maps))
            res['coverage'].append(Microstate.microstates_coverage(label, n_maps))
        return res

    def cross_validation(self, var, n_maps):
        return var * (self.n_ch - 1) ** 2 / (self.n_ch - n_maps - 1.) ** 2

    def variance(self, label, maps, data):
        var = np.sum(data ** 2) - np.sum(np.sum(maps[label, :] * data, axis=1) ** 2)
        var /= (self.n_t * (self.n_ch - 1))
        return var

    def gfp_peaks(self, distance=10, n_std=3):
        gfp = self.data.std(axis=1)
        peaks, _ = signal.find_peaks(gfp, distance=distance,
                                     height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
        self.peaks = peaks
        self.gfp = gfp

    def kmeans_modified(self, data, data_std, n_runs=10, n_maps=4, maxerr=1e-6, maxiter=1000, polarity=False):
        n_gfp = data_std.shape[0]
        cv_list = []
        gev_list = []
        maps_list = []
        label_list = []
        for run in range(n_runs):
            # if run == 0 or run == n_runs - 1:
            #     print(f"{run + 1}/{n_runs}")
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

    def opt_microstate(self, min_maps=2, max_maps=10, distance=10, n_std=3, n_runs=10, maxerr=1e-6, maxiter=1000,
                       polarity=False, peaks_only=True, method='kmeans_modified', opt_k=None):
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
            for n_maps in range(min_maps, temp_max_maps + 1):
                # if n_maps == min_maps or n_maps == max_maps or n_maps % 5 == 0:
                #     print(
                #         "kmeans_number:{number}/{maxi} will run {runs} times".format(number=n_maps, maxi=temp_max_maps,
                #                                                                      runs=n_runs))
                cv, gev, maps, label = self.kmeans_modified(data=temp_data, data_std=temp_data_std, n_runs=n_runs,
                                                            n_maps=n_maps, maxerr=maxerr, maxiter=maxiter,
                                                            polarity=polarity)
                self.cv_list.append(cv)
                self.gev_list.append(gev)
                self.maps_list.append(maps)
                self.label_list.append(label)

        elif method == 'aahc':
            self.cv_list, self.gev_list, self.maps_list, self.label_list = self.aahc(data=temp_data,
                                                                                     data_std=temp_data_std,
                                                                                     min_maps=min_maps,
                                                                                     max_maps=temp_max_maps,
                                                                                     polarity=polarity)

        if opt_k:
            self.opt_k_index = opt_k - min_maps
        else:
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
        # calc each time point original data and maps correlation
        correlation = Microstate.spatial_correlation(data, zero_mean(maps, 1).T, data_std, maps.std(axis=1), self.n_ch)
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
            #print("n_maps:%d" % n_maps)
            correlation = Microstate.spatial_correlation(data, zero_mean(maps, 1).T, data_std, maps.std(axis=1),
                                                         self.n_ch)
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
            maps = np.vstack((maps[:excluded_k, :], maps[excluded_k + 1:, :]))
            re_label = label_list.pop(excluded_k)
            re_cluster = []
            for k in re_label:
                correlation = Microstate.spatial_correlation(data[k, :], zero_mean(maps, 1).T, data_std[k],
                                                             maps.std(axis=1), self.n_ch)
                correlation = correlation if polarity else abs(correlation)
                new_label = np.argmax(correlation)
                re_cluster.append(new_label)
                label_list[new_label].append(k)

            for i in re_cluster:
                idx = label_list[i]
                maps[i] = Microstate.max_evec(data[idx, :], 0)
            n_maps = len(maps)

        return cv_list[::-1], gev_list[::-1], maps_list[::-1], res_label_list[::-1]
