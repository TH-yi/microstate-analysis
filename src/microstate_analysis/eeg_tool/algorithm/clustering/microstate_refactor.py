import numpy as np
from multiprocessing import Pool
import codecs, json
from collections import OrderedDict
from scipy import signal
import os
import mne
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx
from scipy.io import loadmat, savemat
from scipy.stats import sem
import itertools

class Microstate:
    def __init__(self, data):
        self.data = Microstate.substract_mean(data)
        self.n_t = self.data.shape[0]
        self.n_ch = self.data.shape[1]
        self.gfp = np.std(self.data, axis=1)
        self.gfp_peaks = Microstate.locmax(self.gfp)
        self.gfp_values = self.gfp[self.gfp_peaks]
        self.n_gfp = self.gfp_peaks.shape[0]
        self.sum_gfp2 = np.sum(self.gfp_values**2)

    @staticmethod
    def locmax(x):
        dx = np.diff(x)
        zc = np.diff(np.sign(dx))
        m = 1 + np.where(zc == -2)[0]
        return m

    @staticmethod
    def substract_mean(x):
        return x - x.mean(axis=1, keepdims=True)

    @staticmethod
    def assign_labels_kmeans(data, maps, n_ch, gfp, gfp_peaks=None):
        c = np.dot(data, maps.T)
        if isinstance(gfp_peaks, np.ndarray):
            c /= (n_ch * np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))
        else:
            c /= (n_ch * np.outer(gfp, np.std(maps, axis=1)))
        l = np.argmax(c ** 2, axis=1)
        return l, c

    def fit_back(self, maps, threshold=None):
        c = np.dot(self.data, Microstate.substract_mean(maps).T) / (self.n_ch * np.outer(self.gfp, maps.std(axis=1)))
        c = abs(c)
        c_max = np.max(c,axis=1)
        c_max_index = np.argmax(c, axis=1)
        if threshold:
            c_threshold_index = np.where(c_max > threshold)[0]
            l = c_max_index[c_threshold_index]
        else:
            l = c_max_index
        return l

    def gev(self, maps):
        n_maps = len(maps)
        c = np.dot(self.data[self.gfp_peaks], maps.T)
        c /= (self.n_ch * np.outer(self.gfp[self.gfp_peaks], np.std(maps, axis=1)))
        l = np.argmax(c**2, axis=1)
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = l == k
            gev[k] = np.sum(self.gfp_values[r]**2 * c[r,k]**2)/self.sum_gfp2
        return gev, np.sum(gev)

    def kmeans(self, n_maps, maxerr=1e-6, maxiter=500):
        np.random.seed()
        rndi = np.random.permutation(self.n_gfp)[:n_maps]
        data_gfp = self.data[self.gfp_peaks,:]
        sum_v2 = np.sum(data_gfp**2)
        maps = data_gfp[rndi, :]
        maps /= np.sqrt(np.sum(maps ** 2, axis=1, keepdims=True))
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        while ((np.abs((var0 - var1) / var0) > maxerr) & (n_iter < maxiter)):
            l_peaks, c = Microstate.assign_labels_kmeans(data_gfp, maps, self.n_ch, self.gfp, self.gfp_peaks)
            for k in range(n_maps):
                vt = data_gfp[l_peaks==k, :]
                sk = np.dot(vt.T, vt)
                evals, evecs = np.linalg.eig(sk)
                v = np.real(evecs[:, np.argmax(np.abs(evals))])
                maps[k, :] = v / np.sqrt(np.sum(v ** 2))
            var1 = var0
            var0 = sum_v2 - np.sum(np.sum(maps[l_peaks, :] * data_gfp, axis=1) ** 2)
            var0 /= (self.n_gfp * (self.n_ch - 1))
            n_iter += 1
        l, _ = Microstate.assign_labels_kmeans(self.data, maps, self.n_ch, self.gfp)
        var = np.sum(self.data ** 2) - np.sum(np.sum(maps[l, :] * self.data, axis=1) ** 2)
        var /= (self.n_t * (self.n_ch - 1))
        cv = var * (self.n_ch - 1) ** 2 / (self.n_ch - n_maps - 1.) ** 2
        return maps, l, cv

    def wrap_kmeans(self, para):
        return self.kmeans(para[0])

    def kmeans_repetition(self, n_repetition, n_maps, n_pool=11):
        l_list = []
        cv_list = []
        maps_list = []
        pool = Pool(n_pool)
        multi_res = []
        for _ in range(n_repetition):
            multi_res.append(pool.apply_async(self.wrap_kmeans, ([n_maps],)))
        pool.close()
        pool.join()
        for i in range(n_repetition):
            temp = multi_res[i].get()
            maps_list.append(temp[0])
            l_list.append(temp[1])
            cv_list.append(temp[2])
        k_opt = np.argmin(cv_list)
        return maps_list[k_opt], cv_list[k_opt]

    def kmeans_max_maps(self, max_maps, n_repetition, n_pool=11):
        maps_list = []
        cv_list = []
        for n_maps in range(1, max_maps+1):
            temp = self.kmeans_repetition(n_repetition, n_maps, n_pool)
            maps_list.append(temp[0].tolist())
            cv_list.append(temp[1])
        return maps_list, cv_list


class MicrostatePlot:
    def __init__(self, maps=None):
        self.maps = maps
        self.nmaps = len(maps)
        self.info = MicrostatePlot.create_info()

    @staticmethod
    def create_info():
        ch = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1',
              'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7',
              'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz',
              'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
        info = mne.create_info(ch_names=ch, sfreq=500, ch_types='eeg',
                               montage=os.path.join(get_root_path(), 'Cap63.locs'))
        return info

    def plot_individual_map(self):
        pass


if __name__ == '__main__':
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    conditions = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered')
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    maps = np.asarray(load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_conditions\reorder_microstate_across_runs_across_participants_across_conditions.json'))
    # maps = np.asarray(load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_conditions\_microstate_across_runs_across_participants_across_conditions.json')['maps'])

    # res = []
    # for subject in subjects:
    #     print(subject)
    #     os.mkdir(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\sequences' + "\\" + subject)
    #     temp = []
    #     for task in tasks:
    #         print(task)
    #         data_path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_matlab\merged_epochs_data' + "\\" + subject + "\\" +task + '.mat'
    #         data_path_save = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\sequences' + "\\" + subject + "\\" +task + '_seq.mat'
    #         data = loadmat(data_path)['EEG']
    #         ms = Microstate(data.T)
    #         l = ms.fit_back(maps)
    #         gev, gev_total = ms.gev(maps)
    #         temp.append(gev_total)
    #         savemat(data_path_save, {'EEG':l})
    #     res.append(temp)
    # write_info(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\gev.xlsx', 'gev_peaks', res)

    res = read_xlsx(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\gev.xlsx', 'gev_peaks')
    res = np.asarray(res)
    rest = res[:, 0:2].flatten()
    print(round(np.mean(rest)*100,3), round(sem(rest)*100,3))
    for i in range(2, 32, 6):
        temp = res[:, i:i + 6].flatten()
        print(round(np.mean(temp)*100,3), round(sem(temp)*100,3))

    # temp = []
    # title = ['coverage', 'duration','frequency']
    # temp.append(loadmat(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=7\parameters\condition\condition_coverage.mat')['EEG'])
    # temp.append(loadmat(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=7\parameters\condition\condition_duration.mat')['EEG'])
    # temp.append(loadmat(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=7\parameters\condition\condition_frequency.mat')['EEG'])
    # for index, data in enumerate(temp):
    #     res = []
    #     for i in range(0, 189, 7):
    #         res.append(data[:,i:i+7].flatten().tolist())
    #     write_info(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=7\parameters\condition\condition.xlsx', title[index], res)

    # path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\individual_run'
    # res = []
    # for subject in subjects:
    #     data = load_data(path + "\\" + subject +"_microstate.json")
    #     temp = []
    #     for task in tasks:
    #         cv = data[task]['cv']
    #         index = np.argmin(cv)
    #         num = index + 1
    #         temp.append(num)
    #     res.append(temp)
    # write_info(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\stat\num.xlsx', 'num', res)
