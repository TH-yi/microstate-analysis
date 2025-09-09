import numpy as np
from multiprocessing import Pool
import codecs, json
from collections import OrderedDict
from scipy import signal
import os
from scipy.signal import find_peaks
from eeg_tool.utilis import read_subject_info
from scipy.io  import  loadmat, savemat
import mne

class MicrostateUtilities:
    @staticmethod
    def subjects():
        return ['april_02(3)', 'april_08', 'april_15', 'april_16(1)', 'april_16(3)', 'april_18(1)', 'april_18(2)', 'april_22', 'july_30', 'sep_12', 'sep_13', 'eeg_sep_18', 'Feb_18(1)_2014', 'Feb_19(2)_2014', 'Feb_20(2)_2014', 'Mar_12_2014', 'Mar_14(2)_2014', 'april_2(1)', 'april_19(1)', 'april_19(2)', 'april_24', 'Feb_07(1)_2014', 'Feb_18(2)_2014', 'Feb_28(1)_2014', 'Feb_28(2)_2014', 'april_30_2014', 'april_04(1)']

    @staticmethod
    def tasks():
        return ['1_rest', '1_read problem', '1_generate solution', '1_rate generation', '1_evaluate solution#1_type', '1_rate evaluation', '2_read problem', '2_generate solution', '2_rate generation', '2_evaluate solution#2_type', '2_rate evaluation', '3_read problem', '3_generate solution', '3_rate generation', '3_evaluate solution#3_type', '3_rate evaluation', '4_read problem', '4_generate solution', '4_rate generation', '4_evaluate solution#4_type', '4_rate evaluation', '5_read problem', '5_generate solution', '5_rate generation', '5_evaluate solution#5_type', '5_rate evaluation', '6_read problem', '6_generate solution', '6_rate generation', '6_evaluate solution#6_type', '6_rate evaluation']

    @staticmethod
    def load_data(path):
        data_text = codecs.open(path, 'r', encoding='utf-8').read()
        data = json.loads(data_text)
        return data

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
    def locmax(x, distance=10, n_std=3):
        dx = np.diff(x)
        zc = np.diff(np.sign(dx))
        peaks = 1 + np.where(zc == -2)[0]
        # peaks, _ = find_peaks(x, distance=distance, height=(x.mean() - n_std * x.std(), x.mean() + n_std * x.std()))
        return peaks

    @staticmethod
    def substract_mean(x):
        return x - x.mean(axis=1, keepdims=True)

    @staticmethod
    def assign_labels(data, maps, n_ch, gfp, gfp_peaks=None):
        c = np.dot(data, maps.T)
        if isinstance(gfp_peaks, np.ndarray):
            c /= (n_ch * np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))
        else:
            c /= (n_ch * np.outer(gfp, np.std(maps, axis=1)))
        l = np.argmax(c ** 2, axis=1)
        return l, c

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
            l_peaks, c = Microstate.assign_labels(data_gfp, maps, self.n_ch, self.gfp, self.gfp_peaks)
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
        l, _ = Microstate.assign_labels(self.data, maps, self.n_ch, self.gfp)
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


if __name__ == '__main__':
    # tasks = MicrostateUtilities.tasks()
    # subjects = MicrostateUtilities.subjects()
    # read_cwd = '/nfs/speed-scratch/w_ia/clean_data_1_40'
    # save_cwd = '/nfs/speed-scratch/w_ia/microstate'
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    n_maps = 10
    n_repetition = 100
    n_pool = 11
    for subject in subjects:
        print(subject)
        res = {}
        save_path_json = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\individual_run' + "\\" + subject + "_microstate.json"
        # read_path_json = os.path.join(read_cwd, subject + ".json")
        # save_path_json = os.path.join(save_cwd, subject + "_microstate.json")
        # data = MicrostateUtilities.load_data(read_path_json)
        # data = loadmat(read_path_json)['EEG']
        for task in tasks:
            print(task)
            read_path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_matlab\merged_epochs_data' + "\\" + subject + "\\"+ task + '.mat'
            data = loadmat(read_path)['EEG']
            microstate = Microstate(data.T)
            temp = microstate.kmeans_max_maps(n_maps, n_repetition, n_pool)
            res[task] = {'maps': temp[0],'cv': temp[1]}
        json.dump(res, codecs.open(save_path_json, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)