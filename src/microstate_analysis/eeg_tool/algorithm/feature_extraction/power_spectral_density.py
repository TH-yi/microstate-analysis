import numpy as np
from multiprocessing import Pool
import codecs, json
from collections import OrderedDict
from scipy import signal
import os
import matplotlib.pyplot as plt

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

def batch_psd(data, tasks, fs, nperseg, noverlap, window='hamming', nfft=None, detrend=False, scaling='psd'):
    res = {}
    multi_res = []
    data = combine_task_data(data, tasks)
    pool = Pool(11)
    for task in tasks:
        multi_res.append(pool.apply_async(psd, ([np.asarray(data[task]), fs, window, nperseg, noverlap, nfft, detrend, scaling],)))
    pool.close()
    pool.join()
    for i in range(len(tasks)):
        temp = multi_res[i].get()
        res[tasks[i]] = {'f': temp[0].tolist(), 'pxx': temp[1].tolist()}
    return res

def psd(para):
    f, pxx = signal.welch(para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7])
    return f, pxx

if __name__ == '__main__':
    tasks = MicrostateUtilities.tasks()
    subjects = MicrostateUtilities.subjects()
    read_cwd = '/nfs/speed-scratch/w_ia/clean_data_1_40/clean_data'
    save_cwd_psd = '/nfs/speed-scratch/w_ia/psd'
    save_cwd_spectrum = '/nfs/speed-scratch/w_ia/spectrum'
    nperseg = 1000
    noverlap = 750
    fs = 500
    chs = 63
    for subject in subjects:
        print(subject)
        # read_path_json = os.path.join(read_cwd, subject + ".json")
        # save_psd_path_json = os.path.join(save_cwd_psd, subject + "_psd.json")
        # save_spectrum_path_json = os.path.join(save_cwd_spectrum, subject + "_spectrum.json")
        read_path_json = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data' + '\\' + subject + ".json"
        save_spectrum_path_json = r'D:\EEGdata\clean_data_six_problem\1_40\psd' + '\\' + subject +'_psd.json'

        data = MicrostateUtilities.load_data(read_path_json)
        # res = batch_psd(data, tasks, fs, nperseg, noverlap)
        # json.dump(res, codecs.open(save_psd_path_json, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        res = batch_psd(data, tasks, fs, nperseg, noverlap, scaling='density')
        json.dump(res, codecs.open(save_spectrum_path_json, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)

