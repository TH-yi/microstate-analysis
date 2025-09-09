import numpy as np
from itertools import combinations
import math
import codecs, json
from multiprocessing import Pool
import os

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)

def to_string(data):
    res = [str(i) for i in data]
    return '-'.join(res)


class MicrostateUtilities:
    @staticmethod
    def subjects():
        return ['april_02(3)', 'april_08', 'april_15', 'april_16(1)', 'april_16(3)', 'april_18(1)', 'april_18(2)', 'april_22', 'july_30', 'sep_12', 'sep_13', 'eeg_sep_18', 'Feb_18(1)_2014', 'Feb_19(2)_2014', 'Feb_20(2)_2014', 'Mar_12_2014', 'Mar_14(2)_2014', 'april_2(1)', 'april_19(1)', 'april_19(2)', 'april_24', 'Feb_07(1)_2014', 'Feb_18(2)_2014', 'Feb_28(1)_2014', 'Feb_28(2)_2014', 'april_30_2014', 'april_04(1)', 'sep_13(2)']

    @staticmethod
    def tasks():
        return ['1_rest', '1_read problem', '1_generate solution', '1_evaluate solution#1_type', '2_read problem', '2_generate solution', '2_evaluate solution#2_type', '3_read problem', '3_generate solution', '3_evaluate solution#3_type', '4_read problem', '4_generate solution', '4_evaluate solution#4_type', '5_read problem', '5_generate solution', '5_evaluate solution#5_type', '6_read problem', '6_generate solution', '6_evaluate solution#6_type']

    @staticmethod
    def load_data(path):
        data_text = codecs.open(path, 'r', encoding='utf-8').read()
        data = json.loads(data_text)
        return data

class MicrostateLongRangeDependence:
    def __init__(self, sequence, n_microstate):
        self.sequence = sequence
        self.n_microstate = n_microstate
        self.n_sequence = len(sequence)

    def partition_state(self, k):
        comb = combinations([i for i in range(self.n_microstate)], k)
        length = nCr(self.n_microstate, k)
        res = []
        for index, item in enumerate(comb):
            if k % 2 == 0:
                if index == length / 2:
                    break
                else:
                    res.append(item)
            else:
                res.append(item)
        return res

    def embed_random_walk(self, k):
        partitions = self.partition_state(k)
        np_sequence = np.asarray(self.sequence)
        res = {}
        for item in partitions:
            temp_x = np.ones(self.n_sequence) * -1
            for state in item:
                temp_x[np.where(np_sequence == state)[0]] = 1
            res[to_string(item)] = temp_x
        return res

    @staticmethod
    def detrend(embed_sequence, window_size):
        shape = (embed_sequence.shape[0]//window_size, window_size)
        temp = np.lib.stride_tricks.as_strided(embed_sequence, shape)
        window_size_index = np.arange(window_size)
        res = np.zeros(shape[0])
        for index, y in enumerate(temp):
            coeff = np.polyfit(window_size_index, y, 1)
            y_hat = np.polyval(coeff, window_size_index)
            res[index] = np.sqrt(np.mean((y-y_hat)**2))
        return res

    @staticmethod
    def dfa(embed_sequence, segment_range, segment_density):
        y = np.cumsum(embed_sequence-np.mean(embed_sequence))
        scales = np.arange(segment_range[0], segment_range[1], segment_density)
        f = np.zeros(len(scales))
        for index, window_size in enumerate(scales):
            f[index] = np.sqrt(np.mean(MicrostateLongRangeDependence.detrend(y, window_size)**2))
        coeff = np.polyfit(np.log(scales), np.log(f), 1)
        return {'slope': coeff[0], 'fluctuation': f.tolist(), 'scales':scales.tolist()}

def batch_calculate_dfa(data, n_microstate, n_partitions, segment_range, segment_density):
    res = []
    multi_res = []
    pool = Pool(35)
    lrd = MicrostateLongRangeDependence(data, n_microstate)
    partitions = lrd.partition_state(n_partitions)
    embeding = lrd.embed_random_walk(n_partitions)
    for embed_sequence_index, embed_sequence in embeding.items():
        multi_res.append(
            pool.apply_async(dfa, ([embed_sequence, embed_sequence_index, segment_range, segment_density],))
        )
    pool.close()
    pool.join()
    for i in range(len(multi_res)):
        res.append(multi_res[i].get())
    return res

def dfa(para):
    embed_sequence = para[0]
    embed_sequence_index = para[1]
    segment_range = para[2]
    segment_density = para[3]
    res = {}
    res[embed_sequence_index] = MicrostateLongRangeDependence.dfa(embed_sequence, segment_range, segment_density)
    return res


if __name__ == '__main__':
    read_cwd = '/nfs/speed-scratch/w_ia/data'
    save_cwd = '/nfs/speed-scratch/w_ia/hurst'
    subjects = MicrostateUtilities.subjects()
    tasks = MicrostateUtilities.tasks()
    n_repetition = 100
    n_microstates = 7
    n_partitions = 3
    res = {}
    for subject in subjects:
        res[subject] = {}
        read_path_json = os.path.join(read_cwd, subject + "_markov_1_repetition_100.json")
        save_path_json = os.path.join(save_cwd, subject + "_hurst_exponent.json")
        data = MicrostateUtilities.load_data(read_path_json)
        for task in tasks:
            res[subject][task] = []
            for repetition in range(n_repetition):
                segment_range = [50, len(data[task][repetition]['surrogate_data']) // 4]
                segment_density = 50
                res[subject][task].append(
                    batch_calculate_dfa(data[task][repetition]['surrogate_data'], n_microstates, n_partitions,
                                        segment_range,
                                        segment_density))
        json.dump(res, codecs.open(save_path_json, 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True)
