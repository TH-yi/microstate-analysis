import numpy as np
from scipy.stats import entropy
import codecs, json
from multiprocessing import Pool
import os

import datetime

import time

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


class MicrostateMarkov:
    def __init__(self, sequence, n_microstate):
        self.sequence = sequence
        self.n_microstate = n_microstate
        self.n_sequence = len(sequence)

    @staticmethod
    def surrogate_markov_chain(joint_distribution, condition_distribution, k, n_microstate, n_surrogate):
        print(datetime.datetime.now())
        sh = tuple(k * [n_microstate])
        d = {}
        dinv = np.zeros((n_microstate ** k, k))
        for i, idx in enumerate(np.ndindex(sh)):
            d[idx] = i
            dinv[i] = idx
        joint_distribution_sum = np.cumsum(joint_distribution)
        x = np.zeros(n_surrogate)
        x[:k] = dinv[np.min(np.argwhere(np.random.rand() < joint_distribution_sum))]
        condition_distribution_sum = np.cumsum(condition_distribution, axis=1)
        for t in range(n_surrogate - k - 1):
            idx = tuple(x[t:t + k])
            i = d[idx]
            j = np.min(np.argwhere(np.random.rand() < condition_distribution_sum[i]))
            x[t + k] = j
        return x.astype('int')

class MicrostateLongRangeDependence:
    def __init__(self, sequence, n_microstate):
        self.sequence = sequence
        self.n_microstate = n_microstate
        self.n_sequence = len(sequence)

    @staticmethod
    def shanon_entropy(x, nx, ns):
        p = np.zeros(ns)
        for t in range(nx):
            p[x[t]] += 1
        p /= nx
        return -np.sum(p[p>0]*np.log2(p[p>0]))

    @staticmethod
    def shanon_joint_entropy(x, y, nx, ny, ns):
        n = min(nx, ny)
        p = np.zeros((ns, ns))
        for t in range(n):
            p[x[t], y[t]] += 1
        p /= n
        return -np.sum(p[p>0]*np.log2(p[p>0]))

    def mutual_information(self, lag):
        lag = min(self.n_sequence, lag)
        res = np.zeros(lag)
        for time_lag in range(lag):
            nmax = self.n_sequence - time_lag
            h = self.shanon_entropy(self.sequence[:nmax], nmax, self.n_microstate)
            h_lag = self.shanon_entropy(self.sequence[time_lag:time_lag+nmax], nmax, self.n_microstate)
            h_h_lag = self.shanon_joint_entropy(self.sequence[:nmax], self.sequence[time_lag:time_lag+nmax], nmax,
                                                nmax, self.n_microstate)
            res[time_lag] = h + h_lag - h_h_lag
        return res


def batch_surrogate_aif(joint_p, condition_p, n_markov, n_microstate, n_surrogate, n_lag, n_repetiton):
    res = []
    multi_res = []
    pool = Pool(32)
    for _ in range(n_repetiton):
        multi_res.append(
            pool.apply_async(surrogate_aif, ([joint_p, condition_p, n_markov, n_microstate, n_surrogate, n_lag],))
        )

    pool.close()
    pool.join()
    for i in range(len(multi_res)):
        temp = multi_res[i].get()
        res.append({'surrogate_data':temp[0].tolist(), 'mutual_info':temp[1].tolist()})
    return res

def surrogate_aif(para):
    j_p = para[0]
    c_p = para[1]
    n_markov = para[2]
    n_microstate = para[3]
    n_surrogate = para[4]
    n_lag = para[5]
    surrogate_data = MicrostateMarkov.surrogate_markov_chain(j_p, c_p, n_markov, n_microstate, n_surrogate)
    lrd = MicrostateLongRangeDependence(surrogate_data, n_microstate)
    return surrogate_data, lrd.mutual_information(n_lag)


if __name__ == '__main__':
    cwd = '/nfs/speed-scratch/w_ia'
    save_cwd = '/nfs/speed-scratch/w_ia/data'
    subjects = MicrostateUtilities.subjects()
    tasks = MicrostateUtilities.tasks()
    matrix = MicrostateUtilities.load_data(os.path.join(cwd, 'matrix_test.json'))
    tasks_length = MicrostateUtilities.load_data(os.path.join(cwd, 'tasks_length.json'))
    block_repetition = 100
    n_repetition = 1
    n_lag = 2000
    n_microstate = 6
    repetition = [block_repetition for _ in range(n_repetition)]
    for n_markov in range(1, 6):
        for subject in subjects:
            for i in range(n_repetition):
                path_temp = subject + "_markov_" + str(n_markov) + "_repetition_" + str(block_repetition *(i+1)) + ".json"
                path = os.path.join(save_cwd, path_temp)
                res = {}
                for task in tasks:
                    n_surrogate = tasks_length[subject][task]
                    condition_p = matrix[subject][task][str(n_markov)]['condition_p']
                    joint_p = matrix[subject][task][str(n_markov)]['joint_p']
                    res[task] = batch_surrogate_aif(joint_p, condition_p, n_markov, n_microstate, n_surrogate, n_lag, block_repetition)
                json.dump(res, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
