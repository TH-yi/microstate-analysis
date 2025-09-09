import numpy as np
import copy
from scipy.stats import chi2, chi2_contingency, entropy, t, sem, norm
from scipy import signal
from eeg_tool.utilis import load_data, read_subject_info, write_info, read_xlsx, to_string
from multiprocessing import Pool
import codecs, json
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import combinations
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import normalize
import string
import matplotlib.gridspec as gridspec
import math
from scipy import stats
from statsmodels.stats.multitest import multipletests
from eeg_tool.math_utilis import nCr
import mne
from eeg_tool.utilis import create_info
from scipy.io import loadmat, savemat
import os
from collections import OrderedDict

class MicrostateParameter:
    def __init__(self, sequence, n_microstate):
        self.sequence = sequence
        self.n_microstate = n_microstate
        self.n_sequence = len(sequence)

    def calculate_duration(self, window=None):
        def duration(sequence):
            res = []
            label = sequence[0]
            count = 1
            res_temp = {}
            for i in range(self.n_microstate):
                res_temp[i] = []
            for j in range(1, len(sequence)):
                if label == sequence[j]:
                    count += 1
                else:
                    label = sequence[j]
                    res_temp[sequence[j - 1]].append(count)
                    count = 1
            for i in range(self.n_microstate):
                res.append(np.sum(res_temp[i]) / len(res_temp[i]) if len(res_temp[i])>0 else 1)
            return res

        if window:
            n = int(self.n_sequence / window)
            temp = []
            for i in range(n):
                res = duration(self.sequence[i*window:(i+1)*window])
                temp.append(res)
            return np.asarray(temp).mean(axis=0).tolist()
        else:
            return duration(self.sequence)


    def calculate_frequency(self, window):
        res = []
        res_temp = {}
        for i in range(self.n_microstate):
            res_temp[i] = []
        n_block = int(self.n_sequence / window)
        for i in range(n_block):
            label = self.sequence[i*window]
            temp = {}
            for j in range(i*window + 1, (i+1)*window):
                if label != self.sequence[j]:
                    if label in temp:
                        temp[label] += 1
                    else:
                        temp[label] = 1
                    label = self.sequence[j]
            for key, value in temp.items():
                res_temp[key].append(value)

        for i in range(self.n_microstate):
            res.append(np.mean(res_temp[i]))
        return res

    def calculate_coverage(self):
        res = []
        for i in range(self.n_microstate):
            res.append(np.argwhere(np.asarray(self.sequence) == i).shape[0] / self.n_sequence)
        return res

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
        scales = (2**np.arange(segment_range[0], segment_range[1], segment_density)).astype(np.int)
        f = np.zeros(len(scales))
        for index, window_size in enumerate(scales):
            f[index] = np.sqrt(np.mean(MicrostateLongRangeDependence.detrend(y, window_size)**2))
        coeff = np.polyfit(np.log2(scales), np.log2(f), 1)
        return {'slope': coeff[0], 'fluctuation': f.tolist(), 'scales':scales.tolist()}

    @staticmethod
    def shanon_entropy(x, nx, ns):
        p = np.zeros(ns)
        for t in range(nx):
            p[x[t]] += 1.0
        p /= nx
        return -np.sum(p[p>0]*np.log2(p[p>0]))

    @staticmethod
    def shanon_joint_entropy(x, y, nx, ny, ns):
        n = min(nx, ny)
        p = np.zeros((ns, ns))
        for t in range(n):
            p[x[t], y[t]] += 1.0
        p /= n
        return -np.sum(p[p>0]*np.log2(p[p>0]))

    @staticmethod
    def shanon_joint_entropy_k(x, nx, ns, k):
        p = np.zeros(tuple(k*[ns]))
        for t in range(nx-k):
            p[tuple(x[t:t+k])] += 1.0
        p /= (nx-k)
        h = -np.sum(p[p>0]*np.log2(p[p>0]))
        return h

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


    def partial_mutual_information(self, lag):
        p = np.zeros(lag)
        a = self.mutual_information(2)
        p[0], p[1] = a[0], a[1]
        for k in range(2, lag):
            h1 = MicrostateLongRangeDependence.shanon_joint_entropy_k(self.sequence,self.n_sequence,self.n_microstate,lag)
            h2 = MicrostateLongRangeDependence.shanon_joint_entropy_k(self.sequence,self.n_sequence,self.n_microstate,lag-1)
            h3 = MicrostateLongRangeDependence.shanon_joint_entropy_k(self.sequence, self.n_sequence, self.n_microstate,
                                                                      lag + 1)
            p[k] = 2*h1 -h2 - h3
        return p

    def excess_entropy_rate(self, kmax):
        h = np.zeros(kmax)
        for k in range(kmax):
            h[k] = MicrostateLongRangeDependence.shanon_joint_entropy_k(self.sequence, self.n_sequence, self.n_microstate, k+1)
        ks = np.arange(1, kmax+1)
        entropy_rate, excess_entropy = np.polyfit(ks, h, 1)
        return entropy_rate, excess_entropy, ks

    @staticmethod
    def plot_mutual_information(data, title, subject, index, f):
        plt.semilogy(np.arange(0, data.shape[0]*f, f), data)
        plt.xlabel("time lag (1ms)")
        plt.ylabel("AIF (bits)")
        plt.title(subject+"_"+title+"_"+str(index))
        plt.show()


    # @staticmethod
    # def plot_mutual_information(data, title):
    #     for i in range(len(data)):
    #         temp = np.asarray(data[i])
    #         plt.semilogy([j for j in range(2000)], temp[0:2000]/temp[0])
    #         # plt.semilogy([j for j in range(len(data[i][0:2000]))], data[i][0:2000])
    #     plt.xlabel("time lag (2ms)")
    #     plt.ylabel("AIF (bits)")
    #     plt.title(title)
    #     plt.show()

class MicrostateMarkov:
    def __init__(self, sequence, n_microstate):
        self.sequence = sequence
        self.n_microstate = n_microstate
        self.n_sequence = len(sequence)

    def transition_matrix_hat(self, k):
        joint_distribution = np.zeros(self.n_microstate**k)
        condition_distribution = np.zeros((self.n_microstate**k, self.n_microstate))
        sh = tuple(k*[self.n_microstate])
        d = {}
        for i, index in enumerate(np.ndindex(sh)):
            d[index] = i
        for t in range(self.n_sequence-k):
            idx = tuple(self.sequence[t:t+k])
            i = d[idx]
            j = self.sequence[t+k]
            joint_distribution[i] += 1.
            condition_distribution[i,j] += 1.
        joint_distribution /= joint_distribution.sum()
        p_row = condition_distribution.sum(axis=1, keepdims=True)
        p_row[p_row == 0] = 1.
        condition_distribution /= p_row
        return joint_distribution, condition_distribution


    @staticmethod
    def surrogate_markov_chain(joint_distribution, condition_distribution, k, n_microstate, n_surrogate):
        sh = tuple(k * [n_microstate])
        d = {}
        dinv = np.zeros((n_microstate**k, k))
        for i, idx in enumerate(np.ndindex(sh)):
            d[idx] = i
            dinv[i] = idx
        joint_distribution_sum = np.cumsum(joint_distribution)
        x = np.zeros(n_surrogate)
        x[:k] = dinv[np.min(np.argwhere(np.random.rand() < joint_distribution_sum))]
        condition_distribution_sum = np.cumsum(condition_distribution, axis=1)
        for t in range(n_surrogate-k-1):
            idx = tuple(x[t:t+k])
            i = d[idx]
            j = np.min(np.argwhere(np.random.rand() < condition_distribution_sum[i]))
            x[t+k] = j
        return x.astype('int')



    def test_markov(self, order):
        if order == 0:
            return self.test_markov0()

        n = len(self.sequence)
        df = np.power(self.n_microstate, 2 + order) - 2 * np.power(self.n_microstate, 1 + order) + np.power(
            self.n_microstate, order)
        temp = np.zeros(self.n_microstate)
        frequency = []
        # f_ijk..(n+2), f_ij..(n+1)*, f_*jk..(n+2), f_*jk...(n+1)*
        frequency.append(np.tile(temp, [self.n_microstate for i in range(order + 1)] + [1]))
        frequency.append(np.tile(temp, [self.n_microstate for i in range(order)] + [1]))
        frequency.append(np.tile(temp, [self.n_microstate for i in range(order)] + [1]))
        frequency.append(np.tile(temp, [self.n_microstate for i in range(order - 1)] + [1]))
        for t in range(n - order - 1):
            frequency_index = []
            for j in range(order + 2):
                frequency_index.append(self.sequence[t + j])
            frequency[0][tuple(frequency_index)] += 1
            frequency[1][tuple(frequency_index[:-1])] += 1
            frequency[2][tuple(frequency_index[1::])] += 1
            frequency[3][tuple(frequency_index[1:-1])] += 1
        T = 0.0
        for frequency_index in np.ndindex(frequency[0].shape):
            f = frequency[0][tuple(frequency_index)] * frequency[1][tuple(frequency_index[:-1])] * \
                frequency[2][tuple(frequency_index[1::])] * frequency[3][tuple(frequency_index[1:-1])]
            if f > 0:
                T += frequency[0][tuple(frequency_index)] * \
                     np.log((frequency[0][tuple(frequency_index)] * frequency[3][tuple(frequency_index[1:-1])])
                            / (frequency[1][tuple(frequency_index[:-1])] * frequency[2][tuple(frequency_index[1::])]))
        T *= 2.0
        p = chi2.sf(T, df)
        # print(T, df, p)
        return p

    def test_markov0(self):
        n = len(self.sequence)
        f_ij = np.zeros((self.n_microstate, self.n_microstate))
        f_i = np.zeros(self.n_microstate)
        f_j = np.zeros(self.n_microstate)
        for t in range(n - 1):
            i = self.sequence[t]
            j = self.sequence[t + 1]
            f_ij[i, j] += 1.0
            f_i[i] += 1.0
            f_j[i] += 1.0
        T = 0.0
        for i, j in np.ndindex(f_ij.shape):
            f = f_ij[i, j] * f_i[i] * f_j[j]
            if f > 0:
                T += (f_ij[i, j] * np.log((n * f_i[i] * f_j[j]) / f_i[i] * f_j[j]))
        T *= 2.0
        df = (self.n_microstate - 1) * (self.n_microstate - 1)
        p = chi2.sf(T, df)
        return p

    def test_conditional_homogeneity(self, block_size, order, s=None):
        if s is None:
            n = len(self.sequence)
            s = int(np.floor(float(n) / float(block_size)))
        df = (s - 1.) * (np.power(self.n_microstate, order + 1) - np.power(self.n_microstate, order))
        frequency = []
        # f_ijk..(n+2), f_ij..(n+1)*, f_*jk..(n+2), f_*jk...(n+1)*
        frequency.append(np.zeros(([s] + [self.n_microstate for i in range(order + 1)])))
        frequency.append(np.zeros(([s] + [self.n_microstate for i in range(order)])))
        frequency.append(np.zeros(([self.n_microstate for i in range(order + 1)])))
        frequency.append(np.zeros(([self.n_microstate for i in range(order)])))
        l_total = 0
        for i in range(s):
            if isinstance(block_size, list):
                l = block_size[i]
            else:
                l = block_size
            for j in range(l - order):
                frequency_index = []
                frequency_index.append(i)
                for k in range(order + 1):
                    frequency_index.append(self.sequence[l_total + j + k])
                frequency[0][tuple(frequency_index)] += 1.
                frequency[1][tuple(frequency_index[:-1])] += 1.
                frequency[2][tuple(frequency_index[1::])] += 1.
                frequency[3][tuple(frequency_index[1:-1])] += 1.
            l_total += l

        T = 0.0
        for frequency_index in np.ndindex(frequency[0].shape):
            f = frequency[0][tuple(frequency_index)] * frequency[1][tuple(frequency_index[:-1])] * \
                frequency[2][tuple(frequency_index[1::])] * frequency[3][tuple(frequency_index[1:-1])]
            if f > 0:
                T += frequency[0][tuple(frequency_index)] * \
                     np.log((frequency[0][tuple(frequency_index)] * frequency[3][tuple(frequency_index[1:-1])])
                            / (frequency[1][tuple(frequency_index[:-1])] * frequency[2][tuple(frequency_index[1::])]))
        T *= 2.0
        p = chi2.sf(T, df, loc=0, scale=1.)
        return p

    def conditionalHomogeneityTest(self, l):
        n = len(self.sequence)
        ns = self.n_microstate
        X = self.sequence
        r = int(np.floor(float(n) / float(l)))  # number of blocks
        nl = r * l
        f_ijk = np.zeros((r, ns, ns))
        f_ij = np.zeros((r, ns))
        f_jk = np.zeros((ns, ns))
        f_i = np.zeros(r)
        f_j = np.zeros(ns)

        # calculate f_ijk (time / block dep. transition matrix)
        for i in range(r):  # block index
            for ii in range(l - 1):  # pos. inside the current block
                j = X[i * l + ii]
                k = X[i * l + ii + 1]
                f_ijk[i, j, k] += 1.0
                f_ij[i, j] += 1.0
                f_jk[j, k] += 1.0
                f_i[i] += 1.0
                f_j[j] += 1.0
        # conditional homogeneity (Markovianity stationarity)
        T = 0.0
        for i, j, k in np.ndindex(f_ijk.shape):
            # conditional homogeneity
            f = f_ijk[i, j, k] * f_j[j] * f_ij[i, j] * f_jk[j, k]
            if (f > 0):
                T += (f_ijk[i, j, k] * np.log((f_ijk[i, j, k] * f_j[j]) / (f_ij[i, j] * f_jk[j, k])))
        T *= 2.0
        df = (r - 1) * (ns - 1) * ns
        # p = chi2test(T, df, alpha)
        p = chi2.sf(T, df, loc=0, scale=1)
        return p


def paired_t_test_condition_parameter(path, savepath, sheets, n_microstates, conditions):
    n_conditions = len(conditions)
    alphabet_string = list(string.ascii_uppercase)
    for sheet in sheets:
        res = []
        res_correct = []
        data = np.asarray(read_xlsx(path, sheet))
        for i in range(n_microstates):
            data_temp = data[:,i*n_conditions:(i+1)*n_conditions]
            correct_p = []
            count = 0
            for comb in itertools.combinations([j for j in range(n_conditions)], 2):
                p = stats.ttest_rel(data_temp[:, comb[0]], data_temp[:, comb[1]])
                res.append([alphabet_string[i], conditions[comb[0]], conditions[comb[1]], p[0], p[1]])
                res_correct.append([alphabet_string[i], conditions[comb[0]], conditions[comb[1]]])
                correct_p.append(p[1])
                count += 1
            temp = multipletests(correct_p, method='bonferroni')[1].tolist()
            for j in range(len(res)-1, len(res)-count-1, -1):
                res_correct[j].append(temp[count-1])
                count -= 1

        write_info(savepath, sheet, res)
        write_info(savepath, sheet + '_' + 'bonferroni', res_correct)

def paired_t_test_run_parameter(path, savepath, sheets, n_microstates, n_conditions, n_runs):
    for sheet in sheets:
        res = []
        res_correct = []
        data = np.asarray(read_xlsx(path, sheet))[:, n_microstates::]
        for i in range(0, n_microstates*n_conditions*n_runs, n_runs):
            data_temp = data[:, i*n_runs:(i+1)*n_runs]
            correct_p = []
            for comb in itertools.combinations([j for j in range(n_runs)], 2):
                p = stats.ttest_rel(data_temp[:, comb[0]], data_temp[:, comb[1]])
                res.append(p)
                correct_p.append(p)
            # res_correct.extend(multipletests(correct_p, method='bonferroni')[1])
        write_info(savepath, sheet, res)
        # write_info(savepath, sheet+'_'+'bonferroni', res_correct)

def plot_run_parameter(path, sheets, row, n_microstates, n_runs, conditions):
    alphabet_string = list(string.ascii_uppercase)
    column = int(math.ceil(n_microstates / row))
    n_conditions = len(conditions)
    title = ['Microstate' + alphabet_string[i] for i in range(n_microstates)]
    block_size = n_microstates * n_runs
    for sheet in sheets:
        data = np.asarray(read_xlsx(path, sheet))[n_microstates::]
        for n_condition in range(n_conditions):
            fig, ax = plt.subplot(row, column)
            count = 0
            for i in range(row):
                for j in range(column):
                    ax[i][j].errorbar(data[count:count+n_runs], np.mean(data[count:count+n_runs], axis=0),
                                      yerr=stats.sem(data, axis=0))
                    count += n_runs

def label_diff(i, j, yoffset, text, y, ax):
    y = max(y[i], y[j])+ yoffset
    props = {'arrowstyle': '-', 'linewidth': 2}
    ax.annotate(text, xy=(i, y+0.003), zorder=1)
    ax.annotate('', xy=(i, y), xytext=(j, y), arrowprops=props)

def plot_block(mean, yerr, conditions, ax, title, ylabe, p_value=None, p_bar=None):
    n_conditions = len(conditions)
    if p_value is None:
        for index, value in enumerate(conditions):
            ax.errorbar(index, mean[index], yerr[index], fmt='.', marker='.', color='black', capsize=5, capthick=2)
    else:
        for index, value in enumerate(conditions):
            if index == 0:
                ax.errorbar(index, mean[index], yerr[index], fmt='.', marker='.', color='black', capsize=5, capthick=2)
            else:
                p = float(p_value[index-1])
                if p > 0.05:
                    ax.errorbar(index, mean[index], yerr=yerr[index], fmt='.', marker='.', color='black', capsize=5, capthick=2)
                if 0.01 <= p <= 0.05:
                    ax.errorbar(index, mean[index], yerr=yerr[index], fmt='.', marker='*', mfc='red', mec='red', color='black', capsize=5, capthick=2)
                if 0.005 <= p < 0.01:
                    ax.errorbar(index, mean[index], yerr=yerr[index], fmt='.', marker='*', mfc='green', mec='green', color='black', capsize=5, capthick=2)
                if p < 0.005:
                    ax.errorbar(index, mean[index], yerr=yerr[index], fmt='.', marker='*', mfc='blue', mec='blue', color='black', capsize=5, capthick=2)
        for i in range(len(conditions)-1, len(p_value)):
            p = float(p_value[i])
            if p < 0.05:
                p_str = "p=" + format(p, '.3f')
                label_diff(p_bar[i][0], p_bar[i][1], p_bar[i][2], p_str, mean+yerr, ax)

    ax.set_title(title)
    ax.set_xticks(list(range(n_conditions)))
    ax.set_xticklabels(conditions)
    ax.set_ylabel(ylabe)
    ax.set_ylim(ymax=max(mean+yerr) + 0.01)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

def plot_condition_paramter(path, savepath, sheets, row, n_microstates, conditions, ylabe, path_t_test=None, p_value_bar=None):
    alphabet_string = list(string.ascii_uppercase)
    column = int(math.ceil(n_microstates/row))
    n_conditions = len(conditions)
    n_cr = int(nCr(n_conditions,2))
    row_num = column * n_conditions
    n_conditions = len(conditions)
    title = ['Microstate'+ alphabet_string[i] for i in range(n_microstates)]
    for sheet in sheets:
        data = np.asarray(read_xlsx(path, sheet))
        p_value = np.asarray(read_xlsx(path_t_test, sheet))[:,-1] if path_t_test else None
        fig, ax = plt.subplots(row, column, figsize=(14,8))
        for i in range(row):
            for j in range(column):
                if column*i+j == n_microstates:
                    ax[i][j].axis('off')
                    break
                plot_block(data[:, row_num*i+j*n_conditions:row_num*i+(j+1)*n_conditions], conditions, ax[i][j], title[column*i+j], ylabe[sheet], p_value[n_cr*(column*i+j):n_cr*(column*i+j+1)], p_value_bar)
        plt.subplots_adjust(hspace=0.4, wspace=0.5)
        plt.savefig(savepath+"//"+sheet+".png", dpi=1200)
        plt.show()

def reorder_run_parameter(path, savepath, sheets, n_microstates, n_runs, n_conditions, order):
    for sheet in sheets:
        data = np.asarray(read_xlsx(path, sheet))
        data_ordered = np.zeros((data.shape[0], data.shape[1]))
        data_ordered[:, 0:n_microstates] = data[:, np.asarray(order)]
        block_size = n_microstates * n_runs
        for i in range(n_conditions):
            index = []
            for j in order:
                index.extend(list(range(j*n_runs+n_microstates+i*block_size,(j+1)*n_runs+n_microstates+i*block_size)))
            data_ordered[:,i*block_size+n_microstates:(i+1)*block_size+n_microstates] = data[:,index]

        write_info(savepath, sheet, data_ordered)


def reorder_condition_parameter(path, savepath, sheets, n_conditions, order):
    for sheet in sheets:
        data = np.asarray(read_xlsx(path, sheet))
        data_ordered = np.zeros((data.shape[0], data.shape[1]))
        for i, i_value in enumerate(order):
            data_ordered[:, i * n_conditions:(i + 1) * n_conditions] = data[:, i_value * n_conditions:(i_value + 1) * n_conditions]
        write_info(savepath, sheet, data_ordered.tolist())

def formulate_run_parameter(path, save_path, sheets, n_microstates, n_conditions, n_runs):
    block_size = n_microstates * n_runs
    for sheet in sheets:
        res = []
        data = read_xlsx(path, sheet)
        for row in data:
            temp_res = row[0:n_microstates]
            np_data = np.asarray(row[n_microstates::])
            for n_condition in range(n_conditions):
                index = []
                for j in range(n_microstates):
                    for i in range(n_condition*block_size, (n_condition+1)*block_size, n_microstates):
                        index.append(i+j)
                temp_res.extend(np_data[index])
            res.append(temp_res)
        write_info(save_path, sheet, res)


def formulate_condition_parameter(path, save_path, sheets, n_microstates, n_conditions, n_runs):
    for sheet in sheets:
        res = []
        data = read_xlsx(path, sheet)
        for row in data:
            temp = []
            temp_res = []
            for i in range(n_microstates, len(row), n_runs):
                temp.append(np.mean(row[i:i+n_runs]))
            for i in range(n_microstates):
                temp_res.append(row[i])
                for j in range(n_conditions):
                    temp_res.append(temp[i + j*n_microstates])
            res.append(temp_res)

        write_info(save_path, sheet, res)

def batch_test_homogeneity_within_condition_within_subject(data, comb, order, n_microstate):
    res = {}
    multi_res = []
    pool = Pool(8)
    for combination in comb:
        data_merged = []
        block_size = []
        for item in combination:
            data_merged.extend(data[item])
            block_size.append(len(data[item]))
        multi_res.append(
            pool.apply_async(test_homogeneity, ([data_merged, n_microstate, block_size, order, len(combination)],)))
    pool.close()
    pool.join()
    for i in range(len(multi_res)):
        temp = '*'.join(comb[i])
        res[temp] = multi_res[i].get()
    return res


def batch_test_homogeneity_within_task_across_subject(comb, task, order, n_microstate, subject_path):
    res = {}
    multi_res = []
    pool = Pool(8)
    for combination in comb:
        data_merged = []
        block_size = []
        for item in combination:
            data_temp = load_data(subject_path + "\\" + item + "_1_30_microstate_labels.json")[task]
            data_merged.extend(data_temp)
            block_size.append(len(data_temp))
        multi_res.append(
            pool.apply_async(test_homogeneity, ([data_merged, n_microstate, block_size, order, len(combination)],)))
    pool.close()
    pool.join()
    for i in range(len(multi_res)):
        temp = '*'.join(comb[i])
        res[temp] = multi_res[i].get()
    return res

def batch_surrogate_aif(joint_p, condition_p, n_markov, n_microstate, n_surrogate, n_lag, n_repetition):
    res = []
    multi_res = []
    pool = Pool(7)
    for i in range(n_repetition):
        multi_res.append(
            pool.apply_async(surrogate_aif, ([joint_p, condition_p, n_markov, n_microstate, n_surrogate, n_lag],))
        )
    pool.close()
    pool.join()
    for i in range(len(multi_res)):
        res.append(multi_res[i].get().tolist())
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
    return lrd.mutual_information(n_lag)

def batch_calculate_aif(data_path, tasks, n_microstate, lag, window_size, window_step, method='aif'):
    res = {}
    multi_res = []
    if window_size == -1:
        pool = Pool(11)
        for task in tasks:
            res[task] = []
            data = loadmat(data_path + "\\" + task + "_seq.mat")['EEG'].flatten()
            multi_res.append(pool.apply_async(aif, ([data, n_microstate, lag, method],)))
        for i in range(len(multi_res)):
            res[tasks[i]].append(multi_res[i].get().tolist())
    return res

def aif(para):
    lrd = MicrostateLongRangeDependence(para[0], para[1])
    if para[3] == 'aif':
        res = lrd.mutual_information(para[2])
    elif para[3] == 'paif':
        res = lrd.partial_mutual_information(para[2])
    return res


def batch_calculate_dfa(data, n_microstate, n_partitions, segment_range, segment_density):
    res = []
    multi_res = []
    pool = Pool(11)
    lrd = MicrostateLongRangeDependence(data, n_microstate)
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

def test_homogeneity(para):
    sequence = MicrostateMarkov(para[0], para[1])
    p = sequence.test_conditional_homogeneity(para[2], para[3], para[4])
    return p


def conditions_tasks(conditions, tasks):
    res = {}
    for condition in conditions:
        res[condition] = []
        for task in tasks:
            if task.startswith(condition):
                res[condition].append(task)
    return res


if __name__ == '__main__':
    # tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'all')
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')

    # subject_path = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\labels\global_eegmaps_all_conditions'
    # save_path = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\markov\stationary\individual_eegmaps_pu_ig_ie.xlsx'
    # save_path_json = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\markov\stationary\global_eegmaps_all_conditions.json'
    # res_dict = {}
    # block_size = [250, 500, 1000, 2000]
    # block_size = [500, 1000, 5000, 10000]

    parameters = [
        {
        'tasks': read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered'),
        'conditions': read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered'),
        'subject_path': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\sequences',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\runs_parameters.json',
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\runs_parameters.json',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\condition',
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\condition',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\condition_m',
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\condition_m',
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\condition_m',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\condition_m_t_test',

        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\task',
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\task',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\task_m',
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\task_m',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\task_m_t_test',

        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\pic',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=4\parameters\pic',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\aif\subjects'
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\subjects'
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\subjects',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\subjects_formulation',
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\subjects_formulation',
        # 'save_path_fig': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\run-wise_pic',
        # 'save_path_fig': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\condition-wise_pic',
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\dfa\subjects_formulation',

        # 'save_path_excel': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=4\dfa\condition.xlsx',
        # 'read_path_excel': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=4\dfa\condition.xlsx',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\entropy_rate\history_6',
        'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\entropy_rate\history_6',
        'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\entropy_rate\history_6_pic',
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=4\aif\subjects',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=4\aif\subjects_pic',
        # 'save_path_fig': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=7\dfa\condition-wise_pic',
        # 'read_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=7\dfa\subjects_formulation',
        # 'save_path_json': r'D:\EEGdata\clean_data_six_problem\1_40\microstate\k=7\markovianity\markovianity.json',

    },
    ]



    # dfa for surrogate data
    # for para in parameters:
    #     save_path_json = para['save_path_json']
    #     read_path_json = para['read_path_json']
    #     tasks = para['tasks']
    #     n_microstates = 7
    #     n_partitions = 3
    #     n_repetition = 100
    #     res = {}
    #     for subject in subjects:
    #         print(subject)
    #         res[subject] = {}
    #         data = load_data(read_path_json + "\\" + subject + "_markov_1_repetition_100.json")
    #         for task in tasks:
    #             res[subject][task] = []
    #             for repetition in range(n_repetition):
    #                 segment_range = [2, len(data[task][repetition]['surrogate_data'])]
    #                 segment_density = 0.5
    #                 res[subject][task].append(batch_calculate_dfa(data[task][repetition]['surrogate_data'], n_microstates, n_partitions, segment_range,
    #                                                          segment_density))
    #         json.dump(res, codecs.open(save_path_json + "\\" + subject + "_hurst_exponent.json", 'w', encoding='utf-8'),
    #                   separators=(',', ':'), sort_keys=True)

    # plot entropy rate
    for para in parameters:
        # n_row = 2
        # n_col = 3
        n_row = 1
        n_col = 1
        read_path_json = para['read_path_json']
        save_path_json = para['save_path_json']
        conditions = para['conditions']
        tasks = para['tasks']
        title = ['Rest', 'Problem understanding', 'Idea generation', 'Rating idea generation', 'Idea evaluation', 'Rating idea evaluation']
        c_t = conditions_tasks(conditions,tasks)
        for condition in conditions:
            fig, ax = plt.subplots(n_row, n_col, figsize=(20, 8), dpi=600)
            for row in range(n_row):
                for col in range(n_col):
                    m = row * n_col + col
                    # read_path = read_path_json + "\\" + conditions[m] + '.mat'
                    read_path = read_path_json + "\\" + 'condition' + '.mat'
                    data = loadmat(read_path)['EEG']
                    # plot_block(data.mean(axis=1), stats.sem(data, axis=1), c_t[conditions[m]], ax[row][col], title[m], 'Entropy rate')
                    plot_block(data.mean(axis=1), stats.sem(data, axis=1), conditions, ax, 'Design activity', 'Entropy rate')
            plt.subplots_adjust(hspace=0.4, wspace=0.5)
            # plt.savefig(save_path_json + "\\" + 'task-wise.png', dpi=600)
            plt.savefig(save_path_json + "\\" + 'condition-wise.png', dpi=600)
            # plt.show()
            # plt.close()

    # entropy_rate
    # for para in parameters:
    #     read_path_json = para['read_path_json']
    #     save_path_json = para['save_path_json']
    #     tasks = para['tasks']
    #     conditions = para['conditions']
    #     conditions_res = np.zeros((len(conditions), len(subjects)))
    #     for index, condition in enumerate(conditions):
    #         res = []
    #         for task in tasks:
    #             temp = []
    #             if task.startswith(condition):
    #                 for subject in subjects:
    #                     data = load_data(read_path_json + "\\" + subject + ".json")[task]['entropy_rate']
    #                     temp.append(data)
    #             if temp:
    #                 res.append(temp)
    #         np_res = np.asarray(res)
    #         conditions_res[index, :] = np_res.mean(axis=0)
    #         savemat(save_path_json+"\\" + condition + ".mat", {'EEG':np_res})
    #     savemat(save_path_json+"\\" + 'condition' + ".mat", {'EEG':conditions_res})


    # plot and t-test empirical data of hurst exponent
    # for para in parameters:
    #     # read_path_excel = para['read_path_excel']
    #     # p_read_path_excel = para['p_read_path_excel']
    #     read_path_json = para['read_path_json']
    #     save_path_fig = para['save_path_fig']
    #     conditions = para['conditions']
    #     tasks = para['tasks']
    #     c_t = conditions_tasks(conditions, tasks)
    #     n_microstates = 7
    #     n_condition = len(conditions)
    #     n_paritions = 3
    #     row = 2
    #     column = 5
    #     all_comb = set([i for i in range(n_microstates)])
    #     comb = [item for item in combinations([i for i in range(n_microstates)], n_paritions)]
    #     label_str = ['_1_10.png', '_11_20.png', '_21_30.png', '_31_35.png']
    #     label_index = [0, 10, 20, 30]
    #
    #     for index, index_item in enumerate(label_index):
    #         for condition in conditions:
    #             fig, ax = plt.subplots(row, column, figsize=(20, 8), dpi=600)
    #             for i in range(row):
    #                 for j in range(column):
    #                     if i == 1 and index == len(label_index)-1:
    #                         ax[i][j].axis('off')
    #                         continue
    #                     item = comb[i*column+j+index_item]
    #                     data = loadmat(read_path_json+"\\"+'condition'+"_"+to_string(item)+".mat")['EEG']
    #                     # data = np.asarray(read_xlsx(read_path_excel, to_string(item)))
    #                     # p_value = np.asarray(read_xlsx(p_read_path_excel, to_string(item)))[:, 1]
    #                     title = to_string(item) + " Vs. " + to_string(all_comb - set(item))
    #                     if row == 1:
    #                         plot_block(data.mean(axis=0), stats.sem(data, axis=0), conditions, ax[j], title, 'H')
    #                         # plot_block(data.mean(axis=0), stats.sem(data, axis=0), c_t[condition], ax[j], title, 'H')
    #                     else:
    #                         # plot_block(data.mean(axis=0), stats.sem(data, axis=0), c_t[condition], ax[i][j], title, 'H')
    #                         plot_block(data.mean(axis=1), stats.sem(data, axis=1), conditions, ax[i][j], title, 'H')
    #
    #             plt.subplots_adjust(hspace=0.4, wspace=0.5)
    #             # plt.savefig(save_path_fig + "\\" + condition + "\\" + condition + "_1_10" + ".png", dpi=600)
    #             plt.savefig(save_path_fig + "\\" + 'condition' + label_str[index], dpi=600)
    #             # plt.show()
    #             plt.clf()
    #             plt.close()



    # formulate empirical data
    # for para in parameters:
    #     # save_path_excel = para['save_path_excel']
    #     save_path_json = para['save_path_json']
    #     read_path_json = para['read_path_json']
    #     tasks = para['tasks']
    #     conditions = para['conditions']
    #     n_microstates = 7
    #     n_paritions = 3
    #     comb = combinations([i for i in range(n_microstates)], n_paritions)
    #     for index, item in enumerate(comb):
    #         res = np.zeros((len(conditions), len(subjects)))
    #         for c_index, condition in enumerate(conditions):
    #             temp = []
    #             for task in tasks:
    #                 slope = []
    #                 if task.startswith(condition):
    #                     for subject in subjects:
    #                         data = load_data(read_path_json + "\\" + subject + '.json')
    #                         slope.append(data[task][index][to_string(item)]['slope'])
    #                 if slope:
    #                     temp.append(slope)
    #             np_temp = np.asarray(temp)
    #             res[c_index, :] = np_temp.mean(axis=0)
    #             savemat(save_path_json+"\\"+condition+"_"+to_string(item)+".mat",{'EEG':np_temp})
    #         savemat(save_path_json+"\\"+"condition"+"_"+to_string(item)+".mat",{'EEG':res})



    # plot dfa
    # for para in parameters:
    #     save_path_json = para['save_path_json']
    #     read_path_json = para['read_path_json']
    #     tasks = para['tasks']
    #     n_microstates = 4
    #     f = 250
    #     for subject in subjects:
    #         data = load_data(read_path_json + "\\" + subject + '.json')
    #         os.mkdir(save_path_json + "\\" + subject)
    #         for task in tasks:
    #             for partition in data[task]:
    #                 for partition_index, value in partition.items():
    #                     f = value['fluctuation']
    #                     scales = value['scales']
    #                     coeff = np.polyfit(np.log2(scales), np.log2(f), 1)
    #                     slope = value['slope']
    #                     f_fit = 2**np.polyval(coeff, np.log2(scales))
    #                     plt.loglog(scales, f, 'bo')
    #                     plt.loglog(scales, f_fit, 'r', label=r'H = %0.2f'%slope)
    #                     plt.xlabel('time window')
    #                     plt.ylabel('F(t)')
    #                     plt.legend()
    #                     # plt.title(subject + "_" + task + "_partition_" + partition_index)
    #                     plt.title(task + "_partition_" + partition_index)
    #                     # plt.show()
    #                     plt.savefig(save_path_json + "\\" + subject + "\\" + task + "_" + partition_index +".png", dpi=100)
    #                     plt.close()

    # entropy_rate
    # for para in parameters:
    #     save_path_json = para['save_path_json']
    #     subject_path = para['subject_path']
    #     tasks = para['tasks']
    #     n_microstates = 7
    #     for lag in range(4, 7):
    #         for subject in subjects:
    #             print(subject)
    #             res = {}
    #             for task in tasks:
    #                 data = loadmat(subject_path + "\\" + subject + "\\" + task + "_seq.mat")['EEG'].flatten()
    #                 lrd = MicrostateLongRangeDependence(data, n_microstates)
    #                 entropy_rate, excess_entropy, index = lrd.excess_entropy_rate(lag)
    #                 res[task] = {'entropy_rate':entropy_rate.tolist(), 'excess_entropy':excess_entropy.tolist(), 'index':index.tolist()}
    #             json.dump(res, codecs.open(save_path_json + "\\history_" + str(lag) + '\\' + subject + ".json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)

    # dfa for empirical data
    # for para in parameters:
    #     save_path_json = para['save_path_json']
    #     subject_path = para['subject_path']
    #     tasks = para['tasks']
    #     n_microstates = 7
    #     n_partitions = 3
    #     for subject in subjects:
    #         print(subject)
    #         res = {}
    #         for task in tasks:
    #             data = loadmat(subject_path + "\\" + subject + "\\" +task + "_seq.mat")['EEG'].flatten()
    #             segment_range = [2, int(np.log2(len(data)))]
    #             segment_density = 0.25
    #             res[task] = batch_calculate_dfa(data, n_microstates, n_partitions, segment_range, segment_density)
    #         json.dump(res, codecs.open(save_path_json+"\\"+subject+".json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)

    # confidence interval
    # for para in parameters:
    #     read_path_excel = para['read_path_excel']
    #     save_path_excel = para['save_path_excel']
    #     save_path_json = para['save_path_json']
    #     read_path_json = para['read_path_json']
    #     read_path_task_length = para['read_path_task_length']
    #     subject_path = para['subject_path']
    #     tasks = para['tasks']
    #     matrix = load_data(read_path_json)
    #     task_length = load_data(read_path_task_length)
    #     for n_markov in range(5, 6):
    #         print(n_markov)
    #         for subject in subjects:
    #             print(subject)
    #             res = {}
    #             path = save_path_json + subject +"_markov_" + str(n_markov) + ".json"
    #             for task in tasks:
    #                 print(task)
    #                 n_surrogate = task_length[subject][task]
    #                 condition_p = matrix[subject][task][str(n_markov)]['condition_p']
    #                 joint_p = matrix[subject][task][str(n_markov)]['joint_p']
    #                 res[task] = batch_surrogate_aif(joint_p, condition_p, n_markov, 6, n_surrogate, 2000, 20)
    #             json.dump(res, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


    # surrogate data
    # for para in parameters:
    #     read_path_excel = para['read_path_excel']
    #     save_path_excel = para['save_path_excel']
    #     save_path_json = para['save_path_json']
    #     read_path_json = para['read_path_json']
    #     subject_path = para['subject_path']
    #     tasks = para['tasks']
    #     matrix = load_data(read_path_json)
    #     for subject in subjects:
    #         print(subject)
    #         res = {}
    #         path = save_path_json + "\\" + subject + "_1_30_surrogate_microstate_labels.json"
    #         data = load_data(subject_path + "\\" + subject + "_1_30_microstate_labels.json")
    #         for task in tasks:
    #             n_surrogate = len(data[task])
    #             res[task] = {}
    #             for k in range(1, 6):
    #                 condition_p = matrix[subject][task][str(k)]['condition_p']
    #                 joint_p = matrix[subject][task][str(k)]['joint_p']
    #                 res[task][k] = MicrostateMarkov.surrogate_markov_chain(joint_p, condition_p, k, 6, n_surrogate).tolist()
    #         json.dump(res, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    # markov probability
    # for para in parameters:
    #     # read_path_excel = para['read_path_excel']
    #     # save_path_excel = para['save_path_excel']
    #     save_path_json = para['save_path_json']
    #     subject_path = para['subject_path']
    #     tasks = para['tasks']
    #     res = {}
    #     for subject in subjects:
    #         print(subject)
    #         data = load_data(subject_path + "\\" + subject + "_microstate_labels.json")
    #         res[subject] = {}
    #         for task in tasks:
    #             res[subject][task] = {}
    #             markov = MicrostateMarkov(data[task], 7)
    #             for k in range(1, 2):
    #                 joint_p, condition_p = markov.transition_matrix_hat(k)
    #                 res[subject][task][k] = {'joint_p': joint_p.tolist(), 'condition_p':condition_p.tolist()}
    #     json.dump(res, codecs.open(save_path_json, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)



    # plot AIF with 95% confidence interval condition
    # for para in parameters:
    #     path = r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\labels_peaks\markov\confidence_interval\individual'
    #     tasks = para['tasks']
    #     conditions = para['conditions']
    #     alpha = 0.5
    #     empirical_data = load_data(r'D:\EEGdata\clean_data_six_problem\1_30\PU_IG_RATING_(IE+TYPE)_RATING\k=6\labels_peaks\lrd\aif_empirical\global_eegmaps_pu_ig_ie.json')
    #     for subject in subjects:
    #         surrogate_data = load_data("D:\\EEGdata\\clean_data_six_problem\\1_30\\PU_IG_RATING_(IE+TYPE)_RATING\\k=6\\labels_peaks\\markov\\surrogate\\aif\\markov_1\\" + subject + "_markov_1_repetition_100.json")
    #         lag = 2000
    #         lag_interval = [i for i in range(lag)]
    #         n_rep = 100
    #         for condition in conditions:
    #             plt.figure()
    #             for task in tasks:
    #                 if condition in task:
    #                     temp = np.zeros((n_rep, lag))
    #                     for i in range(n_rep):
    #                         temp[i,:] = surrogate_data[task][i]['mutual_info'][0:lag]
    #             ci = np.percentile(temp, [100.*alpha/2, 100.*(1-alpha/2.)], axis=0)
    #             ci[0,:] /= ci[0,0]
    #             ci[1,:] /= ci[1,0]
    #
    #             plt.semilogy(lag_interval, ci[0,:], color='gray')
    #             plt.semilogy(lag_interval, ci[1, :], color='gray')
    #
    #             plt.fill_between(lag_interval, ci[0,:], ci[1,:], alpha=0.5, facecolor='gray', label="Surrogate data")
    #             plt.semilogy(lag_interval, np.asarray(empirical_data[subject][task][0:lag])/empirical_data[subject][task][0], color='blue', label='Empirical data')
    #             plt.xlabel("time lag (2ms)")
    #             plt.ylabel("AIF (bits)")
    #             # plt.ticklabel_format(style='plain', axis='y')
    #             plt.title(subject + "_" + task)
    #             plt.legend()
    #             plt.savefig(path+"\\"+subject + "_" + task)


    # plot AIF with 95% confidence interval individual
    # for para in parameters:
    #     read_path_json = para['read_path_json']
    #     save_path_fig = para['save_path_fig']
    #     # save_path_excel = para['save_path_excel']
    #     tasks = para['tasks']
    #     tasks_title = para['tasks_title']
    #     conditions = para['conditions']
    #     alpha = 0.05
    #     lag = 500
    #     n_rep = 100
    #     f = 2
    #     lag_interval = np.arange(0, lag*2, f)
    #     res = []
    #     count = 1
    #     for subject in subjects[0::]:
    #         subject_name = 'Subject ' + str(count)
    #         count += 1
    #         empirical_data = load_data(read_path_json+"\\"+subject)
    #         surrogate_data = load_data(r"D:\EEGdata\clean_data_six_problem\2_20\PU_IG_RATING_(IE+TYPE)_RATING\labels\mutual_info\surrogate_data\data" + "\\"+ subject + "_markov_1_repetition_100.json")
    #         for task_index, task in enumerate(tasks):
    #             plt.figure()
    #             temp = np.zeros((n_rep, lag))
    #             for i in range(n_rep):
    #                 temp[i,:] = np.asarray(surrogate_data[task][i]['mutual_info'][0:lag])/surrogate_data[task][i]['mutual_info'][0]
    #                 # temp[i, :] = np.asarray(surrogate_data[task][i]['mutual_info'][0:lag])
    #             ci = np.percentile(temp, [100.*alpha/2, 100.*(1-alpha/2.)], axis=0)
    #             empirical = np.asarray(empirical_data[task][0][0:lag])/empirical_data[task][0][0]
    #             singal_peaks_index = signal.find_peaks(empirical)[0]
    #             high_peaks_index = np.where((empirical - ci[1,:]) > ci[1,:]*0.1)[0]
    #             peaks_index = np.intersect1d(singal_peaks_index, high_peaks_index)
    # #             res_temp = [subject, task, first_peaks[0]] if len(first_peaks) > 0 else [subject, task, 'n.k.']
    # #             res.append(res_temp)
    #             plt.plot(lag_interval, ci[0, :], color='gray')
    #             plt.plot(lag_interval, ci[1, :], color='gray')
    # #     write_info(save_path_excel, 'p=0.01', res)
    #             plt.fill_between(lag_interval, ci[0,:], ci[1,:], alpha=0.5, facecolor='gray', label="Surrogate data")
    #             plt.plot(lag_interval, empirical, color='blue', label='Empirical data')
    #             if len(peaks_index) > 0:
    #                 plt.scatter(peaks_index*f, empirical[peaks_index], c='', marker='o', edgecolors='r', linewidths=2)
    #             plt.yscale('log')
    #             plt.xlabel("Time lag (ms)")
    #             plt.ylabel("AIF (bits)")
    #             plt.title(subject_name + "\n" + tasks_title[task_index])
    #             plt.legend()
    #             # plt.show()
    #             plt.savefig(save_path_fig + "\\" + subject + "_" + tasks_title[task_index], dpi=1200)
    #             plt.close()

    #plot AIF
    # for para in parameters:
    #     # read_path_excel = para['read_path_excel']
    #     # save_path_excel = para['save_path_excel']
    #     read_path_json = para['read_path_json']
    #     save_path_json = para['save_path_json']
    #     tasks = para['tasks']
    #     f = 250
    #     dt = 1000 / f
    #     for subject in subjects:
    #         data = load_data(read_path_json+"\\"+subject)
    #         os.mkdir(save_path_json+"\\"+subject)
    #         for task in tasks:
    #             np_data = np.asarray(data[task]).flatten()
    #             plt.semilogy(dt*np.arange(np_data.shape[0]), np_data/np_data[0])
    #             plt.xlabel("Time lag (ms)")
    #             plt.ylabel("AIF (bits)")
    #             plt.title(task)
    #             # plt.show()
    #             plt.savefig(save_path_json+"\\"+subject+"\\"+task+'.png', dpi=200)
    #             plt.clf()
    #             plt.close()

    # AIF or PAIF
    # for para in parameters:
    #     # read_path_excel = para['read_path_excel']
    #     # save_path_excel = para['save_path_excel']
    #     save_path_json = para['save_path_json']
    #     subject_path = para['subject_path']
    #     tasks = para['tasks']
    #     # res = {}
    #     lag = 500
    #     window_size = -1
    #     window_step = 0
    #     n_microstate = 7
    #     for subject in subjects:
    #         print(subject)
    #         res_excel = []
    #         data_path = subject_path + "\\" + subject
    #         res = batch_calculate_aif(data_path, tasks, n_microstate, lag, window_size, window_step, 'aif')
    #         json.dump(res, codecs.open(save_path_json + "\\" +subject, 'w', encoding='utf-8'), separators=(',', ':'))

    # formulate excel by task mean
    # for para in parameters:
    #     read_path_excel = para['read_path_excel']
    #     save_path_excel = para['save_path_excel']
    #     conditions = para['conditions']
    #     sheet_names = ['duration', 'frequency', 'coverage']
    #     for sheet_name in sheet_names:
    #         for condition in conditions:
    #             temp = []
    #             for i in range(6):
    #                 temp_sheet_name = sheet_name + "_" + condition + "M" + str(i)
    #                 data = np.asarray(read_xlsx(read_path_excel, temp_sheet_name))
    #                 temp.append(data.mean(axis=1).tolist())
    #             write_info(save_path_excel, sheet_name + "_" + condition, temp, False)

    #formulate markovianity
    # for para in parameters:
    #     read_path_json = para['save_path_json']
    #     conditions = para['conditions']
    #     tasks = para['tasks']
    #     data = load_data(read_path_json)
    #     orders = [str(i) for i in range(3)]
    #     for order in orders:
    #         for condition in conditions:
    #             count = 0
    #             p_count = 0
    #             for task in tasks:
    #                 for subject in subjects:
    #                     if task.startswith(condition):
    #                         count += 1
    #                         p = data[subject][order][task]
    #                         if p < 0.01:
    #                             p_count += 1
    #             print(order, condition, count, p_count, p_count/count)



    # plot parameters
    # for para in parameters:
    #     n_microstate = 7
    #     n_row = 2
    #     n_col = int(math.ceil(n_microstate / n_row))
    #     read_path_json = para['read_path_json']
    #     save_path_json = para['save_path_json']
    #     conditions = para['conditions']
    #     tasks = para['tasks']
    #     c_t = conditions_tasks(conditions, tasks)
    #     paras = ['duration', 'frequency', 'coverage']
    #     y_label = {'duration':'Duration (ms)', 'frequency':'Occurrence (times/second)', 'coverage':'Coverage (%)'}
    #     alphabet_string = list(string.ascii_uppercase)
    #     title = ['Microstate ' + alphabet_string[i] for i in range(n_microstate)]
    #
    #     for pa in paras:
    #         # for condition in conditions:
    #             fig, ax = plt.subplots(n_row, n_col, figsize=(14, 8))
    #             for row in range(n_row):
    #                 for col in range(n_col):
    #                     if n_col * row + col == n_microstate:
    #                         ax[row][col].axis('off')
    #                         break
    #                     m = row * n_col + col
    #                     read_path = read_path_json + "\\" + 'condition' + '_' + pa +'_m' + str(m) + '.mat'
    #                     save_path = save_path_json + "\\" + 'condition' + '_' + pa + '.png'
    #                     data = loadmat(read_path)['EEG']
    #                     # plot_block(data.mean(axis=1), stats.sem(data, axis=1), c_t[condition], ax[row][col], title[m], y_label[pa])
    #                     plot_block(data.mean(axis=1), stats.sem(data, axis=1), conditions, ax[row][col], title[m], y_label[pa])
    #             plt.subplots_adjust(hspace=0.4, wspace=0.4)
    #             plt.savefig(save_path, dpi=600)
    #             # plt.show()
    #             # plt.close()



    # t-test
    # for para in parameters:
    #     read_path_json = para['read_path_json']
    #     save_path_json = para['save_path_json']
    #     n_microstate = 7
    #     conditions = para['conditions']
    #     paras = ['duration', 'frequency', 'coverage']
    #     alpha = 0.95
    #     sample_size = 27
    #     for pa in paras:
    #         for m in range(n_microstate):
    #             # for condition in conditions:
    #                 res = []
    #                 res_t = []
    #                 res_interval = [[],[]]
    #                 # save_path = save_path_json + "\\" + condition + '_' + pa +"_m" + str(m) + '.mat'
    #                 # read_path = read_path_json + "\\" + condition + '_' + pa +'_m' + str(m) + '.mat'
    #                 save_path = save_path_json + "\\" + 'condition' + '_' + pa + "_m" + str(m) + '.mat'
    #                 save_path_t = save_path_json + "\\" + 'condition' + '_' + pa + "_mt" + str(m) + '.mat'
    #                 save_path_c = save_path_json + "\\" + 'condition' + '_' + pa + "_mc" + str(m) + '.mat'
    #                 read_path = read_path_json + "\\" + 'condition' + '_' + pa + '_m' + str(m) + '.mat'
    #                 data = loadmat(read_path)['EEG']
    #                 for comb in itertools.combinations([i for i in range(data.shape[0])], 2):
    #                     p = stats.ttest_rel(data[comb[0], :], data[comb[1], :])
    #                     res.append(p[1])
    #                     res_t.append(p[0])
    #                     diff = data[comb[0], :] - data[comb[1], :]
    #                     diff_mean = np.mean(diff)
    #                     diff_std = stats.sem(diff)
    #                     ci = stats.t.interval(alpha, sample_size - 1, loc=diff_mean, scale=diff_std)
    #                     res_interval[0].append(ci[0])
    #                     res_interval[1].append(ci[1])
    #                 # savemat(save_path, {'EEG': np.asarray(res)})
    #                 savemat(save_path_t, {'EEG': np.asarray(res_t)})
    #                 savemat(save_path_c, {'EEG': np.asarray(res_interval)})
    # formulate t-test
    # for para in parameters:
    #     read_path_json = para['read_path_json']
    #     save_path_json = para['save_path_json']
    #     conditions = para['conditions']
    #     paras = ['duration', 'frequency', 'coverage']
    #     n_microstate = 7
    #     length = n_microstate * len(subjects)
    #     m_index = []
    #     for m in range(n_microstate):
    #         temp = []
    #         for n in range(m, length, n_microstate):
    #             temp.append(n)
    #         m_index.append(temp)
    #     for pa in paras:
    #         for condition in conditions:
    #             read_path = read_path_json + '\\' + condition + '_' + pa + '.mat'
    #             # read_path = read_path_json + '\\' + 'condition' + '_' + pa + '.mat'
    #             data = loadmat(read_path)['EEG']
    #             for index, value in enumerate(m_index):
    #                 save_path = save_path_json + "\\" + condition + '_' + pa + '_m' + str(index) + '.mat'
    #                 # save_path = save_path_json + "\\" + 'condition' + '_' + pa + '_m' + str(index) + '.mat'
    #                 savemat(save_path, {'EEG':data[:, value]})

    #formulate task-wise and condition-wise paramters
    # for para in parameters:
    #     read_path_json = para['read_path_json']
    #     save_path_json = para['save_path_json']
    #     tasks = para['tasks']
    #     conditions = para['conditions']
    #     n_microstate = 7
    #     paras = ['duration', 'frequency', 'coverage']
    #     for pa in paras:
    #         res = np.zeros((len(conditions), n_microstate*len(subjects)))
    #         for index, condition in enumerate(conditions):
    #             temp_res = []
    #             for task in tasks:
    #                 temp = []
    #                 for subject in subjects:
    #                     if task.startswith(condition):
    #                         data = load_data(read_path_json)[subject][task][pa]
    #                         temp.extend(data)
    #                 if temp:
    #                     temp_res.append(temp)
    #             np_data = np.asarray(temp_res)
    #             res[index, :] = np_data.mean(axis=0)
    #             savemat(save_path_json+"\\"+condition + '_' + pa + '.mat', {'EEG':np_data})
    #         # savemat(save_path_json+"\\"+ 'condition_' + pa + '.mat', {'EEG':res})

    #formulate temporal parameters for ANOVA

    # for para in parameters:
    #     titles = ['coverage','duration','frequency']
    #     save_path_json = para['save_path_json']
    #     read_path_json = para['read_path_json']
    #     for title in titles:
    #         data = loadmat(read_path_json + "\\" +"condition_"+title)['EEG']
    #         res = []
    #         for i in range(0,data.shape[1],7):
    #             temp = data[:, i:i+7]. flatten()
    #             res.append(temp)
    #         write_info(save_path_json, title, res)



    # calculate temporal parameters
    # for para in parameters:
    #     f = 250
    #     f_m = 1000 / f
    #     n_microstate = 7
    #     res = {}
    #     res_xlsx = {'duration':[], 'frequency':[], 'coverage':[]}
    #     tasks = para['tasks']
    #     subject_path = para['subject_path']
    #     save_path_json = para['save_path_json']
    #     save_path_excel = para['save_path_excel']
    #     for subject in subjects:
    #         print(subject)
    #         res[subject] = {}
    #         # data = load_data(subject_path + "\\" + subject + "_microstate_labels.json")
    #         temp = [[],[],[]]
    #         for task in tasks:
    #             data = loadmat(subject_path + "\\" + subject + "\\" + task +"_seq.mat")['EEG'].flatten()
    #             sequence = MicrostateParameter(data, n_microstate)
    #             duration = sequence.calculate_duration()
    #             frequency = sequence.calculate_frequency(f)
    #             coverage = sequence.calculate_coverage()
    #             duration = [i*f_m for i in duration]
    #             coverage = [i*100 for i in coverage]
    #             res[subject][task] = {'duration': duration, 'frequency': frequency, 'coverage': coverage}
    #         #     temp[0].extend(duration)
    #         #     temp[1].extend(frequency)
    #         #     temp[2].extend(coverage)
    #         # res_xlsx['duration'].append([i*f_m for i in temp[0]])
    #         # res_xlsx['frequency'].append(temp[1])
    #         # res_xlsx['coverage'].append([i*100 for i in temp[2]])
    #     json.dump(res, codecs.open(save_path_json, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
    #     # write_info(save_path_excel, 'duration', res_xlsx['duration'])
    #     # write_info(save_path_excel, 'frequency', res_xlsx['frequency'])
    #     # write_info(save_path_excel, 'coverage', res_xlsx['coverage'])

    # test homogeneity between the same tasks across subjects
    # for para in parameters:
    #     print(para)
    #     tasks = para['tasks']
    #     conditions = para['conditions']
    #     subject_path = para['subject_path']
    #     save_path_json = para['save_path_json']
    #     cond = conditions_tasks(conditions, tasks)
    #     res = {}
    #     for task in tasks[1::]:
    #         print(task)
    #         res[task] = {}
    #         for combination_order in range(2, len(subjects)):
    #             index = 'combination_' + str(combination_order)
    #             res[task][index] = {}
    #             comb = [temp for temp in itertools.combinations(subjects, combination_order)]
    #             for order in range(3, 6):
    #                 p = batch_test_homogeneity_within_task_across_subject(comb, task, order, 6, subject_path)
    #                 res[task][index][str(order)] = p
    #     json.dump(res, codecs.open(save_path_json, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,
    #               indent=4)


    # test homogeneity between the similar task within subjects
    # for para in parameters:
    #     tasks = para['tasks']
    #     conditions = para['conditions']
    #     subject_path = para['subject_path']
    #     save_path_json = para['save_path_json']
    #     cond = conditions_tasks(conditions, tasks)
    #     res = {}
    #     for subject in subjects:
    #         print(subject)
    #         data = load_data(subject_path + "\\" + subject + "_1_30_microstate_labels.json")
    #         res[subject] = {}
    #         for key, value in cond.items():
    #             # print(value)
    #             for combination_order in range(2, 7):
    #                 index = 'combination_'+str(combination_order)
    #                 res[subject][index] = {}
    #                 comb = [temp for temp in itertools.combinations(value, combination_order)]
    #                 for order in range(1, 6):
    #                     p = batch_test_homogeneity_within_condition_within_subject(data, comb, order, 6)
    #                     res[subject][index][order] = p
    #                     # print(p)
    #     json.dump(res, codecs.open(save_path_json, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True,
    #               indent=4)

    # test Markovianity
    # for para in parameters:
    #     tasks = para['tasks']
    #     subject_path = para['subject_path']
    #     save_path_json = para['save_path_json']
    #     # save_path_excel = para['save_path_excel']
    #     n_microstate = 7
    #     n_order = 3
    #     res_dict = {}
    #     res = []
    #     for subject in subjects:
    #         print(subject)
    #         res_dict[subject] = {}
    #         # data = load_data(subject_path + "\\" + subject + "_microstate_labels.json")
    #         temp = []
    #         for order in range(n_order):
    #             print(order)
    #             res_dict[subject][str(order)] = {}
    #             for a_task in tasks:
    #                 data = loadmat(subject_path + "\\" + subject +"\\" + a_task +"_seq.mat")['EEG'].flatten()
    #                 sequence = MicrostateMarkov(data, n_microstate)
    #                 p = sequence.test_markov(order)
    #                 res_dict[subject][str(order)][a_task] = p
    #                 temp.append(p)
    #         res.append(temp)
    #     # write_info(save_path_excel, 'markovianity', res)
    #     json.dump(res_dict, codecs.open(save_path_json, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
