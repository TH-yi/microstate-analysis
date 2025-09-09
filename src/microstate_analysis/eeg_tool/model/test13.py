import numpy as np
from multiprocessing import Pool
import codecs, json
from collections import OrderedDict
from scipy import signal
import os
import mne
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, create_info
from scipy.io import loadmat, savemat
from scipy.stats import sem
from scipy import trapz, argmax
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import rcParams
from scipy.signal import find_peaks
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']


def locmax(x, distance=2, n_std=3):
    dx = np.diff(x)
    zc = np.diff(np.sign(dx))
    peaks = 1 + np.where(zc == -2)[0]
    # peaks, _ = find_peaks(x, distance=distance, height=(x.mean() - n_std * x.std(), x.mean() + n_std * x.std()))
    return peaks

def peaks(data, title):
    res = []
    for i in range(data.shape[0]):
        res.append(locmax(data[i,:])[0]*4)
    print(round(np.mean(res),3), round(sem(res),3), title)

def plot_block(data, alpha, lag_interval, ax, title):
    mean_res_task = data.mean(axis=0)
    normalize_mean = mean_res_task / mean_res_task[0]
    ci = np.percentile(data, [100. * alpha / 2, 100. * (1 - alpha / 2.)], axis=0)
    ci[0, :] /= ci[0, 0]
    ci[1, :] /= ci[1, 0]
    ax.semilogy(lag_interval, ci[0, :], color='gray')
    ax.semilogy(lag_interval, ci[1, :], color='gray')
    ax.fill_between(lag_interval, ci[0, :], ci[1, :], label="95% CI", alpha=0.1, facecolor='gray')
    ax.semilogy(lag_interval, normalize_mean, color='red', label='Mean')
    ax.set_xlabel("Time lag (ms)", fontsize=14)
    ax.set_ylabel("AIF (bits)", fontsize=14)
    # ax.legend(fontsize=14)
    ax.set_title(title, fontsize=14)

def plot_block_ci(data, alpha, lag_interval, ax, title, condition_title=None):
    color = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan']
    n = len(data)
    for i in range(n):
        mean_res_task = data[i].mean(axis=0)
        normalize_mean = mean_res_task / mean_res_task[0]
        ci = np.percentile(data[i], [100. * alpha / 2, 100. * (1 - alpha / 2.)], axis=0)
        ci[0, :] /= ci[0, 0]
        ci[1, :] /= ci[1, 0]
        # ax.semilogy(lag_interval, ci[0, :], color='gray')
        # ax.semilogy(lag_interval, ci[1, :], color='gray')
        # ax.fill_between(lag_interval, ci[0, :], ci[1, :], alpha=0.1, color=color[i])
        ax.semilogy(lag_interval, normalize_mean, label=title[i], c=color[i], linewidth=2)
        ax.set_xlabel("Time lag", fontsize=14)
        ax.set_ylabel("AIF (bits)", fontsize=14)
        ax.legend(fontsize=10)
    if condition_title:
        ax.set_title(condition_title, fontsize=14,)


def plot_blocks(data, alpha, lag_interval, title, save_path, fit=False):
    n = len(data)
    if n < 3:
        row = 1
        col = 2
    else:
        row = 2
        col = 3
    fig = plt.figure(figsize=(19.2, 9.6))
    for i in range(n):
        ax = fig.add_subplot(row, col, i+1)
        if fit:
            exponential_fit_plot_block(data[i], lag_interval, ax, title[i])
        else:
            plot_block(data[i],alpha, lag_interval, ax, title[i])
            peaks(data[i], title[i])
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    if save_path:
        plt.savefig(save_path, dpi=600)
    else:
        plt.show()
    plt.close()

def exponential_fit_plot_block(data, lag_interval, ax, title):
    lag_interval = np.asarray(lag_interval, dtype=np.float32).flatten()
    mean_res_task = data.mean(axis=0)
    normalize_mean = (mean_res_task / mean_res_task[0]).flatten()
    popt, pcov = curve_fit(func_exp, lag_interval, normalize_mean)
    print(*popt, title)
    ax.scatter(lag_interval, normalize_mean, label='Mean')
    ax.plot(lag_interval, func_exp(lag_interval, *popt), 'r-', label='Fitted curve')
    ax.legend(fontsize=14)
    ax.set_title(title, fontsize=14, fontweight='bold')

def func_exp(x, b,):
    return np.exp(-b*x)



if __name__ == '__main__':
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    tasks_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered_name')
    conditions = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered')
    conditions_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered_name')
    conditions_name = ['REST', 'PU', 'IG', 'RIG', 'IE', 'RIE']
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\aif\subjects'
    resolution = int(1/250*1000)
    lag = 200
    lag_interval = [i for i in range(0, lag, resolution)]
    alpha = 0.05
    res = []
    fig = plt.figure(figsize=(19.2, 9.6))
    # ax = fig.add_subplot(1, 1, 1)
    for index, condition in enumerate(conditions):
        ax = fig.add_subplot(2,3,index+1)
        res_condition = []
        res_task_condition = []
        task_title = []
        save_path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\aif\t_200\task-wise' + "\\" + condition + "-wise.png"
        for task_index, task in enumerate(tasks):
            res_task = []
            if task.startswith(condition):
                task_title.append(tasks_name[task_index])
                for subject in subjects:
                    data = load_data(path + "\\" + subject)
                    res_task.append(data[task][0])
                    res_condition.append(data[task][0])
            if res_task:
                np_res_task = np.asarray(res_task)[:, 0:lag//resolution]
                res_task_condition.append(np_res_task)
        # plot_blocks(res_task_condition, alpha, lag_interval, task_title, save_path=save_path, fit=False)
        # plot_block_ci(res_task_condition, alpha, lag_interval, ax, task_title, conditions_name[index])
        np_res_condition = np.asarray(res_condition)[:, 0:lag//resolution]
        res.append(np_res_condition)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    # plt.savefig(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\aif\t_500\mean\task-wise.png',dpi=600)
    save_path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\aif\t_200\condition-wise' + "\\" + "condition-wise.eps"
    plot_blocks(res, alpha, lag_interval, conditions_name, save_path=save_path)

    # plot_block_ci(res, alpha, lag_interval, ax, conditions_name)
    # plt.show()
    # plt.savefig(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\aif\t_100\mean\condition-wise.png',dpi=600)

