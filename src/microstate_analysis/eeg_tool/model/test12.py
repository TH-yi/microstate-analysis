import numpy as np
from multiprocessing import Pool
import codecs, json
from collections import OrderedDict
from scipy import signal
import os
import mne
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, create_info
from eeg_tool.math_utilis import scaling_psd, bandpower
from scipy.io import loadmat, savemat
from scipy.stats import sem
from scipy import trapz, argmax
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

def merge_lobes():
    ch1 = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
    ch2 = [['Fp1', 'AF3', 'AF7', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3'],
          ['Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4'],
          ['FC5', 'C1', 'C3', 'C5'],
          ['FC4', 'C2', 'C4', 'C6'],
          ['FT7', 'T7', 'TP7', 'CP5', 'P5'],
          ['FT8', 'T8', 'TP8', 'CP6', 'P6'],
          ['CP1', 'CP3', 'P1', 'P3'],
          ['CP2', 'CP4', 'P2', 'P4'],
          ['PO3', 'PO7', 'P7', 'O1'],
          ['PO4', 'PO8', 'P8', 'O2']]
    res = []
    for lobes in ch2:
        temp = []
        for lobe in lobes:
            for index, ch in enumerate(ch1):
                if lobe == ch:
                    temp.append(index)
        res.append(temp)
    return res

def plot_eegmap_one_row(maps, info, title, savepath=None):
    n = maps.shape[0]
    max = np.max(maps)
    min = np.min(maps)
    fig = plt.figure()
    diff = maps - maps[0, :]
    cmap = 'RdBu_r' if max > 0 else 'Blues'
    print(cmap)
    for i in range(n):
        ax = fig.add_subplot(1, n, i+1)
        img, _ = mne.viz.plot_topomap(maps[i, :], info, cmap=cmap, show=False, axes=ax, vmin=min, vmax=max, image_interp='spline36', contours=6)
        ax.set_title(title[i])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, ax=ax, cax=cax, orientation='vertical', cmap=cmap)

    if savepath:
        plt.savefig(savepath, dpi=600)
    plt.show()

def plot_eegmaps(maps, row, col, info, title, power=['TRP Theta', 'TRP Alpha', 'TRP Beta', 'TRP Gamma'], savepath=None):

    fig = plt.figure(figsize=(19.2, 9.6))
    for i in range(row-1):
        cmap = 'RdBu_r' if np.max(maps[i]) > 0 else 'Blues_r'
        min = np.min(maps[i])
        max = np.max(maps[i])
        for j in range(col):
            ax = fig.add_subplot(row, col, i*col+j+1)
            img, _ = mne.viz.plot_topomap(maps[i][j, :], info, cmap=cmap, show=False, axes=ax, vmin=min, vmax=max,
                                          image_interp='spline36', contours=6)
            # if j == 0:
            #     x = ax.get_xlim()[0] - 1.5
            #     y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2 - 0.1
            #     ax.text(x, y, power[i], fontsize=18)
            # if i == 0:
            ax.set_title(title[j], fontsize=14, pad=15)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad='20%')
        fig.colorbar(img, ax=ax, cax=cax, orientation='vertical', cmap=cmap)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    if savepath:
        plt.savefig(savepath, dpi=600)
    else:
        plt.show()


if __name__ == '__main__':
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    conditions = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered')
    conditions_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered_name')
    conditions_name = ['REST','PU','IG','RIG','IE','RIE']
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_matlab\merged_epochs_data'
    title = ['Rest', 'Problem understanding', 'Idea generation', 'Rating idea generation', 'Idea evaluation',
             'Rating idea evaluation']
    info = create_info(250)
    window = 500
    scaling = 1
    band = [4, 7.5, 8, 13.5, 14, 29]
    lobes_index = merge_lobes()
    res = {}
    # for subject in subjects:
    #     res[subject] = {}
    #     for task in tasks:
    #         res[subject][task] = {}
    #         read_path = path + "\\" + subject + "\\" + task + ".mat"
    #         raw = mne.io.RawArray(loadmat(read_path)['EEG'], info)
    #         psds, freqs = mne.time_frequency.psd_welch(raw, 1, 40, n_fft=window, n_overlap=window // 2, n_per_seg=window, average='mean')
    #         psds = scaling_psd(psds, scaling)
    #         for i in range(0, len(band), 2):
    #             pow = bandpower(psds, freqs, band[i], band[i+1])
    #             temp = str(band[i]) + "_" + str(band[i+1])
    #             res[subject][task][temp] = pow.tolist()
    # json.dump(res,codecs.open(r'D:\EEGdata\clean_data_six_problem\1_40\trp\bandpower.json', 'w', encoding='utf-8'),separators=(',', ':'), sort_keys=True)
    #
    # data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\trp\bandpower.json')
    # res = {}
    # for subject in subjects:
    #     res[subject] = {}
    #     for task in tasks:
    #         res[subject][task] = {}
    #         for i in range(0, len(band), 2):
    #             temp = str(band[i]) + "_" + str(band[i + 1])
    #             activation_power = np.asarray(data[subject][task][temp])
    #             reference_power = np.asarray(data[subject]['rest_1'][temp])
    #             trp = np.log(activation_power) - np.log(reference_power)
    #             res[subject][task][temp] = trp.tolist()
    # json.dump(res, codecs.open(r'D:\EEGdata\clean_data_six_problem\1_40\trp\task_wise_trp.json', 'w', encoding='utf-8'),
    #           separators=(',', ':'), sort_keys=True)
    #
    # data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\trp\task_wise_trp.json')
    # res = {}
    # for i in range(0, len(band), 2):
    #     temp = str(band[i]) + "_" + str(band[i + 1])
    #     res[temp] = {}
    #     for subject in subjects:
    #         res[temp][subject] = {}
    #         for condition in conditions[1::]:
    #             condition_res = []
    #             for task in tasks:
    #                 if task.startswith(condition):
    #                     condition_res.append(data[subject][task][temp])
    #             mean = np.asarray(condition_res).mean(axis=0)
    #             res[temp][subject][condition] = mean.tolist()
    # json.dump(res, codecs.open(r'D:\EEGdata\clean_data_six_problem\1_40\trp\condition_wise_trp.json', 'w', encoding='utf-8'),separators=(',', ':'), sort_keys=True)
    #
    # data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\trp\condition_wise_trp.json')
    # res = {}
    # for i in range(0, len(band), 2):
    #     temp = str(band[i]) + "_" + str(band[i + 1])
    #     res[temp] = {}
    #     for subject in subjects:
    #         res[temp][subject] = {}
    #         for condition in conditions[1::]:
    #             np_data = np.asarray(data[temp][subject][condition])
    #             lobe = []
    #             for index in lobes_index:
    #                 lobe.append(np_data[index].mean())
    #             res[temp][subject][condition] = lobe
    # json.dump(res, codecs.open(r'D:\EEGdata\clean_data_six_problem\1_40\trp\condition_lobe_wise_trp.json', 'w', encoding='utf-8'),separators=(',', ':'), sort_keys=True)
    #
    # data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\trp\condition_lobe_wise_trp.json')
    # for i in range(0, len(band), 2):
    #     temp = str(band[i]) + "_" + str(band[i + 1])
    #     res = []
    #     for subject in subjects:
    #         temp_res = []
    #         for condition in conditions[1::]:
    #             temp_res.extend(data[temp][subject][condition])
    #         res.append(temp_res)
    #     write_info(r'D:\EEGdata\clean_data_six_problem\1_40\trp\trp.xlsx', temp, res)

    data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\trp\condition_wise_trp.json')
    res_maps = []
    row = 4
    col = 5
    # for i in range(0, len(band), 2):
    for i in range(0, len(band), 2):
        temp = str(band[i]) + "_" + str(band[i + 1])
        res = []
        for condition in conditions[1::]:
            res_condition = []
            for subject in subjects:
                res_condition.append(data[temp][subject][condition])
            res_condition = np.stack(res_condition)
            mean = res_condition.mean(axis=0)
            res.append(mean.tolist())
        res_maps.append(np.asarray(res))
    #     # plot_eegmap_one_row(np.asarray(res), info, conditions[1::])
    savepath = r'D:\EEGdata\clean_data_six_problem\1_40\trp\trp-topo_theta_alpha_beta.png'
    plot_eegmaps(res_maps, row, col, info, conditions_name[1::], savepath=savepath)
    plt.show()