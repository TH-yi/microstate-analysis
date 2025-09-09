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
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def scaling_psd(psds, scaling, dB=False):
    psds *= scaling * scaling
    if dB:
        np.log10(np.maximum(psds, np.finfo(float).tiny), out=psds)
    return psds



def plot_eegmap_one_row(maps, info, title, savepath=None):
    n = maps.shape[0]
    max = np.max(maps)
    min = np.min(maps)
    fig = plt.figure()
    diff = maps - maps[0, :]
    min_diff = np.min(diff)
    max_diff = np.max(diff)
    cmap = 'RdBu_r' if max > 0 else 'Blues'
    cmap_diff = 'RdBu_r' if max_diff > 0 else 'Blues'
    for i in range(n):
        ax = fig.add_subplot(2, n, i+1)
        img, _ = mne.viz.plot_topomap(maps[i, :], info, cmap=cmap, show=False, axes=ax, vmin=min, vmax=max, image_interp='spline36', contours=6)
        ax.set_title(title[i])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, ax=ax, cax=cax, orientation='vertical', cmap=cmap)

    for i in range(n, 2*n):
        ax = fig.add_subplot(2, n, i+1)
        img, _ = mne.viz.plot_topomap(diff[i-n, :], info, cmap=cmap_diff, show=False, axes=ax, vmin=min_diff, vmax=max_diff, image_interp='spline36', contours=6)
        ax.set_title(title[i-n])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, ax=ax, cax=cax, orientation='vertical', cmap=cmap_diff)

    if savepath:
        plt.savefig(savepath, dpi=600)
    plt.show()

def bandpower(psds, freqs, fmin, fmax, dx=1.0):
    ind_min = argmax(freqs > fmin) - 1
    ind_max = argmax(freqs > fmax) - 1
    return trapz(psds[:, ind_min: ind_max], freqs[ind_min: ind_max], dx=dx)

if __name__ == '__main__':
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    conditions = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered')
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_matlab\merged_epochs_data'
    title = ['Rest', 'Problem understanding', 'Idea generation', 'Rating idea generation', 'Idea evaluation', 'Rating idea evaluation']
    titles = {'rest':{'eeg':'Rest'},'pu':{'eeg':'Problem understanding'},'ig':{'eeg':'Idea generation'},
              'rig':{'eeg':'Rating idea generation'},'ie':{'eeg':'Idea evaluation'},'rie':{'eeg':'Rating idea evaluation'}}
    info = create_info(250)
    window = 500
    n_ch = 63
    n_freq = 79
    res = {}
    scaling = 1
    for i, condition in enumerate(conditions):
        temp = []
        res[condition] = {}
        for task in tasks:
            if task.startswith(condition):
                for subject in subjects:
                    read_path = path + "\\" + subject + "\\" + task + ".mat"
                    raw = mne.io.RawArray(loadmat(read_path)['EEG'], info)
                    psds, freqs = mne.time_frequency.psd_welch(raw, 1, 40, n_fft=window, n_overlap=window//2, n_per_seg=window, average='mean')
                    psds = scaling_psd(psds, scaling)
                    temp.append(psds)
        psds = np.stack(temp, axis=0).mean(axis=0)
        fig, ax = plt.subplots()
        mne.viz.utils.plot_psd_freq(info, freqs, psds, fig, [ax], titles[condition], semilogy=True)
        plt.show()
    #     res[condition]['psds'] = psds.tolist()
    #     res[condition]['freqs'] = freqs.tolist()
    # json.dump(res, codecs.open(r'D:\EEGdata\clean_data_six_problem\1_40\mean_psd\psd_1_40.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)

    # data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\mean_psd\psd_1_40.json')
    # fmin = 32
    # fmax = 40
    # res = {}
    # for i, condition in enumerate(conditions):
    #     psds = np.asarray(data[condition]['psds'])
    #     freqs = np.asarray(data[condition]['freqs'])
    #     # fig, ax = plt.subplots()
    #     # fig = mne.viz.utils.plot_psd_freq(info, freqs, psds, fig, [ax],{'eeg':title[i]})
    #     # fig.savefig(r'D:\EEGdata\clean_data_six_problem\1_40\mean_psd' + '\\' + title[i] +".png", dpi=600)
    #     power = bandpower(psds, freqs, fmin, fmax)
    #     res[condition] = power.tolist()
    # json.dump(res,codecs.open(r'D:\EEGdata\clean_data_six_problem\1_40\mean_psd\bandpower_32_40.json', 'w', encoding='utf-8'),separators=(',', ':'), sort_keys=True)

    # data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\mean_psd\bandpower_4_6.json')
    # savepath = r'D:\EEGdata\clean_data_six_problem\1_40\mean_psd\alpha_power.png'
    # res = []
    # for condition in conditions:
    #     res.append(data[condition])
    # np_res = np.stack(res, axis=0)
    # plot_eegmap_one_row(np_res, info, conditions)