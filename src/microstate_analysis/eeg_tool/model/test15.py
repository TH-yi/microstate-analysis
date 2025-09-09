import numpy as np
from multiprocessing import Pool
import codecs, json
from collections import OrderedDict
from scipy import signal
import os
import mne
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, create_info, to_string
from scipy.io import loadmat, savemat
from scipy.stats import sem
from scipy import trapz, argmax
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

def plot_eegmaps(maps, row, col, info, title, ylabel=None, savepath=None, sign=None):
    fig = plt.figure(figsize=(19.2, 9.6))
    for i in range(row):
        for j in range(col):
            ax = fig.add_subplot(row, col, i*col+j+1)
            img, _ = mne.viz.plot_topomap(maps[i][j, :]*sign[j], info, show=False, axes=ax, vmin=min, vmax=max,
                                          image_interp='spline36', contours=6)
            if j == 0:
                x = ax.get_xlim()[0] - 1.2
                y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2-0.15
                ax.text(x, y, ylabel[i], fontsize=14)
                # ax.set_ylabel(ylabel[i], rotation='horizontal', labelpad=100, fontsize=14, fontweight='bold')
            if i == 0:
                ax.set_title(title[j], fontsize=14, pad=15)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad='15%')
        # fig.colorbar(img, ax=ax, cax=cax, orientation='vertical', cmap=cmap)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    if savepath:
        plt.savefig(savepath, dpi=600)
    else:
        plt.show()



if __name__ == '__main__':
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    tasks_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered_name')
    conditions = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered')
    conditions_name = read_subject_info(r'D:\EEGdata\clean_data_six_problem\condition_name.xlsx', 'm_all_ordered_name')
    conditions = ['1_rest', 'idea generation', 'idea evolution', 'idea rating']
    conditions_name.insert(0, 'Global')
    conditions_name = ['GLOBAL', 'REST', 'IG', 'IEV', 'RIGE']
    # conditions_name = ['GLOBAL', 'REST', 'PU', 'IG', 'RIG', 'IE', 'RIE']

    order = [5, 1, 4, 3, 0, 2]
    sign = [-1, 1, -1, 1, -1, -1]

    res = []
    info = create_info(250)

    # title = ['Microstate Class A','Microstate Class B','Microstate Class C','Microstate Class D','Microstate Class E','Microstate Class F','Microstate Class G']
    title = ['Microstate Class A', 'Microstate Class B', 'Microstate Class C', 'Microstate Class D',
             'Microstate Class E', 'Microstate Class F']

    title = ['Microstate class A','Microstate class B','Microstate class C','Microstate class D','Microstate class E','Microstate class F','Microstate class G']
    maps = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_conditions\reorder_microstate_across_runs_across_participants_across_conditions.json')
    res.append(np.asarray(maps))
    maps = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_participants\reorder_microstate_across_runs_across_participants.json')
    for condition in conditions:
        res.append(np.asarray(maps[condition]))


    # maps = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_conditions\reorder_microstate_across_runs_across_participants_across_conditions.json')
    maps = load_data(
        r'D:\EEGdata\clean_data_creativity\microstate_across_runs_across_participants_across_conditions.json')

    res.append(np.asarray(maps)[order,::])
    # maps = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\across_participants\reorder_microstate_across_runs_across_participants.json')
    maps = load_data(
        r'D:\EEGdata\clean_data_creativity\microstate_across_runs_across_participants.json')
    for condition in conditions:
        temp = np.asarray(maps[condition])[order,::]
        res.append(temp)

    # save_path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\pic\EEG_maps.png'
    save_path = r'D:\EEGdata\clean_data_creativity\EEG_maps.png'
    plot_eegmaps(res, 5, 6, info, title, conditions_name, savepath=save_path, sign=sign)


