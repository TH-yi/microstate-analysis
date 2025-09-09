from eeg_tool.utilis import load_data, read_subject_info, create_info
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne
import os


def plot_eegmap_one_row(maps, order=None, sign=None, savepath=None, savetitle=None):
    info = create_info(250)
    col = maps.shape[0]
    ax_all = gridspec.GridSpec(1, col, figure=plt.figure(figsize=(10, 10)))
    order = order if order else [i for i in range(col)]
    sign = sign if sign else [1 for i in range(col)]
    for index, j in enumerate(order):
        ax = plt.subplot(ax_all[index])
        temp = (maps[j] - maps[j].mean()) * sign[index]
        mne.viz.plot_topomap(temp, info, show=False, axes=ax, image_interp='spline36', contours=6)
    plt.title(savetitle)
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.close()


if __name__ == '__main__':
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    for subject in subjects:
        data = load_data(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\individual_run'+'\\'+subject+"_microstate.json")
        for task in tasks:
            savepath = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\pic\nmaps_4' + '\\' + subject + '_' + task +'.png'
            maps = np.asarray(data[task]['maps'][3])
            plot_eegmap_one_row(maps, savepath=savepath, savetitle=subject+"_"+task)

