from microstate_analysis.eeg_tool.algorithm.clustering.microstate import eegmaps_similarity
from microstate_analysis.microstate_base.data_handler import create_info
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import mne
import os
import codecs, json
from microstate_analysis.microstate_base.similarity_calc import eegmaps_similarity_old, eegmaps_similarity_hungarian

def plot_eeg(data, condition_name, n_k, channel_names=None,
             montage_path=None, sampling_frequency=500, channel_types="eeg", missing_channel_behavior="raise"):
    info = create_info(ch=channel_names, locs_path=montage_path, sfreq=sampling_frequency, ch_types=channel_types, on_missing=missing_channel_behavior)
    for k in range(n_k):
        ax_all = gridspec.GridSpec(10, 5, figure=plt.figure(figsize=(10, 100)))
        for a_condition_name in condition_name:
            maps = data[a_condition_name]
            for row in range(10):
                for col in range(5):
                    ax = plt.subplot(ax_all[row, col])
                    mne.viz.plot_topomap(np.asarray(maps[row][k]), info, show=False, axes=ax, image_interp='spline36',
                                         contours=6)
        plt.show()


def plot_eegmap_conditions(eegmaps, condition_name, n_k, channel_names=None, order=None, sign=None, savepath=None,
                           montage_path=None, sampling_frequency=500, channel_types="eeg", missing_channel_behavior="raise"):
    info = create_info(ch=channel_names, locs_path=montage_path, sfreq=sampling_frequency, ch_types=channel_types, on_missing=missing_channel_behavior)
    n_condition = len(condition_name)
    row = n_condition
    col = n_k
    fig = plt.figure(figsize=(10, 10))
    ax_all = gridspec.GridSpec(row, col, figure=fig)
    order = order if order else [i for i in range(col)]
    sign = sign if sign else [1 for i in range(col)]
    #condition_name = ['idea generation','idea evolution','idea rating' , '1_rest']

    for i in range(row):
        if isinstance(eegmaps[condition_name[i]], dict) and "maps" in eegmaps[condition_name[i]]:
            maps = np.asarray(eegmaps[condition_name[i]]['maps'])[order]
        else:
            maps = np.asarray(eegmaps[condition_name[i]])[order]
        for j in range(n_k):
            ax = plt.subplot(ax_all[i, j])
            temp = (maps[j] - maps[j].mean()) * sign[j]
            mne.viz.plot_topomap(temp, info, show=False, axes=ax, image_interp='spline36', contours=6)

    if savepath:
        plt.show()
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Conditions map saved at: {savepath}")
    else:
        plt.show()


def plot_eegmap_one_row(maps, order=None, sign=None, savepath=None, channel_names=None,
                        montage_path=None, sampling_frequency=500, channel_types="eeg",
                        missing_channel_behavior="raise"):
    info = create_info(ch=channel_names, locs_path=montage_path, sfreq=sampling_frequency, ch_types=channel_types, on_missing=missing_channel_behavior)

    # If maps is a dict and contains 'maps', extract it
    if isinstance(maps, dict) and "maps" in maps:
        maps = maps['maps']

    # Convert maps to a numpy array if it is a list
    if isinstance(maps, list):
        maps = np.array(maps)

    col = maps.shape[0]  # Now maps has a shape attribute

    # Set default values for order and sign if they are not provided
    order = order if order else [i for i in range(col)]
    sign = sign if sign else [1 for i in range(col)]

    # Create grid spec for plotting
    ax_all = gridspec.GridSpec(1, col, figure=plt.figure(figsize=(10, 10)))

    # Iterate over the columns and plot each topomap
    for index, j in enumerate(order):
        ax = plt.subplot(ax_all[index])
        temp = (maps[j] - maps[j].mean()) * sign[index]
        mne.viz.plot_topomap(temp, info, show=False, axes=ax, image_interp='spline36', contours=6)

    # Optionally save the plot
    if savepath:
        # Create the directory if it does not exist
        directory = os.path.dirname(savepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(savepath)

    #plt.show()


def plot_eegmaps_old(maps, task_name, channel_names=None, montage_path=None, sampling_frequency=500, channel_types="eeg", missing_channel_behavior="raise"):
    info = create_info(ch=channel_names, locs_path=montage_path, sfreq=sampling_frequency, ch_types=channel_types, on_missing=missing_channel_behavior)
    min_value = maps[0].min()
    max_value = maps[0].max()
    if len(task_name) == 1:
        ax_all = gridspec.GridSpec(1, len(task_name), figure=plt.figure(figsize=(10, 100)))
        for j in range(len(maps[0])):
            ax = plt.subplot(ax_all[j])
            mne.viz.plot_topomap(maps[0].T[:, j], info, show=False, axes=ax, image_interp='spline36', contours=6,
                                 vmin=min_value, vmax=max_value)
        plt.show()
    else:
        similarity = []
        row = len(task_name)
        col = 0
        for i in range(1, len(task_name)):
            temp = eegmaps_similarity(maps[0], maps[i])
            similarity.append(temp)
            col = max(col, max([item['index'] for item in [*temp.values()]]) + 1)
        ax_all = gridspec.GridSpec(row, col, figure=plt.figure(figsize=(10, 100)))
        ax_all.update(hspace=0.5)
        for i in range(row):
            for j in range(len(maps[i])):
                if i == 0:
                    ax = plt.subplot(ax_all[i, j])
                else:
                    ax = plt.subplot(ax_all[i, similarity[i - 1][j]['index']])
                    ax.set_title(round(similarity[i - 1][j]['similarity'], 4), fontsize=5)
                mne.viz.plot_topomap(maps[i].T[:, j], info, show=False, axes=ax, image_interp='spline36', contours=6,
                                     vmin=min_value, vmax=max_value)
        plt.show()


def plot_eegmaps(data, task_names, first_row_order=[], sign=None, savepath=None, minmax=False, channel_names=None,
                 montage_path=None, sampling_frequency=500, channel_types="eeg", missing_channel_behavior="raise"):
    info = create_info(ch=channel_names, locs_path=montage_path, sfreq=sampling_frequency, ch_types=channel_types, on_missing=missing_channel_behavior)
    maps = []
    for task_name in task_names:
        if task_name not in data:
            raise ValueError(f"Task {task_name} not found in the data!")
        if isinstance(data[task_name], list):
            task_maps = np.array(data[task_name])
        elif isinstance(data[task_name], dict):
            task_maps = np.array(data[task_name]['maps'])
        else:
            raise TypeError("Invalid input data")
        if task_maps.shape != (6, 63):
            raise ValueError(f"Task {task_name} maps should have shape [6, 63], but got {task_maps.shape}.")
        maps.append(task_maps)

    # Reorder first row maps
    reordered_maps = reorder_ndarray(maps[0], first_row_order)
    maps[0] = reordered_maps

    min_value = None
    max_value = None
    if minmax:
        min_value = np.min([map.min() for map in maps])
        max_value = np.max([map.max() for map in maps])

    fig = plt.figure(figsize=(15, 10))

    reordered_data = {}  # Store reordered maps and indices

    if len(task_names) == 1:
        if sign is None:
            sign = np.ones(6)
        ax_all = gridspec.GridSpec(1, 6, figure=fig)
        for j in range(len(maps[0])):
            ax = plt.subplot(ax_all[j])
            map_to_plt = (maps[0].T[:, j] - maps[0].T[:, j].mean()) * sign[j]
            mne.viz.plot_topomap(map_to_plt, info, show=False, axes=ax, image_interp='cubic',
                                 extrapolate='head', outlines='head', contours=6, sphere=0.095,
                                 vlim=(min_value, max_value))               # Add labels A-F above the first row
            ax.set_title(chr(65 + j), fontsize=16, weight='bold')  # chr(65) is 'A'
        reordered_data = first_row_order
    else:
        sign = sign if sign else np.ones((4, 6))
        similarity = []
        row = len(task_names)
        col = 6  # Fixed number of maps per task

        for i in range(1, len(task_names)):
            temp = eegmaps_similarity_hungarian(maps[0], maps[i])
            similarity.append(temp)

        ax_all = gridspec.GridSpec(row, col + 1, figure=fig)  # Extra column for task names
        ax_all.update(hspace=0.5)

        for i in range(row):
            reordered_data_task_key = task_names[i]
            if i == 0:
                reordered_data_task_indices = first_row_order
            else:
                reordered_data_task_indices = [-1] * len(first_row_order)

            # Add task name as a label in the first column
            ax_label = plt.subplot(ax_all[i, 0])
            ax_label.axis('off')  # Turn off the axis
            ax_label.text(0.2, 0.5, task_names[i], va='center', ha='center', fontsize=20, transform=ax_label.transAxes)

            for j in range(len(maps[i])):
                if i == 0:
                    ax = plt.subplot(ax_all[i, j + 1])
                    # Add labels A-F above the first row
                    ax.set_title(chr(65 + j), fontsize=16, weight='bold')  # chr(65) is 'A'
                else:
                    new_place = similarity[i - 1][j]['index']
                    ax = plt.subplot(ax_all[i, new_place + 1])
                    reordered_data_task_indices[int(new_place)] = j

                map_to_plt = (maps[i].T[:, j] - maps[i].T[:, j].mean()) * sign[i][j]
                mne.viz.plot_topomap(map_to_plt, info, show=False, axes=ax, image_interp='cubic',
                                     extrapolate='head', outlines='head',  contours=6, sphere=0.095,
                                     vlim=(min_value, max_value))

            reordered_data[reordered_data_task_key] = reordered_data_task_indices

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Conditions map saved at: {savepath}")
    else:
        plt.show()

    return reordered_data



def reorder_nested_list(nested_list, indices):
    """
    Reorder a nested list (6x63) based on the given indices.

    Parameters:
        nested_list (list of lists): The input nested list with shape (6x63).
        indices (list of int): A list specifying the new order of the rows.

    Returns:
        list of lists: The reordered nested list.
    """
    if len(nested_list) != len(indices):
        raise ValueError("Length of indices must match the number of rows in the nested list.")
    if not all(0 <= idx < len(nested_list) for idx in indices):
        raise ValueError("Indices must be valid row indices of the nested list.")

    # Reorder the nested list
    reordered_list = [nested_list[i] for i in indices]
    return reordered_list


def reorder_ndarray(arr, order):
    """
    Reorder the rows of a 2D numpy array based on a given order list.

    Parameters:
        arr (ndarray): A 2D numpy array with shape (6, 63).
        order (list of int): A list specifying the new order of the rows.

    Returns:
        ndarray: The reordered numpy array.
    Default value: 0, 1, 2, 3, 4, 5
     Example:
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        order = [2, 0, 1]
        reordered = reorder_ndarray(arr, order)
        print(reordered)
        [[5 6]
         [1 2]
         [3 4]]
    """
    # Check that the length of the order list is correct
    if len(arr) != len(order):
        raise ValueError("The length of the order list must match the number of rows in the array.")

    # Reorder the ndarray using the order list
    reordered_arr = arr[order]

    return reordered_arr
def save_eegmpas(clean_data_fname, data_fname, task_name):
    data_text = codecs.open(clean_data_fname + "\\" + data_fname, 'r', encoding='utf-8').read()
    data = json.loads(data_text)
    microstate_map = []
    for a_task_name in task_name:
        microstate_map.append(np.asarray(data[a_task_name]["maps_list"][data[a_task_name]["opt_k_index"]]))
    plot_eegmaps(microstate_map, task_name)


def save_eegmaps_subjects(clean_data_fname, data_fname, task_name, subjects):
    for subject in subjects:
        save_eegmpas(clean_data_fname, subject + data_fname, task_name)