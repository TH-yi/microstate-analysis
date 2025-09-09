import numpy as np
from multiprocessing import Pool
import json
import codecs

from microstate_analysis.microstate_base.microstate import Microstate
from microstate_analysis.microstate_base.data_handler import load_data
from microstate_analysis.eeg_tool.algorithm.clustering.microstate import exclude_zero_mean


def eegmaps_parameters(clean_data_fname, subjects, data_fname, eegmaps_fname, clean_data_fname_save, data_fname_save,
                       task_name):
    eegmaps = load_data(eegmaps_fname)
    eegmaps = eegmaps['maps'] if 'maps' in eegmaps else eegmaps
    for subject in subjects:
        print(subject)
        temp_res = []
        res = {}
        pool = Pool(len(task_name))
        data = load_data(clean_data_fname + "\\" + subject + data_fname)
        for a_task_name in task_name:
            temp_res.append(
                pool.apply_async(batch_microstate_parameters, ([data[a_task_name], eegmaps, 10, 3, False, 500, 2],)))
        pool.close()
        pool.join()
        for i, a_task_name in enumerate(task_name):
            res[a_task_name] = temp_res[i].get()
        json.dump(res, codecs.open(clean_data_fname_save + "\\" + subject + data_fname_save, 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)


def batch_microstate_parameters(para):
    data = para[0]
    maps = para[1]
    distance = para[2]
    n_std = para[3]
    polarity = para[4]
    sfreq = para[5]
    epoch = para[6]
    _ = Microstate(np.asarray(data))
    return Microstate.microstates_parameters(np.asarray(data), np.asarray(maps), distance, n_std, polarity, sfreq,
                                             epoch)


def eegmaps_parameters_across_runs(clean_data_fname, subjects, data_fname, data_fname_save, condition_name, tasks_name):
    for subject in subjects:
        print(subject)
        data = load_data(clean_data_fname + "\\" + subject + data_fname)
        res = {}
        for i, task_name in enumerate(tasks_name):
            duration = []
            coverage = []
            res[condition_name[i]] = {}
            for a_task_name in task_name:
                duration.extend(data[a_task_name]["duration"])
                coverage.extend(data[a_task_name]["coverage"])
            res[condition_name[i]]['duration'] = exclude_zero_mean(np.asarray(duration)).tolist()
            res[condition_name[i]]['coverage'] = np.asarray(coverage).mean(axis=0).tolist()
        json.dump(res, codecs.open(clean_data_fname + "\\" + subject + data_fname_save, 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)
