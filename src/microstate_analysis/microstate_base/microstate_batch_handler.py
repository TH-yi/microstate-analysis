import numpy as np

from microstate_analysis.microstate_base.microstate import Microstate
from microstate_analysis.microstate_base.meanmicrostate import MeanMicrostate

def batch_microstate_state(data, topographies, task_name):
    for a_task_name in task_name:
        microstate = Microstate(data[a_task_name])
        microstate.gfp_peaks()
        microstate.maps = np.asarray(topographies[a_task_name]["maps_list"][topographies[a_task_name]["opt_k_index"]])
        microstate.microstate_state()


def batch_order_mean_microstate(para):
    data = para[0]
    n_k = para[1]
    n_ch = para[2]
    n_condition = para[3]
    mean_microstate = para[4]
    microstate = MeanMicrostate(data, n_k, n_ch, n_condition)
    res = microstate.reorder_microstates(microstate.data, mean_microstate, polarity=False)
    return res


def batch_mean_microstate(para):
    data = para[0]
    n_k = para[1]
    n_ch = para[2]
    n_condition = para[3]
    microstates = MeanMicrostate(data, n_k, n_ch, n_condition)
    eegmaps, label, mean_similarity, std_similarity = microstates.mean_microstates()
    return {"maps": eegmaps, "label": label, "mean_similarity": mean_similarity, "std_similarity": std_similarity}


def batch_microstate(para):
    data = para[0]
    peaks_only = para[1]
    min_maps = para[2]
    max_maps = para[3]
    opt_k = para[4] if len(para) > 4 else None
    method = para[5] if len(para) > 5 else 'kmeans_modified'
    n_std = para[6] if len(para) > 6 else 3
    n_runs = para[7] if len(para) > 7 else 100
    microstate = Microstate(data)
    microstate.opt_microstate(min_maps, max_maps, n_std=n_std, n_runs=n_runs, peaks_only=peaks_only, method=method,
                              opt_k=opt_k)
    return microstate
