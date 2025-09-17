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


def batch_microstate_parameters_selective(para):
    """
    Batch wrapper for selective microstate metrics, designed for multiprocessing pools.

    Accepts either:
      - a tuple/list in the legacy positional format, with optional extras:
            [data, maps, distance, n_std, polarity, sfreq, epoch,
             parameters=None, include_duration_seconds=False, log_base=math.e, states=None]
        (extras are optional; missing ones fall back to defaults)
      - OR a dict with explicit keys:
            {
              "data": ...,
              "maps": ...,
              "distance": 10,
              "n_std": 3,
              "polarity": False,
              "sfreq": 500,
              "epoch": 2.0,
              "parameters": {"coverage", "duration_seconds", "transition_frequency"},
              "include_duration_seconds": True,
              "log_base": math.e,
              "states": [0,1,2,3]  # optional explicit order
            }

    Returns:
        Dict[str, List[Any]]: one entry per requested metric; each value is a list aligned by epoch.
    """
    import math
    import numpy as np

    # -------- unpack inputs (support tuple/list or dict) --------
    if isinstance(para, dict):
        data = para.get("data")
        maps = para.get("maps")
        distance = para.get("distance", 10)
        n_std = para.get("n_std", 3)
        polarity = para.get("polarity", False)
        sfreq = para.get("sfreq", 500)
        epoch = para.get("epoch", 2.0)
        parameters = para.get("parameters", None)  # e.g., {"coverage","duration_seconds","entropy_rate"}
        include_duration_seconds = para.get("include_duration_seconds", False)
        log_base = para.get("log_base", math.e)
        states = para.get("states", None)
    else:
        # positional: keep full backward-compat with the first 7 fields
        # extras (optional): parameters, include_duration_seconds, log_base, states
        data = para[0]
        maps = para[1]
        distance = para[2]
        n_std = para[3]
        polarity = para[4]
        sfreq = para[5]
        epoch = para[6]

        parameters = None
        include_duration_seconds = False
        log_base = math.e
        states = None

        if len(para) >= 8:
            parameters = para[7]
        if len(para) >= 9:
            include_duration_seconds = para[8]
        if len(para) >= 10:
            log_base = para[9]
        if len(para) >= 11:
            states = para[10]

    # -------- ensure numpy arrays where appropriate --------
    data = np.asarray(data)
    maps = np.asarray(maps)

    # instantiate to keep consistency with your existing pattern (even if unused directly)
    _ = Microstate(data)

    # -------- compute selective metrics --------
    return Microstate.microstates_parameters_selective(
        data=data,
        maps=maps,
        distance=distance,
        n_std=n_std,
        polarity=polarity,
        sfreq=sfreq,
        epoch=epoch,
        parameters=parameters,
        log_base=log_base,
        states=states,
        include_duration_seconds=include_duration_seconds,
    )
