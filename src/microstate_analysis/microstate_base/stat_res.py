from collections import OrderedDict
import codecs
import numpy as np
import json

def microstate_stat(clean_data_fname, subjects, data_fname, data_fname_save, task_name):
    """
    计算微状态的统计信息。

    参数:
    - clean_data_fname: 干净数据的文件路径。
    - subjects: 受试者列表。
    - data_fname: 数据文件名。
    - data_fname_save: 保存统计信息的文件名。
    - task_name: 任务名称列表。

    返回:
    - res: 微状态的统计信息，字典形式。
    """
    res = OrderedDict()
    for subject in subjects:
        data_text = codecs.open(clean_data_fname + "\\" + subject + data_fname, 'r', encoding='utf-8').read()
        data = json.loads(data_text)
        for a_task_name in task_name:
            if a_task_name not in res:
                res[a_task_name] = {'opt_k': [], 'min_maps': data[a_task_name]["min_maps"],
                                    'max_maps': data[a_task_name]["max_maps"]}
            res[a_task_name]['opt_k'].append(data[a_task_name]["opt_k"])
    for a_task_name in task_name:
        res[a_task_name]['mean_opt_k'] = int(np.asarray(res[a_task_name]['opt_k']).mean())
        res[a_task_name]['mean_opt_k_index'] = res[a_task_name]['mean_opt_k'] - int(res[a_task_name]['min_maps'])
    json.dump(res, codecs.open(clean_data_fname + "\\" + data_fname_save, 'w', encoding='utf-8'), separators=(',', ':'),
              sort_keys=True, indent=4)
    return res