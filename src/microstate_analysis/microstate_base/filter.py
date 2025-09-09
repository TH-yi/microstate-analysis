import mne
import json
import codecs
from collections import OrderedDict
from multiprocessing import Pool


def filter_data(data):
    """
    对数据进行滤波。

    参数:
    - data: 输入数据，形状为(n_samples, n_channels)。

    返回:
    - 滤波后的数据，形状与输入相同。
    """
    return mne.filter.filter_data(data, 500, 1., 30.)


def batch_filter(clean_data_fname, subjects, data_fname, data_fname_save_dict, data_fname_save, task_name):
    """
    批量对数据进行滤波处理。

    参数:
    - clean_data_fname: 干净数据的文件路径。
    - subjects: 受试者列表。
    - data_fname: 数据文件名。
    - data_fname_save_dict: 保存滤波后数据的目录。
    - data_fname_save: 保存滤波后数据的文件名。
    - task_name: 任务名称列表。

    返回:
    - 无返回值，但会对每个受试者的每个任务进行滤波并保存结果。
    """
    for subject in subjects:
        print(subject)
        res = OrderedDict()
        data_text = codecs.open(clean_data_fname + "\\" + subject + data_fname, 'r', encoding='utf-8').read()
        data = json.loads(data_text)
        pool = Pool(8)
        multi_res = [pool.apply_async(filter_data, (data[a_task_name]['task_data'],)) for a_task_name in task_name]
        pool.close()
        pool.join()
        for i in range(len(task_name)):
            res[task_name[i]] = multi_res[i].get().tolist()
        json.dump(res, codecs.open(data_fname_save_dict + "\\" + subject + data_fname_save, 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)
