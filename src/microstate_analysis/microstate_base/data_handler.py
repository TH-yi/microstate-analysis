import mne
import os
import json
import codecs
import numpy as np
from collections import OrderedDict


def create_info():
    """
    创建EEG信息对象并导出到.locs文件。

    参数:
    - output_path: 输出文件的路径，默认路径为当前目录下的'eeg_tool/cap63.locs'。

    返回:
    - info: 包含EEG通道信息的对象。
    """
    ch = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz',
          'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
          'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
          'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']


    locs_path = os.path.abspath(os.path.join('..', '..', 'eeg_code', 'eeg_tool', 'cap63.locs'))
    #montage = mne.channels.read_custom_montage(locs_path)
    # 创建info对象
    info = mne.create_info(ch_names=ch, sfreq=500, ch_types='eeg', montage=locs_path)

    return info


def exclude_zero_mean(data):
    """
    排除均值为零的数据。

    参数:
    - data: 输入数据，形状为(n_samples, n_channels)。

    返回:
    - 平均值不为零的数据。
    """
    sum = np.sum(data, axis=0)
    col = (data != 0).sum(0)
    return sum / col


def load_data(path):
    """
    加载数据。

    参数:
    - path: 数据路径。

    返回:
    - data: 加载的数据。
    """
    data_text = codecs.open(path, 'r', encoding='utf-8').read()
    data = json.loads(data_text)
    return data


def read_microstate_stat(data_fname):
    """
    读取微状态统计信息。

    参数:
    - data_fname: 数据文件名。

    返回:
    - res: 微状态统计信息，字典形式。
    """
    res = {}
    data_text = codecs.open(data_fname, 'r', encoding='utf-8').read()
    data = json.loads(data_text)
    for task_name, task_data in data.items():
        res[task_name] = {'mean_opt_k': task_data['mean_opt_k']}
    return res


def normalized_data(data):
    """
    标准化数据。

    参数:
    - data: 输入数据，形状为(n_samples, n_channels)。

    返回:
    - 标准化后的数据，形状与输入相同。
    """
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = data[i][j] - np.mean(data[i][j])
    return data


def list_to_matrix(data_list):
    """
    将一个包含 n 个长度为 m 列表的列表转换为一个形状为 (m, n) 的 numpy 矩阵。

    参数:
    data_list: 一个包含 n 个长度为 m 列表的列表

    返回:
    matrix_transposed: 形状为 (m, n) 的 numpy 矩阵
    """
    # 检查输入数据是否为空
    if not data_list:
        raise ValueError("输入列表为空")

    # 将列表转换为 numpy 数组
    matrix = np.array(data_list)

    # 检查每个子列表是否具有相同长度
    if not all(len(row) == len(data_list[0]) for row in data_list):
        raise ValueError("输入列表中的子列表长度不一致")

    # 转置矩阵，返回 (m, n) 的结果
    # matrix_transposed = matrix.T

    return matrix

def format_data(data, condition, n_subject, n_ch, ith_class):
    """
    格式化数据。

    参数:
    - data: 输入数据，字典形式。
    - condition: 条件列表。
    - n_subject: 受试者数量。
    - n_ch: 通道数量。
    - ith_class: 第i类。

    返回:
    - data_temp: 格式化后的数据，形状为(n_subject, n_condition, n_ch)。
    """
    n_condition = len(condition)
    data_temp = np.zeros((n_subject, n_condition, n_ch))
    for i in range(n_subject):
        for j in range(n_condition):
            data_temp[i, j] = data[condition[j]][i][ith_class]
    return data_temp


def shuffle_data(data, n_condition):
    """
    随机打乱数据。

    参数:
    - data: 输入数据，形状为(n_samples, n_channels)。
    - n_condition: 条件数量。

    返回:
    - 打乱后的数据，形状与输入相同。
    """
    for i in range(data.shape[0]):
        random_index = np.random.permutation(n_condition)
        data[i] = data[i][random_index]
    return data


def combine_task_data(data, combined_task):
    """
    组合任务数据。

    参数:
    - data: 输入数据，字典形式。
    - combined_task: 组合任务列表。

    返回:
    - data_temp: 组合后的数据，字典形式。
    """
    data_temp = OrderedDict()
    for task in combined_task:
        a_task = task.split("#")
        if len(a_task) > 1:
            temp = np.asarray(data[a_task[0]])
            for i in range(1, len(a_task)):
                temp = np.concatenate((temp, np.asarray(data[a_task[i]])), axis=1)
            data_temp[task] = temp.tolist()
        else:
            data_temp[task] = data[task]
    return data_temp

def task_index_runs(index, opt_k):
    """
    根据索引返回运行任务的索引。

    参数:
    - index: 索引。
    - opt_k: 最佳k值。

    返回:
    - 运行任务的索引。
    """
    if index < opt_k:
        return 1
    elif opt_k <= index < 2 * opt_k:
        return 2
    elif 2 * opt_k <= index < 3 * opt_k:
        return 3


def concatenate_data_by_condition(data, task_condition, exclude_task):
    """
    根据条件连接数据。

    参数:
    - data: 输入数据，字典形式。
    - task_condition: 任务条件列表。
    - exclude_task: 要排除的任务列表。

    返回:
    - res: 连接后的数据，字典形式。
    """
    res = OrderedDict()
    for condition in task_condition:
        for task_name, task_data in data.items():
            if task_name not in exclude_task:
                if condition == task_name.split("_")[1]:
                    res[condition] = np.asarray(task_data) if condition not in res else np.concatenate(
                        (res[condition], np.asarray(task_data)), axis=1)
    return res
