import json
import numpy as np
from scipy import stats
import codecs, json
import itertools
from operator import itemgetter
from scipy.optimize import linear_sum_assignment


def eegmaps_similarity_old(baseline, task, threshold=0.0):
    """
    计算两个EEG图之间的相似性。

    参数:
    - baseline: 基线EEG图，形状为(n_maps, n_channels)。
    - task: 任务EEG图，形状为(n_maps, n_channels)。
    - threshold: 相似性阈值，默认值为0.0。

    返回:
    - res: 相似性字典。
    """
    res = {}
    similarity = np.zeros((baseline.shape[0], task.shape[0]))
    for i in range(baseline.shape[0]):
        for j in range(task.shape[0]):
            similarity[i][j] = abs(stats.pearsonr(baseline[i, :], task[j, :])[0])
    iter = min(baseline.shape[0], task.shape[0])
    while iter != 0:
        max_similarity = np.max(similarity)
        if max_similarity < threshold:
            iter -= 1
            continue
        temp = np.argwhere(similarity == max_similarity)
        row = temp[0][0]
        col = temp[0][1]
        res[col] = {'index': row, 'similarity': max_similarity}
        iter -= 1
        similarity[row, ::] = -3
        similarity[::, col] = -3
    if len(res) != task.shape[0]:
        task_index = [*res]
        diff = list(set([i for i in range(task.shape[0])]) - set(task_index))
        temp = baseline.shape[0]
        for i in diff:
            res[i] = {'index': temp, 'similarity': -3}
            temp += 1
    return res

def eegmaps_averaged_similarity(similarity):
    """
    计算EEG图的平均相似性。

    参数:
    - similarity: 相似性字典。

    返回:
    - 平均相似性得分。
    """
    res = 0
    for key, value in similarity.items():
        res += value['similarity']
    return res / len(similarity)

def eegmaps_similarity_across_runs(clean_data_fname, subjects, data_fname, task_name, opt_k_index):
    """
    计算跨运行的EEG图相似性。

    参数:
    - clean_data_fname: 干净数据的文件路径。
    - subjects: 受试者列表。
    - data_fname: 数据文件名。
    - task_name: 任务名称列表。
    - opt_k_index: 最佳k值索引。

    返回:
    - res: 跨运行的EEG图相似性。
    """
    res = []
    for subject in subjects:
        data_text = codecs.open(clean_data_fname + "\\" + subject + data_fname, 'r', encoding='utf-8').read()
        data = json.loads(data_text)
        res_temp = []
        for a_task_name in task_name:
            for comb in itertools.combinations([i for i in range(1, len(task_name) + 1)], 2):
                similarity = eegmaps_similarity(
                    np.asarray(data[str(comb[0]) + "_" + a_task_name]["maps_list"][opt_k_index]),
                    np.asarray(data[str(comb[1]) + "_" + a_task_name]["maps_list"][opt_k_index]), 0.0)
                averaged_similarity = eegmaps_averaged_similarity(similarity)
                res_temp.append(averaged_similarity)
        res.append(res_temp)
    return res


def eegmaps_similarity_highest_comb(data):
    """
    计算EEG图相似性最高的组合。

    参数:
    - data: 输入数据，列表形式。

    返回:
    - res: 相似性最高的组合。
    """
    data = sorted(data, key=itemgetter(1), reverse=True)
    res = []
    for item in data:
        for item_res in res:
            if len(set(item[0]).intersection(set(item_res[0]))) != 0:
                break
        else:
            res.append(item)
    return res


def eegmaps_similarity_highest_merged(comb, eegmaps, n_runs, opt_k):
    """
    合并相似性最高的EEG图。

    参数:
    - comb: 组合列表。
    - eegmaps: EEG映射，列表形式。
    - n_runs: 运行次数。
    - opt_k: 最佳k值。

    返回:
    - res_eegmaps: 合并后的EEG图。
    """
    res_eegmaps = []
    for item in comb:
        res = []
        for i in range(len(item[0])):
            temp = 0
            for j in range(len(item[0])):
                temp += abs(stats.pearsonr(eegmaps[i][item[0][i] % n_runs], eegmaps[j][item[0][j] % n_runs])[0])
            res.append(temp / len(item[0]))
        index = np.argmax(np.asarray(res))
        res_eegmaps.append(eegmaps[task_index_runs(item[0][index], opt_k) - 1][item[0][index] % opt_k].tolist())
    return res_eegmaps

# Subfunction: Calculate the Pearson correlation coefficient between two vectors (absolute value, polarity-independent)
def pearson_similarity(vec1, vec2):
    return abs(stats.pearsonr(vec1, vec2)[0])

def eegmaps_similarity_hungarian(baseline, task, threshold=0.0):
    """
    使用匈牙利算法计算两个EEG图之间的全局相似性匹配。

    参数:
    - baseline: 基线EEG图，形状为(n_maps, n_channels)。
    - task: 任务EEG图，形状为(n_maps, n_channels)。
    - threshold: 相似性阈值，默认值为0.0。

    返回:
    - res: 匹配结果字典，包含任务图的索引与对应的基线图索引及相似性。
    """
    # 初始化相似度矩阵
    n_baseline, n_task = baseline.shape[0], task.shape[0]
    max_maps = max(n_baseline, n_task)

    # 创建方阵，填充为负无穷（相似性越大，成本越小）
    similarity_matrix = np.full((max_maps, max_maps), -np.inf)

    # 计算皮尔逊相关系数的绝对值
    for i in range(n_baseline):
        for j in range(n_task):
            similarity = abs(stats.pearsonr(baseline[i, :], task[j, :])[0])
            if similarity >= threshold:
                similarity_matrix[i, j] = -similarity  # 匈牙利算法求最小值，取负数

    # 解决方阵中无法匹配的情况，用0填充空位
    similarity_matrix[similarity_matrix == -np.inf] = 0

    # 使用匈牙利算法找到最优匹配
    row_ind, col_ind = linear_sum_assignment(similarity_matrix)

    # 构建结果字典
    res = {}
    for task_idx, baseline_idx in zip(col_ind, row_ind):
        similarity_value = -similarity_matrix[baseline_idx, task_idx]  # 恢复为正值
        if baseline_idx < n_baseline and task_idx < n_task:
            res[task_idx] = {'index': baseline_idx, 'similarity': similarity_value}
        else:
            # 如果任务图或基线图超出实际范围，标记为未匹配
            res[task_idx] = {'index': -1, 'similarity': -1}

    return res