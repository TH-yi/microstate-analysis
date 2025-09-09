import numpy as np


def diss(v_ij, v_j):
    """
    计算不相似度。

    参数:
    - v_ij: 输入向量，形状为(n_samples, n_channels)。
    - v_j: 输入向量，形状为(n_channels,)。

    返回:
    - 不相似度。
    """
    return np.sum(np.sqrt(np.mean(np.power(v_ij - v_j, 2), 1)))


def diss_interaction(level_mean, grand_mean):
    """
    计算交互不相似度。

    参数:
    - level_mean: 水平均值，形状为(n_samples, n_channels)。
    - grand_mean: 总体均值，形状为(n_channels,)。

    返回:
    - 交互不相似度。
    """
    residual = np.power(level_mean - grand_mean, 2)
    return np.sum(np.sqrt(np.mean(residual, 2)))


def generalized_dissimilarity(data):
    """
    计算广义不相似度。

    参数:
    - data: 输入数据，形状为(n_samples, n_channels)。

    返回:
    - 广义不相似度。
    """
    grand_mean_across_subjects = np.mean(data, axis=0)
    grand_mean_across_subjects_across_conditions = np.mean(data, axis=(0, 1))

    residual = np.power(grand_mean_across_subjects - grand_mean_across_subjects_across_conditions, 2)
    s = np.sum(np.sqrt(np.mean(residual, axis=1)))
    return s
