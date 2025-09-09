import math
import numpy as np
from numpy import argmax


def zero_mean(data, axis=0):
    mean = data.mean(axis=axis, keepdims=True)
    return data - mean


def square_to_condensed(i, j, n):
    if i < j:
        i, j = j, i
    return int(n * j - j * (j + 1) / 2 + i - 1 - j)


def calc_row_idx(k, n):
    return int(math.ceil((1 / 2.) * (- (-8 * k + 4 * n ** 2 - 4 * n - 7) ** 0.5 + 2 * n - 1) - 1))


def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i * (i + 1)) / 2


def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)


def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


def sigmoidal_overlap_smooth(tau, n_overlap):
    x = np.asarray([i for i in range(n_overlap)])
    smooth = 1. / (1 + np.exp(-tau * (x - n_overlap / 2) / n_overlap))
    return smooth


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def bandpower(psds, freqs, fmin, fmax, dx=1.0):
    ind_min = argmax(freqs > fmin) - 1
    ind_max = argmax(freqs > fmax) - 1
    return np.trapzoid(psds[:, ind_min: ind_max], freqs[ind_min: ind_max], dx=dx)


def scaling_psd(psds, scaling, dB=False):
    psds *= scaling * scaling
    if dB:
        np.log10(np.maximum(psds, np.finfo(float).tiny), out=psds)
    return psds


def ceil_decimal(num, n):
    return math.ceil(num * 10 ** n) / (10 ** n)
