import numpy as np
from multiprocessing import Pool
import codecs, json
from collections import OrderedDict
from scipy import signal
import os
import mne
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, create_info, to_string
from scipy.io import loadmat, savemat
from scipy.stats import sem
from scipy import trapz, argmax
from itertools import combinations
import matplotlib.pyplot as plt
import itertools
import scipy.stats as st

if __name__ == '__main__':
    title = ['PU', 'IG', 'RIG', 'IE', 'RIE']
    data = read_xlsx(r'C:\Users\Zeng\Desktop\design_problem\v2\trp\multicomparison.xlsx', 'theta_c_a_simple')
    block = [0, 1, 2, 3, 5, 6, 7, 10, 11, 15]
    block_index = []
    block_step = 5
    for i in block:
        for j in range(block_step):
            block_index.append(i + j*20)
    for index, item in enumerate(data):
        if isinstance(item[0], str):
            temp = item[0].replace('*','')
            data[index][0] = temp
    np_data = np.asarray(data)[block_index, :]
    block = 0
    for comb in itertools.combinations([i for i in range(len(title))], 2):
        item_str = title[comb[0]] + "&Vs.&" + title[comb[1]] + "&"
        for i in range(block_step):
            arrow_str = ''
            p = float(np_data[i+block, 1])
            p = round(p, 3)
            if p > 0.05:
                p_str = str(p) + "& & "
            elif 0.05 >= p > 0.01:
                p_str = str(p) + '\\' + 'tnote{*} & '
            elif 0.01 >= p > 0.005:
                p_str = str(p) + '\\' + 'tnote{**} & '
            else:
                p_str = str(p) + '\\' + 'tnote{***} & '
            if p <= 0.05:
                if float(np_data[i+block, 0]) < 0:
                    arrow_str = r'$\nearrow$ &'
                else:
                    arrow_str = r'$\searrow$ &'
            #     low = round(float(np_data[i+block, 2]), 3)
            #     high = round(float(np_data[i+block, 3]),3)
            #     arrow_str = r'95% C=[' + str(low) + "," + str(high) + "]"

            item_str = item_str + p_str + arrow_str
        print(item_str)
        block += block_step

