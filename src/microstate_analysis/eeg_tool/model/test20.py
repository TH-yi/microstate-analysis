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
import numpy as np

if __name__ == '__main__':
    data = np.asarray(read_xlsx(r'C:\Users\Zeng\Desktop\design_problem\v2\maps\nmaps.xlsx','nmaps'))
    temp = data[0:2, :].flatten()
    print(round(np.mean(temp),3), round(st.sem(temp),3))
    for i in range(2, len(data), 6):
        temp = data[i:i+6, :].flatten()
        print(round(np.mean(temp),3), round(st.sem(temp),3))