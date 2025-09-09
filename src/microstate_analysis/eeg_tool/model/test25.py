from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, write_append_info, load_config_design, read_info
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, ttest_rel

if __name__ == '__main__':
    data = np.asarray(read_info(r'C:\Users\Zeng\Desktop\design_problem\behaviour.xlsx','data'))
    n_items = 6*5*2
    ig = []
    ie = []
    for i in range(0, n_items, 10):
        ig.append(data[:, i:i+5])
    for i in range(5, n_items, 10):
        ie.append(data[:, i:i + 5])
    ig = np.stack(ig)
    ie = np.stack(ie)
    ig_mean = ig.mean(axis=0)
    ie_mean = ie.mean(axis=0)
    for i in range(5):
        p = ttest_rel(ig_mean[:,i], ie_mean[:,i])
        print(i, p[0], p[1])