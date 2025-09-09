import numpy as np
from scipy.stats import sem
import itertools
import codecs, json
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, write_append_info, load_config_design
from scipy.stats import sem

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import mpl_toolkits.axisartist as axisartist

if __name__ == '__main__':
    data = load_data(r'D:\EEGdata\reconstuction_sequences\classification\res\all.json')
    seqs = ['100', '200', '400', '800']
    methods = ['nb', 'svm', 'fnn', 'cnn']
    methods_title = ['NB','SVM','FNN','CNN']
    # 0: sensitivity, 1:specificity, 2:precision, 3:F-measure
    para = 3
    for seq in seqs:
        for i, method in enumerate(methods):
            str_res = '& ' + methods_title[i] + '&'
            res_con = []
            for c_i in range(6):
                res = []
                for sub_j in range(27):
                    res.append(data[seq][method][sub_j][0][c_i][para])
                    res_con.append(data[seq][method][sub_j][0][c_i][para])
                mean = np.mean(res)
                se = sem(res)
                str_res = str_res + str(format(mean*100, '.2f')) + '$\pm$' + str(format(se*100, '.2f')) + " & "
            str_res = str_res + str(format(np.mean(res_con)*100, '.2f')) + '$\pm$' + str(format(sem(res_con)*100, '.2f')) + r'\\'
            print(str_res)

