from sklearn.metrics import confusion_matrix
from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, write_append_info, load_config_design
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import itertools
import codecs, json

def parameters_confusion_matrix(matrix):
    row, col = matrix.shape[0], matrix.shape[1]
    sum_all = np.sum(matrix)
    res = []
    for i in range(row):
        tp = matrix[i][i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp
        tn = sum_all - tp - fp - fn
        sensitivity = tp / (tp + fn)
        specificity = tn / (fp + tn)
        if (tp+fp) == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        recall = sensitivity
        if precision == 0:
            f_measure = 0
        else:
            f_measure = 2 * (precision*recall) / (precision+recall)
        res.append([sensitivity, specificity, precision, f_measure])
    return res



if __name__ == '__main__':
    subjects, tasks, conditions, conditions_tasks = load_config_design()
    path_dir = {'100': [r'D:\EEGdata\reconstuction_sequences\\classification\100\seq_100_0.75_cl_cnn',
                r'D:\EEGdata\reconstuction_sequences\\classification\100\seq_100_0.75_cl_fnn',
                r'D:\EEGdata\reconstuction_sequences\\classification\100\seq_100_0.75_cl_ml',
                r'D:\EEGdata\reconstuction_sequences\\classification\100\seq_100_0.75_cl_ml',],
                '200': [r'D:\EEGdata\reconstuction_sequences\\classification\200\seq_200_0.75_cl_cnn',
                        r'D:\EEGdata\reconstuction_sequences\\classification\200\seq_200_0.75_cl_fnn',
                        r'D:\EEGdata\reconstuction_sequences\\classification\200\seq_200_0.75_cl_ml',
                        r'D:\EEGdata\reconstuction_sequences\\classification\200\seq_200_0.75_cl_ml', ],
                '400': [r'D:\EEGdata\reconstuction_sequences\\classification\400\seq_400_0.75_cl_cnn',
                        r'D:\EEGdata\reconstuction_sequences\\classification\400\seq_400_0.75_cl_fnn',
                        r'D:\EEGdata\reconstuction_sequences\\classification\400\seq_400_0.75_cl_ml',
                        r'D:\EEGdata\reconstuction_sequences\\classification\400\seq_400_0.75_cl_ml', ],
                '800': [r'D:\EEGdata\reconstuction_sequences\\classification\800\seq_800_0.75_cl_cnn',
                        r'D:\EEGdata\reconstuction_sequences\\classification\800\seq_800_0.75_cl_fnn',
                        r'D:\EEGdata\reconstuction_sequences\\classification\800\seq_800_0.75_cl_ml',
                        r'D:\EEGdata\reconstuction_sequences\\classification\800\seq_800_0.75_cl_ml', ],

                }
    output_test_key = ['output_test', 'output_test', 'output_test_nb', 'output_test_svm']
    res = {'100': {}, '200': {}, '400': {}, '800': {}}
    seqs = ['100', '200', '400', '800']
    methods = ['cnn', 'fnn', 'nb', 'svm']
    for seq in seqs:
        print(seq)
        for i, path_temp in enumerate(path_dir[seq]):
            print(path_temp)
            sub = []
            for sub_j, subject in enumerate(subjects):
                print(subject)
                path = path_temp + "\\" + subject + "\\labels.json"
                if i < 2:
                    label_test = list(itertools.chain(*load_data(path)[0]['label_test']))
                    output_test = list(itertools.chain(*load_data(path)[0][output_test_key[i]]))
                else:
                    label_test = load_data(path)[0]['label_test']
                    output_test = load_data(path)[0][output_test_key[i]]
                c_m = confusion_matrix(label_test, output_test)
                para = parameters_confusion_matrix(c_m)
                sub.append([para])
            res[seq][methods[i]] = sub
    json.dump(res, codecs.open(r'D:\EEGdata\reconstuction_sequences\classification\res\all.json', 'w', encoding='utf-8'),
              separators=(',', ':'), sort_keys=True, indent=4)
