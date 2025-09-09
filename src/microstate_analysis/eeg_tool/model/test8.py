import numpy as np
from eeg_tool.utilis import read_xlsx, write_info
def ch_locs_1():
    ch = ['Fp1', 'Fp2', 'AF7', 'AF8', 'AF3', 'AF4', 'F7', 'F8',
          'F5', 'F6', 'F3', 'F4', 'F1', 'F2', 'FT9', 'FT10',
          'FT7', 'FT8', 'FC5', 'FC6', 'FC3', 'FC4', 'FC1',
          'FC2', 'T7', 'T8', 'C5', 'C6', 'C3', 'C4', 'C1',
          'C2', 'TP9', 'TP10', 'TP7', 'TP8', 'CP5', 'CP6',
          'CP3', 'CP4', 'CP1', 'CP2', 'P7', 'P8', 'P5',
          'P6', 'P3', 'P4', 'P1', 'P2', 'PO7', 'PO8',
          'PO3', 'PO4', 'O1', 'O2']
    return ch

def ch_locs_2():
    frontal_left = ['Fp1', 'AF3', 'AF7', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3']
    frontal_right = ['Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4']
    central_left = ['FC5', 'C1', 'C3', 'C5']
    central_right = ['FC4', 'C2', 'C4', 'C6']
    temporal_left = ['FT7', 'T7', 'TP7', 'CP5', 'P5']
    temporal_right = ['FT8', 'T8', 'TP8', 'CP6', 'P6']
    parietal_left = ['CP1', 'CP3', 'P1', 'P3']
    parietal_right = ['CP2', 'CP4', 'P2', 'P4']
    occipital_left = ['PO3', 'PO7', 'P7', 'O1']
    occipital_right = ['PO4', 'PO8', 'P8', 'O2']
    ch = [['Fp1', 'AF3', 'AF7', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3'],
          ['Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4'],
          ['FC5', 'C1', 'C3', 'C5'],
          ['FC4', 'C2', 'C4', 'C6'],
          ['FT7', 'T7', 'TP7', 'CP5', 'P5'],
          ['FT8', 'T8', 'TP8', 'CP6', 'P6'],
          ['CP1', 'CP3', 'P1', 'P3'],
          ['CP2', 'CP4', 'P2', 'P4'],
          ['PO3', 'PO7', 'P7', 'O1'],
          ['PO4', 'PO8', 'P8', 'O2']]
    return ch

def ch_index(ch_1, ch_2):
    res = []
    for lobes in ch_2:
        temp = []
        for lobe in lobes:
            for index, ch in enumerate(ch_1):
                if lobe == ch:
                    temp.append(index)
        res.append(temp)
    return res

if __name__ == '__main__':
    data = np.asarray(read_xlsx(r'D:\EEGdata\clean_data_creativity\clean_subject_28\trp_28_8_10_10_12hz.xlsx', '10_12'))
    block = 56
    n_block = 3
    lobes = ch_index(ch_locs_1(), ch_locs_2())
    n_lobes = len(lobes)
    temp_data = np.zeros((data.shape[0], n_lobes*n_block))
    for n in range(n_block):
        for index, ch in enumerate(lobes):
            temp = np.asarray(ch) + n * block
            temp_data[:, index + n * n_lobes ] = data[:, temp].mean(axis=1)
    write_info(r'D:\EEGdata\clean_data_creativity\clean_subject_28\trp_28_lobes.xlsx', '10_12', temp_data)