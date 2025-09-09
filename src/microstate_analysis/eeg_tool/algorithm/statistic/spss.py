import numpy as np
def electrodes_spss():
    electrodes_all = []
    electrodes_index = []
    electrodes_origine = ["FP1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7",
                          "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2",
                          "P4", "P8", "TP10", "CP6", "CP2", "C4", "T8", "FT10", "FC6",
                          "FC2", "F4", "F8", "FP2", "AF7", "AF3", "AFz", "F1", "F5",
                          "FT7", "FC3", "FCz", "C1", "C5", "TP7", "CP3", "P1", "P5",
                          "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2", "CPz", "CP4", "TP8",
                          "C6", "C2", "FC4", "FT8", "F6", "F2", "AF4", "AF8"]
    electrodes_spss = ['FP1', 'AF7', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'T7', 'C5',
                       'C3',
                       'C1', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1', 'PO7', 'PO3', 'O1']
    for elec in electrodes_spss:
        for i in range(len(elec)):
            try:
                num = int(elec[i]) + 1
                temp = elec.replace(elec[i], str(num))
            except:
                pass
        electrodes_all.append(elec)
        electrodes_all.append(temp)
    for i in range(0, len(electrodes_all)):
        for j in range(0, len(electrodes_origine)):
            if electrodes_all[i] == electrodes_origine[j]:
                electrodes_index.append(j)
                break
    # temp = np.asarray(electrodes_origine)
    # print(temp[electrodes_index])
    return electrodes_index

if __name__ == '__main__':
    electrodes_spss()