import os
import mne
from scipy.io import loadmat, savemat
from mne.preprocessing import ICA
import matplotlib.pyplot as plt


def load_tasks_name():
    name = [i+'_'+str(j) for i in ['pu', 'ig', 'rig', 'ie', 'rie'] for j in range(1, 7)]
    name.insert(0, 'rest_2')
    name.insert(0, 'rest_1')
    print(name)
    return name


def create_info(sfreq, montage_path):
    ch = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
    info = mne.create_info(ch_names=ch, sfreq=sfreq, ch_types='eeg', montage=montage_path)
    return info


def read_mat(data_path, montage_path, sfreq):
    info = create_info(sfreq, montage_path)
    return mne.io.RawArray(loadmat(data_path)['EEG'], info)


if __name__ == '__main__':
    tasks = load_tasks_name()
    # sub_path = r'D:\april_22'
    sub_path = r'D:\EEGdata\clean_data_six_problem\1_40\clean_data_mat\Feb_28(2)_2014'
    montage_path = r'D:\Cap63.locs'
    sfreq = 250
    n_components = 63
    for task in tasks:
        print(task)
        data_path = sub_path + "\\" + task + ".mat"
        raw = read_mat(data_path, montage_path, sfreq)
        # raw.plot()
        ica = ICA(n_components=n_components, random_state=97, method='infomax')
        ica = ica.fit(raw)
        # ica.plot_properties(raw, picks=[i for i in range(n_components)])
        ica.plot_sources(raw, picks=[i for i in range(n_components)])
        plt.show()
