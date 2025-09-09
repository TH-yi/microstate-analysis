import matplotlib.pyplot as plt
import numpy as np
from eeg_tool.utilis import load_data, read_subject_info, write_info, read_xlsx, to_string

def plot_psd(f, pxx, task, subject):
    plt.semilogy(f, pxx)
    plt.show()
    plt.title(subject+"_"+task)


if __name__ == '__main__':
    read_path_json = r'D:\EEGdata\clean_data_six_problem\1_40\psd\psd'
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'all')
    for subject in subjects:
        data = load_data(read_path_json + "\\" + subject +'_psd.json')
        for task in tasks:
            f = np.asarray(data[task]['f'])[0, 0:80]
            pxx = np.asarray(data[task]['pxx'])[0, 0:80]
            plot_psd(f, pxx, task, subject)