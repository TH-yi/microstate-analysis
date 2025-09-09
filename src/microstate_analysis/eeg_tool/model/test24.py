from eeg_tool.utilis import get_root_path, read_subject_info, load_data, write_info, read_xlsx, write_append_info, load_config_design
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

if __name__ == '__main__':
    subjects, tasks, conditions, conditions_tasks = load_config_design()
    conditions_name = ['REST', 'PU', 'IG', 'RIG', 'IE', 'RIE']
    path_dir = r'D:\EEGdata\reconstuction_sequences\single_reconstruction\gru\seq_800_0.5'
    # n_epochs = 25
    # n_epochs_save = 10
    str_temp = ''
    valid_accuracy_all = []
    for index, condition in enumerate(conditions):
        valid_accuracy = []
        train_accuracy = []
        for index, subject in enumerate(subjects):
                # print(index)
            # if index < 23:
                train_path = path_dir + "\\" + subject + "\\" + condition +"_train_epochs.json"
                valid_path = path_dir + "\\" + subject + "\\" + condition +"_valid_epochs.json"
                sub_valid_accuracy = load_data(valid_path)[0]['valid_accuracy']
                # sub_train_accuracy = [item['train_accuracy'] for item in load_data(train_path)]
                valid_accuracy.append(sub_valid_accuracy)
                valid_accuracy_all.append(sub_valid_accuracy)
                # train_accuracy.append(sub_train_accuracy)
        print(condition, np.mean(valid_accuracy).round(2), sem(valid_accuracy).round(2))
        str_temp += str(format(np.mean(valid_accuracy) * 100, '.2f')) + '$\pm$' + str(format(sem(valid_accuracy) * 100, '.2f')) + " & "
    str_temp += str(format(np.mean(valid_accuracy_all) * 100, '.2f')) + '$\pm$' + str(format(sem(valid_accuracy_all) * 100, '.2f')) +r'\\'
    print(str_temp)



        # plt.errorbar(index, np.mean(valid_accuracy), sem(valid_accuracy), ecolor='black', elinewidth=2, fmt='ro')
    # plt.xticks([i for i in range(len(conditions))], conditions)
    #     plt.errorbar([i for i in range(n_epochs)], np.mean(np.asarray(train_accuracy), axis=0), sem(np.asarray(train_accuracy), axis=0), label=conditions_name[index])
    # plt.xticks([i for i in range(n_epochs)], [(i+1)*n_epochs_save for i in range(n_epochs)])

    # plt.xlabel('Design Activities')

    # plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Valid Accuracy')
    # plt.show()
