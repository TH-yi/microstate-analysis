from eeg_tool.utilis import load_data, read_subject_info, write_info, read_xlsx, to_string
from multiprocessing import Pool
import codecs, json
from scipy.stats import sem
import numpy as np

subjects = read_subject_info(r'C:\Users\Zeng\Desktop\six-problem\subjects.xlsx', 'subjects')
tasks = read_subject_info(r'C:\Users\Zeng\Desktop\six-problem\task_name_combined.xlsx', 'pu_ig_ie')
conditions = read_subject_info(r'C:\Users\Zeng\Desktop\six-problem\condition_name.xlsx', 'pu_ig_ie')
channels = r'C:\Users\Zeng\Desktop\six-problem\clean_data_six_problem.xlsx'

def bad_epochs(path):
    res = {}
    for condition in conditions:
        res[condition] = []
        for subject in subjects:
            data = read_xlsx(path, subject)
            for task in data:
                if task and condition in task[0]:
                    print(task[0])
                    num = 0
                    for a_task in task:
                        if '1' in a_task.split('_'):
                            num += 1
                    res[condition].append(num)
        print(condition, np.mean(res[condition]), sem(res[condition]))

def bad_global_channels(path):
    res = []
    for subject in subjects:
        data = read_xlsx(path, subject)
        res.append(len(data[0][1].split("_")))
    print(np.mean(res), sem(res))


if __name__ == '__main__':
    subjects = read_subject_info(r'C:\Users\Zeng\Desktop\six-problem\subjects.xlsx', 'subjects')
    tasks = read_subject_info(r'C:\Users\Zeng\Desktop\six-problem\task_name_combined.xlsx', 'pu_ig_ie')
    conditions = read_subject_info(r'C:\Users\Zeng\Desktop\six-problem\condition_name.xlsx', 'pu_ig_ie')
    res = {}
    for condition in conditions:
        res[condition] = []
        for subject in subjects:
            data = read_xlsx(r'C:\Users\Zeng\Desktop\six-problem\clean_data_six_problem.xlsx', subject)
            for task in data:
                if task and condition in task[0]:
                    print(task[0])
                    num = 0
                    for a_task in task:
                        if '1' in a_task.split('_'):
                            num += 1
                    res[condition].append(num)
        print(condition, np.mean(res[condition]), sem(res[condition]))
        # global bad channel
        # res.append(len(data[0][1].split("_")))
    # print(np.mean(res),sem(res))