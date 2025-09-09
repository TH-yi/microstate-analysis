from eeg_tool.utilis import *

if __name__ == '__main__':
    subjects = read_subject_info(r'D:\EEGdata\clean_data_six_problem\subjects.xlsx', 'subjects_1_40')
    tasks = read_subject_info(r'D:\EEGdata\clean_data_six_problem\task_name_combined.xlsx', 'm_all_ordered')
    path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\individual_run'
    nmaps_list = []
    cv_list = []
    for subject in subjects:
        data = load_data(path + "\\" + subject +"_microstate.json")
        temp_maps = []
        temp_cv = []
        for task in tasks:
            index = np.argmin(data[task]['cv'])
            temp_maps.append(index+1)
            temp_cv.append(data[task]['cv'][index])
        nmaps_list.append(temp_maps)
        cv_list.append(temp_cv)
    write_info(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\nmaps.xlsx','nmaps', nmaps_list)
    # write_info(r'D:\EEGdata\clean_data_six_problem\1_40\microstate\nmaps.xlsx', 'cv', cv_list)

