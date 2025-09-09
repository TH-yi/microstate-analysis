from scipy.io import loadmat, savemat
import itertools
from eeg_tool.math_utilis import ceil_decimal
from statsmodels.stats.multitest import multipletests

if __name__ == '__main__':
    titles = ['condition_coverage_m', 'condition_duration_m', 'condition_frequency_m']
    conditions = ['REST', 'PU', 'IG', 'RIG', 'IE', 'RIE']
    path = r'D:\EEGdata\clean_data_six_problem\1_40\microstate\parameters\condition_m_t_test'
    alpha = 0.05
    for title in titles:
        index = 0
        print(title)
        for comb in itertools.combinations([i for i in range(len(conditions))], 2):
            item_str = conditions[comb[0]] + "&Vs.&" + conditions[comb[1]] + "&"
            for i in range(7):
                p_list = loadmat(path + "\\" + title + str(i) + ".mat")['EEG'].flatten()
                bonferroni_corrected = multipletests(p_list, alpha, 'bonferroni')
                p_list = bonferroni_corrected[1]
                t_list = loadmat(path + "\\" + title + 't' + str(i) + ".mat")['EEG'].flatten()
                c_list = loadmat(path + "\\" + title + 'c' + str(i) + ".mat")['EEG']
                p = ceil_decimal(round(p_list[index],4), 3)
                t = t_list[index]
                # c = c_list[index]
                if p > 0.05:
                    p_str = str(p) + "& & "
                elif 0.05 >= p > 0.01:
                    p_str = str(p) + '\\' + 'tnote{*} & '
                elif 0.01 >= p > 0.005:
                    p_str = str(p) + '\\' + 'tnote{**} & '
                else:
                    p_str = str(p) + '\\' + 'tnote{***} & '
                if p <= 0.05:
                    if t < 0:
                        arrow_str = r'$\nearrow$ &'
                    else:
                        arrow_str = r'$\searrow$ &'
                else:
                    arrow_str = ''
                item_str = item_str + p_str + arrow_str
            index += 1
            print(item_str)

