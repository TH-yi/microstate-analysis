import mne
import os

class RawMul(mne.io.BaseRaw):
    pass


if __name__ == '__main__':
    path = 'D:\\EEGdata\\an\'s folder\\Data\\six-problem\\EEG_six_problem\\eeg_april_02(1)\\april_2(1).vhdr'
    raw_data = mne.io.read_raw_brainvision(vhdr_fname=path)