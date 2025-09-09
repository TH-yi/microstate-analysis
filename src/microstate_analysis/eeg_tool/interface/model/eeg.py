from eeg_tool.model.raw_data import RawData


class Eeg:
    def __init__(self):
        self.raw_eeg = RawData()

    def import_eeg(self, data_fname, fmontage):
        self.raw_eeg.read_raw_data(fname=data_fname, montage=fmontage, preload=True)

    # def import_montage(self, montage_fname):
    #     self.raw_eeg.read_montage(montage_fname)

    def import_trial_info(self, fname, subject_name=None):
        self.raw_eeg.read_trial_info(fname=fname, sheet_name=subject_name)

    def import_bad_eeg(self):
        pass




