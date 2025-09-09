import functools
from sklearn.decomposition import FastICA
from eeg_tool.model.eeg import Eeg

# class Subject:
    # _instance = None
    #
    # def __new__(cls, *args, **kw):
    #     if cls._instance is None:
    #         cls._instance = object.__new__(cls, *args, **kw)
    #     return cls._instance
    # def __int__(self, subject_info=None, eeg=None, hrv=None, gsr=None):
    #     self.subject_info = subject_info
    #     if eeg is None:
    #         eeg = Eeg()
    #         eeg.read_raw_data(subject_info.subject_data_path, subject_info.montage_path)
    #         eeg.read_trial_info(subject_info.experiment_conf, subject_info.subject_name)
    # def preprocess(self, preprocess_eeg, low_frequency=1., high_frequency=50., line_frequency=5.):
    #     preprocess_eeg.filter()
    #     pass



def singleton(cls):
    cls.__new_original__ = cls.__new__

    @functools.wraps(cls.__new__)
    def singleton_new(cls, *args, **kwargs):
        it = cls.__dict__.get('__it__')
        if it is not None:
            return it
        cls.__it__ = it = cls.__new_original__(cls, *args, **kwargs)
        it.__init_original__(*args, **kwargs)
        return it

    cls.__new__ = singleton_new
    cls.__init_original__ = cls.__init__
    cls.__init__ = object.__init__
    return cls

@singleton
class SubjectInfo(object):
    def __new__(cls, *args, **kwargs):
        cls.x = 10
        return object.__new__(cls)
    def __init__(self, subject_data_path=None, experiment_conf=None, condition_name=None, montage_path=None,
                 subject_name=None, n_run=None):
        assert self.x == 10
        self.subject_data_path = subject_data_path
        self.experiment_conf = experiment_conf
        self.condition_name = condition_name
        self.montage_path = montage_path
        self.subject_name = subject_name
        self.n_run = n_run

@singleton
class Subject(object):
    def __new__(cls, *args, **kwargs):
        cls.x = 10
        return object.__new__(cls)
    def __init__(self, subject_info=None, eeg=None, hrv=None, gsr=None):
        assert self.x == 10
        self.subject_info = subject_info
        if eeg is None:
            self.eeg = Eeg()
            self.eeg.read_raw_data(subject_info.subject_data_path, subject_info.montage_path)
            self.eeg.read_trial_info(subject_info.experiment_conf, subject_info.subject_name)
            self.eeg.split_tasks()
            self.eeg.data_segment()




if __name__ == '__main__':
    s1 = SubjectInfo()
    s2 = SubjectInfo()
