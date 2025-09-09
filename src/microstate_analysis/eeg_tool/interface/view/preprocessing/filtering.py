from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *
import numpy as np
import mne

class Filtering(QDialog):
    filter_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super(Filtering, self).__init__(parent)
        self.setWindowTitle('Filtering')

        # self.filter_types = QLabel('Filter types:', self)
        # self.iir_butterworth = QCheckBox('IIR Butterworth', self)
        # self.fir = QCheckBox('FIR', self)
        # self.parks_mcclellan_notch = QCheckBox('Parks-McClellan Notch', self)
        # self.gaussian = QCheckBox('Gaussian', self)
        # self.display = QLabel('Display:', self)
        # self.filter_frequency_response = QCheckBox('Fliter frequency response', self)
        # self.filter_impulse_response = QCheckBox('Filter impulse response', self)
        # self.unfiltered_data_frequency_response = QCheckBox('Unfiltered data frequency response', self)
        # self.preview_filtered_frequency_response = QCheckBox('Preview filtered frequency response', self)

        self.component_config = [
            {'label_name': 'High frequency:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            {'label_name': 'Low frequency:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            {'label_name': 'Line frequency:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
        ]
        self.component_obj = []
        for item in self.component_config:
            if item['type'] == 'LabelInput':
                self.component_obj.append(
                    LabelInput(label=item['label'], label_name=item['label_name'], input=item['input']))

        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)
        self.set_label_width_by_maximum()

        layout = QGridLayout()
        # layout.addWidget(self.filter_types, 0, 0)
        # layout.addWidget(self.iir_butterworth, 1, 0)
        # layout.addWidget(self.fir, 2, 0)
        # layout.addWidget(self.parks_mcclellan_notch, 3, 0)
        # layout.addWidget(self.gaussian, 4, 0)
        # layout.addWidget(self.display, 5, 0)
        # layout.addWidget(self.filter_frequency_response, 6, 0)
        # layout.addWidget(self.filter_impulse_response, 7, 0)
        # layout.addWidget(self.unfiltered_data_frequency_response, 8, 0)
        # layout.addWidget(self.preview_filtered_frequency_response, 9, 0)
        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[i], i, 0)
        layout.addWidget(self.cancel_button, 3, 1)
        layout.addWidget(self.ok_button, 3, 2)
        self.setLayout(layout)

        self.cancel_button.clicked.connect(self.clicked_cancel_button)
        self.ok_button.clicked.connect(self.clicked_ok_button)

    def clicked_cancel_button(self):
        # subject = Subject()
        # subject.preprocess()
        self.close()

    def clicked_ok_button(self):
        h_freq = float(self.component_obj[0].input.text())
        l_freq = float(self.component_obj[1].input.text())
        line_noise = float(self.component_obj[2].input.text())
        subject_info = SubjectInfo()
        subject = Subject(subject_info)

        # tasks_values = list(subject.eeg.trial_info.values())
        # np_tasks_values = np.array(tasks_values)
        # self.tasks_start = list(np_tasks_values[:, 0])
        # self.tasks_end = list(np_tasks_values[:, 1])
        # self.tasks_start = list(map(int, self.tasks_start))
        # self.tasks_end = list(map(int, self.tasks_end))
        filtered_list = []
        # for i in range(len(self.tasks_start)):
        #     task_data = list(subject.eeg.tasks_data.items())
        #     task_data_list = []
        #     task_data_list.append(task_data[i])
        #     tasks = dict(task_data_list)
            #
            # trial_data = list(subject.eeg.trial_info.items())
            # trial_data_list = []
            # trial_data_list.append(trial_data[i])
            # trial_info = dict(trial_data_list)

        #     subject.eeg.init_preprocessing(raw=subject.eeg.raw_data, tasks=tasks, trial_info=trial_info)
        #     subject.eeg.preprocessing.concatenate_tasks()
        #     subject.eeg.preprocessing.filter(low_frequency=l_freq, high_frequency=h_freq)
        #     subject.eeg.preprocessing.remove_line_noise(line_frequency=line_noise)
        #     self.filter_eeg = subject.eeg.preprocessing.tasks_cleaned
        #     filter_eeg_T = self.filter_eeg.get_data().T
        #     filter_eeg_T_tolist = filter_eeg_T.tolist()
        #     filtered_list += filter_eeg_T_tolist
        # self.filter = np.array(filtered_list)
        # self.filtered_data = mne.io.RawArray(self.filter.T, subject.eeg.raw_data.info)
        # self.auto_bad_channel = []
        tasks_cleaned_list = []
        for i in range(len(subject.eeg.new_trial)):
            subject.eeg.init_preprocessing(raw=subject.eeg.new_raw_data, tasks=subject.eeg.task_array[i],
                                           trial_info=subject.eeg.new_trial_array[i], tasks_cleaned=None)
            subject.eeg.preprocessing.concatenate_tasks()
            subject.eeg.preprocessing.filter(low_frequency=l_freq, high_frequency=h_freq)
            subject.eeg.preprocessing.remove_line_noise(line_frequency=line_noise)
            self.filter_eeg = subject.eeg.preprocessing.tasks_cleaned
            filter_eeg_T = self.filter_eeg.get_data().T
            filter_eeg_T_tolist = filter_eeg_T.tolist()
            if i > 0 and subject.eeg.new_trial[i] == subject.eeg.new_trial[i - 1]:
                filtered_list = filtered_list
                tasks_cleaned_list.append(self.filter_eeg)
            else:
                filtered_list += filter_eeg_T_tolist
                tasks_cleaned_list.append(self.filter_eeg)

            # subject.eeg.preprocessing.remove_bad_channel(thread=5, threshold=0.1)
            # bads = subject.eeg.preprocessing.global_bads
            # self.auto_bad_channel += bads
            # break
        self.filter = np.array(filtered_list)
        self.filtered_data = mne.io.RawArray(self.filter.T, subject.eeg.raw_data.info)
        # self.global_bad_channel = sorted(set(self.auto_bad_channel), key=self.auto_bad_channel.index)
        # self.select_algorithm = BadGlobalChannels(self.global_bad_channel)
        # print(self.global_bad_channel)

        subject.eeg.tasks_cleaned = tasks_cleaned_list
        self.filter_signal.emit(self.filtered_data)
        # self.filter_signal.emit(self.filtered_data)
        self.close()



    def set_label_width_by_maximum(self):
        max_width = 0
        for item in self.component_config:
            width = item['label'].fontMetrics().boundingRect(item['label'].text()).width()
            if max_width < width:
                max_width = width
        for item in self.component_config:
            item['label'].setFixedWidth(max_width)