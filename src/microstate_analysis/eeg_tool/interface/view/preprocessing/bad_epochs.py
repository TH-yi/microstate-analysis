import mne
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QLabel, QComboBox, QPushButton, QGridLayout, QDialog, \
    QLineEdit, QRadioButton, QListWidget, QListWidgetItem
from eeg_tool.interface.view.dialog_component import *
from eeg_tool.interface.model.subject import *
import numpy as np


# class BadEpochs(QWidget):
#
#     def __init__(self):
#         super().__init__()
#         # self.auto_bad_channel = auto_bad_channel
#         self.manual_dialog = ManualDialog()
#         self.auto_dialog = AutoDialog()
#         self.initUI()
#
#     def initUI(self):
#         self.manual = QRadioButton('Manual', self)
#         self.automatic = QRadioButton('Automatic', self)
#         self.cancel_button = QPushButton('Cancel')
#         self.ok_button = QPushButton('OK')
#
#         layout = QGridLayout()
#         layout.addWidget(self.manual, 1, 0)
#         layout.addWidget(self.automatic, 0, 0)
#         layout.addWidget(self.cancel_button, 2, 2)
#         layout.addWidget(self.ok_button, 2, 3)
#         self.setLayout(layout)
#
#         self.setWindowTitle('Bad Epochs')
#         self.show()
#         self.ok_button.clicked.connect(self.select_channels)
#
#     def select_channels(self):
#         if self.manual.isChecked():
#             self.manual_dialog.show()
#             self.manual_dialog.exec()
#             self.close()
#         elif self.automatic.isChecked():
#             self.auto_dialog.show()
#             self.auto_dialog.exec()
#             self.close()


class BadEpochs(QWidget):
    epochs_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # self.bad_epochs = QLabel('Mark bad epochs and bad local channels:', self)
        self.auto_label = QLabel('Automatic calculation of bad epochs:', self)
        # self.amplitude_range_label = QLabel('Amplitude range:', self)
        # self.amplitude_range_input = QLineEdit()
        # self.variance_label = QLabel('Variance:', self)
        # self.variance_input = QLineEdit()
        # self.channel_deviation_label = QLabel('Channel deviation:', self)
        # self.channel_deviation_input = QLineEdit()
        self.manual_label = QLabel('Manually add bad epochs:', self)
        # self.select_act_label = QLabel('Select Activity:', self)
        # subject = Subject()
        # self.activity = list(subject.eeg.new_trial_info)
        # self.combo = ComboCheckBox(self.activity)
        # self.input_label = QLabel('Bad Epochs (Input time):')
        # self.input_time = QLineEdit()
        self.cancel_button = QPushButton('Cancel')
        self.ok_button = QPushButton('OK')

        subject = Subject()
        task_name = list(subject.eeg.new_trial_info)
        self.manual_config = []
        for i in range(len(subject.eeg.artifacts_removal_eeg_list)):
            self.manual_config.append(
                {'label_name': task_name[i], 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'}
            )
        self.manual_obj = []
        for item in self.manual_config:
            if item['type'] == 'LabelInput':
                self.manual_obj.append(
                    LabelInput(label=item['label'], label_name=item['label_name'], input=item['input']))

        self.component_config = [
            {'label_name': 'Amplitude range:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            {'label_name': 'Variance:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            {'label_name': 'Channel deviation:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            # {'label_name': 'Bad local channels:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            # {'label_name': 'Criteria:', 'label': QLabel(), 'combobox': QComboBox(), 'type': 'LabelComboBox',
            #  'combobox_items': ['Variance', 'Median gradient', 'Amplitude range', 'Channel deviation']}
        ]
        self.component_obj = []
        for item in self.component_config:
            if item['type'] == 'LabelInput':
                self.component_obj.append(
                    LabelInput(label=item['label'], label_name=item['label_name'], input=item['input']))

        self.set_label_width_by_maximum()


        layout = QGridLayout()
        layout.addWidget(self.auto_label, 0, 0)
        # layout.addWidget(self.amplitude_range_label, 0, 1)
        # layout.addWidget(self.amplitude_range_input, 0, 2)
        # layout.addWidget(self.variance_label, 1, 1)
        # layout.addWidget(self.variance_input, 1, 2)
        # layout.addWidget(self.channel_deviation_label, 2, 1)
        # layout.addWidget(self.channel_deviation_input, 2, 2)


        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[i], i, 1)
        for i in range(len(self.manual_obj)):
            layout.addWidget(self.manual_obj[i], i+3, 1)
        layout.addWidget(self.manual_label, 3, 0)
        # layout.addWidget(self.select_act_label, 4, 1)
        # layout.addWidget(self.combo, 4, 2)
        # layout.addWidget(self.input_label, 5, 1)
        # layout.addWidget(self.input_time, 5, 2)
        layout.addWidget(self.cancel_button, 7, 4)
        layout.addWidget(self.ok_button, 7, 5)
        self.setLayout(layout)
        self.setWindowTitle('Bad epochs')

        self.ok_button.clicked.connect(self.clicked_ok)
        self.show()

    def clicked_ok(self):
        amplitude_range = float(self.component_obj[0].input.text())
        variance = float(self.component_obj[1].input.text())
        channel_deviation = float(self.component_obj[2].input.text())

        # add_task1 = int(self.manual_obj[0].input.text())
        # add_task2 = int(self.manual_obj[1].input.text())
        # add_task3 = int(self.manual_obj[2].input.text())

        subject = Subject()
        task_name = list(subject.eeg.new_trial_info)
        self.task_start = []
        self.task_end = []
        self.mark_epochs = []
        for i in range(len(subject.eeg.new_trial_info)):
            value_start = subject.eeg.new_trial_info.get(task_name[i])[0]
            self.task_start.append(value_start)
            value_end = subject.eeg.new_trial_info.get(task_name[i])[1]
            self.task_end.append(value_end)

        for i in range(len(subject.eeg.artifacts_removal_eeg_list)):
            # self.start = subject.eeg.preprocessing.start_point
            # self.end = subject.eeg.preprocessing.end_point
            # self.epochs_number = subject.eeg.preprocessing.epochs_point
            self.start = 0
            self.end = int(self.task_end[i]) - int(self.task_start[i])
            sfreq = int(2 * subject.eeg.raw_data.info['sfreq'])
            self.sum_epochs = list(range(self.start, self.end, sfreq))
            self.mark_epochs.append(self.sum_epochs)
            print(self.sum_epochs)
        print(self.mark_epochs)
        subject.eeg.mark_epochs = self.mark_epochs
        self.bad_channel_epochs_list = []
        for i in range(len(subject.eeg.artifacts_removal_eeg_list)):
            subject.eeg.preprocessing.mark_bad_epochs(drop_epoch=0.25, n_times=2,
                                                      threshold=[variance, channel_deviation,
                                                                 amplitude_range, 3.29053],
                                                      filtered_data=subject.eeg.artifacts_removal_eeg_list[i],
                                                      task_start=self.task_start[i], task_end=self.task_end[i],
                                                      trials_name=task_name[i],
                                                      global_good_index=subject.eeg.global_good_index_list[i],
                                                      global_bads=subject.eeg.bad_array[i])
            self.bad_channel_epochs = subject.eeg.preprocessing.bad_channel_epochs.T
            self.itemindex = np.argwhere(self.bad_channel_epochs == 1)
            self.itemindex_list = list(self.itemindex.T[1])

            self.timeindex = [i * 2 for i in self.itemindex_list]
            self.timeindex_list = list(set(self.timeindex))
            print(self.timeindex_list)
            # if selected_task == task_name[i]:
            #     self.timeindex_list.append(selected_time)
            add_time_str = self.manual_obj[i].input.text()
            add_time = add_time_str.split(', ')
            add_time_list = []
            for j in range(len(add_time)):
                value = int(add_time[j])
                add_time_list.append(value)
            self.timeindex_list += add_time_list
            self.bad_channel_epochs_list.append(self.timeindex_list)


        self.bad_epochs_eeg_list = []
        for i in range(len(self.bad_channel_epochs_list)):
            bad_annot = mne.Annotations(onset=self.bad_channel_epochs_list[i], duration=2, description=['Bad Epoch'])
            bads = subject.eeg.artifacts_removal_eeg_list[i].set_annotations(bad_annot)
            self.bad_epochs_eeg_list.append(bads)

        subject.eeg.bad_epochs_list = self.bad_epochs_eeg_list
        self.close()
        # subject.eeg.preprocessing.mark_bad_epochs(drop_epoch=0.25, n_times=2, threshold=[variance, channel_deviation,
        #                                                                                  amplitude_range, 3.29053])
        # self.start = subject.eeg.preprocessing.start_point
        # self.end = subject.eeg.preprocessing.end_point
        # self.epochs_number = subject.eeg.preprocessing.epochs_point
        # self.sum_epochs = []
        # for i in range(len(self.start)):
        #     start = int(self.start[i])
        #     end = int(self.end[i])
        #     sfreq = int(2 * subject.eeg.preprocessing.tasks_cleaned.info['sfreq'])
        #     i_epochs = list(range(start, end, sfreq))
        #     self.sum_epochs += i_epochs
        # self.epochs_signal.emit(self.sum_epochs)
        # self.close()




    def set_label_width_by_maximum(self):
        max_width = 0
        for item in self.component_config:
            width = item['label'].fontMetrics().boundingRect(item['label'].text()).width()
            if max_width < width:
                max_width = width
        for item in self.component_config:
            item['label'].setFixedWidth(max_width)

#
# class ManualDialog(QWidget):
#
#     def __init__(self):
#         super().__init__()
#
#         self.initUI()
#
#     def initUI(self):
#         self.setWindowTitle('Bad Epochs')
#         self.select_act_label = QLabel('Select Activity:', self)
#         subject = Subject()
#         self.activity = list(subject.eeg.new_trial_info)
#         self.combo = ComboCheckBox(self.activity)
#         self.input_label = QLabel('Bad Epochs (Input time):')
#         self.input_time = QLineEdit()
#         self.cancel_button = QPushButton('Cancel')
#         self.ok_button = QPushButton('OK')
#
#         layout = QGridLayout()
#         layout.addWidget(self.select_act_label, 0, 0)
#         layout.addWidget(self.combo, 0, 1)
#         layout.addWidget(self.input_label, 1, 0)
#         layout.addWidget(self.input, 1, 1)
#         layout.addWidget(self.cancel_button, 2, 2)
#         layout.addWidget(self.ok_button, 2, 3)
#         self.setLayout(layout)
#
#         self.ok_button.clicked.connect(self.clicked_ok_button)
#
#     def clicked_ok_button(self):
#         selected_task = str(self.combo.currentText()[0])
#         selected_time = int(self.input_time.text())
#         print(selected_task)
#         print(type(selected_task))
#         subject = Subject()
#         if subject.eeg.bad_epochs_list is None:
#             for i in range(len(subject.eeg.artifacts_removal_eeg_list)):
#                 if selected_task == self.activity[i]:
#
#
#
#         self.close()

class ComboCheckBox(QComboBox):
    def __init__(self, items: list):
        super(ComboCheckBox, self).__init__()
        self.items = items
        self.box_list = []
        self.text = QLineEdit()
        # self.state = 0

        q = QListWidget()
        for i in range(len(self.items)):
            self.box_list.append(QCheckBox())
            self.box_list[i].setText(self.items[i])
            item = QListWidgetItem(q)
            q.setItemWidget(item, self.box_list[i])
            self.box_list[i].stateChanged.connect(self.show_selected)

        q.setStyleSheet("font-size: 20px; font-weight: bold; height: 40px; margin-left: 5px")
        self.setStyleSheet("width: 300px; height: 50px; font-size: 21px; font-weight: bold")
        self.text.setReadOnly(True)
        self.setLineEdit(self.text)
        self.setModel(q.model())
        self.setView(q)

    def get_selected(self) -> list:
        ret = []
        for i in range(0, len(self.items)):
            if self.box_list[i].isChecked():
                ret.append(self.box_list[i].text())
        return ret

    def show_selected(self):
        self.text.clear()
        ret = '; '.join(self.get_selected())
        self.text.setText(ret)

