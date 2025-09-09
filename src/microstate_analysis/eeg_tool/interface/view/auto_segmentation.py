from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *
from eeg_tool.interface.model.subject import *



class AutoSegmentation(QDialog):
    signal_draw_eeg_parameter = pyqtSignal(object)

    def __init__(self, parent=None):
        super(AutoSegmentation, self).__init__(parent)
        self.setWindowTitle('Auto Segmentation')
        self.parent = parent

        self.auto_segmentation = QRadioButton('Auto Segmentation', self)
        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)

        layout = QGridLayout()

        layout.addWidget(self.auto_segmentation, 0, 0)
        layout.addWidget(self.cancel_button, 4, 1)
        layout.addWidget(self.ok_button, 4, 2)
        self.setLayout(layout)

        self.ok_button.clicked.connect(self.click_ok_button)

    def click_ok_button(self):
        if self.auto_segmentation.isChecked():
            new_activity_list = ['new_activity', [200000, 220000]]
            new_activity_dict = {'new_activity': [200000, 220000]}
            new_task_name = list(new_activity_dict)[0]

        # new_task_name = self.component_obj[0].input.text()
        # start = int(self.component_obj[1].input.text())
        # end = int(self.component_obj[2].input.text())
        # time_list = []
        # time_list.append(start)
        # time_list.append(end)
        # new_task = []
        # new_task.append(new_task_name)
        # new_task.append(time_list)
        # new_task_dict = {}
        # new_task_dict[new_task_name] = time_list

            subject = Subject()
            subject.eeg.new_trial.insert(1, new_activity_list)
            subject.eeg.new_trial_info = dict(subject.eeg.new_trial)
            subject.eeg.task_segmentation()

            task_data_list = list(subject.eeg.new_tasks_data.values())
            task_dict = {}
            task_dict[new_task_name] = task_data_list[1]
            subject.eeg.task_array.insert(1, task_dict)

            subject.eeg.new_trial_array.insert(1, new_activity_dict)
            self.signal_draw_eeg_parameter.emit(subject.eeg.new_raw_data)
            self.close()