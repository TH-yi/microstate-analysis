from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *
from eeg_tool.interface.model.subject import *



class ManualSegmentation(QDialog):
    signal_draw_eeg_parameter = pyqtSignal(object)

    def __init__(self, parent=None):
        super(ManualSegmentation, self).__init__(parent)
        self.setWindowTitle('Manual Segmentation')
        self.parent = parent
        self.component_config = [
            {'label_name': 'Task name:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            {'label_name': 'Start:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            {'label_name': 'End:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'}
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

        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[i], i, 0)
        layout.addWidget(self.cancel_button, 4, 1)
        layout.addWidget(self.ok_button, 4, 2)
        self.setLayout(layout)

        self.ok_button.clicked.connect(self.click_ok_button)

    def click_ok_button(self):
        new_task_name = self.component_obj[0].input.text()
        start = int(self.component_obj[1].input.text())
        end = int(self.component_obj[2].input.text())
        time_list = []
        time_list.append(start)
        time_list.append(end)
        new_task = []
        new_task.append(new_task_name)
        new_task.append(time_list)
        new_task_dict = {}
        new_task_dict[new_task_name] = time_list

        subject = Subject()
        subject.eeg.new_trial.insert(1, new_task)
        subject.eeg.new_trial_info = dict(subject.eeg.new_trial)
        subject.eeg.task_segmentation()

        task_data_list = list(subject.eeg.new_tasks_data.values())
        task_dict = {}
        task_dict[new_task_name] = task_data_list[1]
        subject.eeg.task_array.insert(1, task_dict)

        subject.eeg.new_trial_array.insert(1, new_task_dict)
        self.signal_draw_eeg_parameter.emit(subject.eeg.new_raw_data)
        self.close()




    def set_label_width_by_maximum(self):
        max_width = 0
        for item in self.component_config:
            width = item['label'].fontMetrics().boundingRect(item['label'].text()).width()
            if max_width < width:
                max_width = width
        for item in self.component_config:
            item['label'].setFixedWidth(max_width)