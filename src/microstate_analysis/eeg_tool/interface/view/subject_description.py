import sys

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *
from eeg_tool.interface.view.plot_raw import MainWindowWidget
from eeg_tool.interface.model.subject import *



class SubjectDescription(QDialog):
    signal_draw_eeg_parameter = pyqtSignal(object)

    def __init__(self, parent=None):
        super(SubjectDescription, self).__init__(parent)
        self.setWindowTitle('Subject description')
        self.parent = parent
        self.component_config = [
            {'label_name': 'Subject data:', 'label': QLabel(), 'input': QLineEdit(), 'button': QPushButton(),
             'button_name': 'Open file', 'type': 'OpenFileLabelInput'},
            {'label_name': 'Experiment configure:', 'label': QLabel(), 'input': QLineEdit(), 'button': QPushButton(),
             'button_name': 'Open file', 'type': 'OpenFileLabelInput'},
            {'label_name': 'Condition name:', 'label': QLabel(), 'input': QLineEdit(), 'button': QPushButton(),
             'button_name': 'Open file', 'type': 'OpenFileLabelInput'},
            {'label_name': 'Montage path:', 'label': QLabel(), 'input': QLineEdit(), 'button': QPushButton(),
             'button_name': 'Open file', 'type': 'OpenFileLabelInput'},
            {'label_name': 'Subject name:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            {'label_name': 'The number of run:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'}
        ]

        self.component_obj = []
        for item in self.component_config:
            if item['type'] == 'OpenFileLabelInput':
                self.component_obj.append(OpenFileLabelInput(label=item['label'], label_name=item['label_name'],
                                                             input=item['input'], button=item['button'],
                                                             button_name=item['button_name']))
            elif item['type'] == 'LabelInput':
                self.component_obj.append(
                    LabelInput(label=item['label'], label_name=item['label_name'], input=item['input']))

        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)
        self.set_label_width_by_maximum()
        layout = QGridLayout()

        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[i], i, 0)
        layout.addWidget(self.cancel_button, 6, 1)
        layout.addWidget(self.ok_button, 6, 2)
        self.setLayout(layout)

        self.ok_button.clicked.connect(self.click_ok_button)

        self.plot_raw = MainWindowWidget(self.parent)

    def click_ok_button(self):
        # self.component_obj[0].input_path
        # print(self.component_obj[0].file_name)
        # file_name0 = self.component_obj[0].file_name
        # file_name1 = self.component_obj[1].file_name
        # file_name2 = self.component_obj[2].file_name
        # file_name3 = self.component_obj[3].file_name

        # file_name, file_type = QFileDialog.getOpenFileName(self.component_obj[0].file_name, "Open file", "./", "*(*.*)")
        # self.component_obj[0].input.setText(file_name)
        # self.component_obj[1].input.setText(file_name)
        # self.component_obj[2].input.setText(file_name)
        # self.component_obj[3].input.setText(file_name)
        subject_info = SubjectInfo(subject_data_path=self.component_obj[0].file_name,
                                   experiment_conf=self.component_obj[1].file_name,
                                   # condition_name=self.component_obj[2].input.setText(file_name2),
                                   montage_path=self.component_obj[3].file_name,
                                   subject_name=self.component_obj[4].input.text(),
                                   n_run=self.component_obj[5].input.text())
        self.subject = Subject(subject_info)
        self.signal_draw_eeg_parameter.emit(self.subject.eeg.new_raw_data)
        self.parent.setFocus()
        self.close()

    def set_label_width_by_maximum(self):
        max_width = 0
        for item in self.component_config:
            width = item['label'].fontMetrics().boundingRect(item['label'].text()).width()
            if max_width < width:
                max_width = width
        for item in self.component_config:
            item['label'].setFixedWidth(max_width)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = SubjectDescription()
#     win.show()
#     sys.exit(app.exec_())
