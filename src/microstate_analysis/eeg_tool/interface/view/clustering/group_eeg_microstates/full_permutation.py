import sys

from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *

class FullPermutation(QDialog):

    def __init__(self, parent=None):
        super(FullPermutation, self).__init__(parent)
        self.setWindowTitle('Group EEG Microstates')

        self.data_type = QLabel('Data type:  Global field power', self)
        self.algorithm = QLabel('Algorithm:  Full permutation', self)
        self.input_para = QLabel("Input parameters specific for 'Full permutation'.", self)
        self.check_box = [
            {'checkbox_name1': 'Across runs', 'checkbox1': QCheckBox(), 'checkbox_name2': 'Across conditions', 'checkbox2': QCheckBox(),
             'checkbox_name3': 'Across participants', 'checkbox3': QCheckBox(), 'type': 'Checkbox'}
        ]
        self.label_edit = [
            {'label_name': 'Number of microstates:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            {'label_name': 'Statements:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'}
        ]
        self.check_box_obj = []
        for item in self.check_box:
            if item['type'] == 'Checkbox':
                self.check_box_obj.append(
                    Checkbox(checkbox1=item['checkbox1'], checkbox_name1=item['checkbox_name1'],
                             checkbox2=item['checkbox2'], checkbox_name2=item['checkbox_name2'],
                             checkbox3=item['checkbox3'], checkbox_name3=item['checkbox_name3'])
                )
        self.label_edit_obj = []
        for item in self.label_edit:
            if item['type'] == 'LabelInput':
                self.label_edit_obj.append(
                    LabelInput(label=item['label'], label_name=item['label_name'], input=item['input']))
        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)
        self.set_label_width_by_maximum()

        layout = QGridLayout()
        for i in range(len(self.check_box_obj)):
            layout.addWidget(self.check_box_obj[i], 0, 0)
        layout.addWidget(self.data_type, 1, 0)
        layout.addWidget(self.algorithm, 2, 0)
        layout.addWidget(self.input_para, 4, 0)
        for i in range(len(self.label_edit_obj)):
            layout.addWidget(self.label_edit_obj[0], 3, 0)
            layout.addWidget(self.label_edit_obj[1], 5, 0)
        layout.addWidget(self.cancel_button, 6, 1)
        layout.addWidget(self.ok_button, 6, 2)
        self.setLayout(layout)

    def set_label_width_by_maximum(self):
        max_width = 0
        for item in self.label_edit:
            width = item['label'].fontMetrics().boundingRect(item['label'].text()).width()
            if max_width < width:
                max_width = width
        for item in self.label_edit:
            item['label'].setFixedWidth(max_width)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FullPermutation()
    win.show()
    sys.exit(app.exec_())