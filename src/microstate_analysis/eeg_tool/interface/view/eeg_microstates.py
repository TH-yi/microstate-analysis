import sys

from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *

class EEGMicrostates(QDialog):

    def __init__(self, parent=None):
        super(EEGMicrostates, self).__init__(parent)
        self.setWindowTitle('EEG Microstates')

        self.temporal_para = QLabel('Temporal parameters:', self)
        self.component_config = [
            {'label_name': 'Label path:', 'label': QLabel(), 'input': QLineEdit(), 'button': QPushButton(),
             'button_name': 'Open file', 'type': 'OpenFileLabelInput'},
            {'label_name': 'Special parameter:', 'label': QLabel(), 'checkbox_name': 'Topograph', 'checkbox': QCheckBox(),
             'type': 'LabelCheckbox'},
            {'checkbox_name1': 'Duration', 'checkbox1': QCheckBox(), 'checkbox_name2': 'Occurrence',
             'checkbox2': QCheckBox(), 'checkbox_name3': 'Coverage', 'checkbox3': QCheckBox(), 'type': 'Checkbox'}
        ]
        self.component_obj = []
        for item in self.component_config:
            if item['type'] == 'OpenFileLabelInput':
                self.component_obj.append(OpenFileLabelInput(label=item['label'], label_name=item['label_name'],
                                                             input=item['input'], button=item['button'],
                                                             button_name=item['button_name']))
            elif item['type'] == 'LabelCheckbox':
                self.component_obj.append(LabelCheckbox(label=item['label'], label_name=item['label_name'],
                                                        checkbox=item['checkbox'], checkbox_name=item['checkbox_name']))
            elif item['type'] == 'Checkbox':
                self.component_obj.append(Checkbox(checkbox1=item['checkbox1'], checkbox_name1=item['checkbox_name1'],
                                                   checkbox2=item['checkbox2'], checkbox_name2=item['checkbox_name2'],
                                                   checkbox3=item['checkbox3'], checkbox_name3=item['checkbox_name3']))
        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)

        layout = QGridLayout()
        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[0], 0, 0)
            layout.addWidget(self.component_obj[1], 1, 0)
            layout.addWidget(self.component_obj[2], 3, 0)
        layout.addWidget(self.temporal_para, 2, 0)
        layout.addWidget(self.cancel_button, 4, 1)
        layout.addWidget(self.ok_button, 4, 2)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = EEGMicrostates()
    win.show()
    sys.exit(app.exec_())