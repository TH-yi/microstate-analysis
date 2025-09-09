from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *
import sys

class SmoothsMicrostatesLabels(QDialog):

    def __init__(self, parent=None):
        super(SmoothsMicrostatesLabels, self).__init__(parent)
        self.setWindowTitle('Smooths microstates labels')

        self.component_config = [
            {'label_name': 'Smooth label obtained from:', 'label': QLabel(), 'combobox': QComboBox(), 'type': 'LabelComboBox',
             'combobox_items': ['Backfitting prototypes to EEG','']},
            {'label_name': 'Choose smoothing method:', 'label': QLabel(), 'combobox': QComboBox(), 'type': 'LabelComboBox',
             'combobox_items': ['Reject small segments','']},
            {'label_name': 'Redistribute segments smaller than(in ms):', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'}
        ]
        self.component_obj = []
        for item in self.component_config:
            if item['type'] == 'LabelInput':
                self.component_obj.append(
                    LabelInput(label=item['label'], label_name=item['label_name'], input=item['input']))
            elif item['type'] == 'LabelComboBox':
                self.component_obj.append(
                    LabelComboBox(label=item['label'], label_name=item['label_name'], combobox=item['combobox'],
                                  combobox_items=item['combobox_items']))
        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)
        self.set_label_width_by_maximum()

        layout = QGridLayout()
        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[i], i, 0)
        layout.addWidget(self.cancel_button, 4, 1)
        layout.addWidget(self.ok_button, 4, 2)
        self.setLayout(layout)

    def set_label_width_by_maximum(self):
        max_width = 0
        for item in self.component_config:
            width = item['label'].fontMetrics().boundingRect(item['label'].text()).width()
            if max_width < width:
                max_width = width
        for item in self.component_config:
            item['label'].setFixedWidth(max_width)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SmoothsMicrostatesLabels()
    win.show()
    sys.exit(app.exec_())