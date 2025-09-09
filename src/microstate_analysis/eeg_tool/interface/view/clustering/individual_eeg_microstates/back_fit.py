import sys
from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *

class BackFit(QDialog):

    def __init__(self, parent=None):
        super(BackFit, self).__init__(parent)
        self.setWindowTitle('Back-fit')

        self.component_config = [
            {'label_name': 'Microstates classes:', 'label': QLabel(), 'input': QLineEdit(), 'button': QPushButton(),
             'button_name': 'Open file', 'type': 'OpenFileLabelInput'},
            {'label_name': 'Instantaneous raw EEG:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            {'label_name': 'Label criteria:', 'label': QLabel(), 'combobox': QComboBox(), 'type': 'LabelComboBox',
             'combobox_items': ['GFP','All EEG data']}
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
    win = BackFit()
    win.show()
    sys.exit(app.exec_())