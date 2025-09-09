import sys
from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *
from eeg_tool.interface.view.kmeans import Kmeans

class GroupEEGMicrostates(QDialog):

    def __init__(self, parent=None):
        super(GroupEEGMicrostates, self).__init__(parent)
        self.setWindowTitle('Group EEG Microstates')

        self.data_type = QLabel('Data type:  Global field power', self)
        self.algorithm = QLabel('Algorithm:  Modified K-means', self)
        self.check_box = [
            {'checkbox_name1': 'Across runs', 'checkbox1': QCheckBox(), 'checkbox_name2': 'Across conditions', 'checkbox2': QCheckBox(),
             'checkbox_name3': 'Across participants', 'checkbox3': QCheckBox(), 'type': 'Checkbox'}
        ]
        self.component_config = [
            {'label_name': 'Number of microstates:', 'label': QLabel(), 'input': QLineEdit(), 'type': 'LabelInput'},
            {'label_name': 'Criteria for the number of microstates:', 'label': QLabel(), 'combobox': QComboBox(), 'type': 'LabelComboBox',
             'combobox_items': ['Global explained variance', 'Cross-validation', 'Dispersion', 'Krzanowski-Lai criterion','Normalised Krzanowski-Lai criterion']}
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
        self.check_box_obj = []
        for item in self.check_box:
            if item['type'] == 'Checkbox':
                self.check_box_obj.append(
                    Checkbox(checkbox1=item['checkbox1'], checkbox_name1=item['checkbox_name1'],
                             checkbox2=item['checkbox2'], checkbox_name2=item['checkbox_name2'],
                             checkbox3=item['checkbox3'], checkbox_name3=item['checkbox_name3'])
                )
        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)
        self.set_label_width_by_maximum()

        layout = QGridLayout()
        for i in range(len(self.check_box_obj)):
            layout.addWidget(self.check_box_obj[i], 0, 0)
        layout.addWidget(self.data_type, 1, 0)
        layout.addWidget(self.algorithm, 2, 0)
        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[i], i+3, 0)
        layout.addWidget(self.cancel_button, 5, 1)
        layout.addWidget(self.ok_button, 5, 2)
        self.setLayout(layout)

        self.ok_button.clicked.connect(self.kmeans)

    def set_label_width_by_maximum(self):
        max_width = 0
        for item in self.component_config:
            width = item['label'].fontMetrics().boundingRect(item['label'].text()).width()
            if max_width < width:
                max_width = width
        for item in self.component_config:
            item['label'].setFixedWidth(max_width)


    def kmeans(self):
        self.kmeans = Kmeans()
        self.kmeans.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GroupEEGMicrostates()
    win.show()
    sys.exit(app.exec_())