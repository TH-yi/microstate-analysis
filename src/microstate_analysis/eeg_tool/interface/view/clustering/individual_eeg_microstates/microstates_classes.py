from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *
from eeg_tool.interface.view.kmeans import Kmeans

class MicrostatesClasses(QDialog):

    def __init__(self, parent=None):
        super(MicrostatesClasses, self).__init__(parent)
        self.setWindowTitle('Microstates classes')

        self.data_type = QLabel('Data type:  Global field power', self)
        self.component_config = [
            {'label_name': 'Algorithm:', 'label': QLabel(), 'combobox': QComboBox(), 'type': 'LabelComboBox',
             'combobox_items': ['Modified K-means', '']},
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
        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)
        self.set_label_width_by_maximum()

        layout = QGridLayout()
        layout.addWidget(self.data_type, 0, 0)
        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[i], i+1, 0)
        layout.addWidget(self.cancel_button, 4, 1)
        layout.addWidget(self.ok_button, 4, 2)
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