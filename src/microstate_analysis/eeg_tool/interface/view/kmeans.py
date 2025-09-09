from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *

class Kmeans(QDialog):
    def __init__(self, parent=None):
        super(Kmeans, self).__init__(parent)
        self.setWindowTitle('Modified K-means')

        self.input_parameters = QLabel("Input parameters specific for 'Modified K-means'.", self)
        self.component_config = [
            {'label_name': 'No. of random initializations:', 'label': QLabel(), 'input': QLineEdit(),
             'type': 'LabelInput'},
            {'label_name': 'Max no. of iterations:', 'label': QLabel(), 'input': QLineEdit(),
             'type': 'LabelInput'},
            {'label_name': 'Relative threshold for convergence:', 'label': QLabel(), 'input': QLineEdit(),
             'type': 'LabelInput'}
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
        layout.addWidget(self.input_parameters, 0, 0)
        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[i], i+1, 0)
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