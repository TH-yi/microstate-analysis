from PyQt5.QtWidgets import *
from eeg_tool.interface.view.dialog_component import *

class Plot(QDialog):

    def __init__(self, parent=None):
        super(Plot, self).__init__(parent)
        self.setWindowTitle('Plot')

        self.component_config = [
            {'label_name': 'Plot classes:', 'label': QLabel(), 'input': QLineEdit(), 'button': QPushButton(),
             'button_name': 'Open file', 'type': 'OpenFileLabelInput'},
            {'label_name': 'Plot labels:', 'label': QLabel(), 'input': QLineEdit(), 'button': QPushButton(),
             'button_name': 'Open file', 'type': 'OpenFileLabelInput'}
            ]
        self.component_obj = []
        for item in self.component_config:
            if item['type'] == 'OpenFileLabelInput':
                self.component_obj.append(OpenFileLabelInput(label=item['label'], label_name=item['label_name'],
                                                             input=item['input'], button=item['button'],
                                                             button_name=item['button_name']))
        self.cancel_button = QPushButton('Cancel', self)
        self.ok_button = QPushButton('OK', self)
        self.set_label_width_by_maximum()

        layout = QGridLayout()
        for i in range(len(self.component_obj)):
            layout.addWidget(self.component_obj[i], i, 0)
        layout.addWidget(self.cancel_button, 3, 1)
        layout.addWidget(self.ok_button, 3, 2)
        self.setLayout(layout)

    def set_label_width_by_maximum(self):
        max_width = 0
        for item in self.component_config:
            width = item['label'].fontMetrics().boundingRect(item['label'].text()).width()
            if max_width < width:
                max_width = width
        for item in self.component_config:
            item['label'].setFixedWidth(max_width)