from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QLabel, QComboBox, QPushButton, QGridLayout, QDialog, QLineEdit


class GlobalFieldPower(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.data_type = QLabel('Data type', self)
        self.settings = QLabel('Settings for the extraction of GFP peak maps.', self)
        self.min_peak_distance = QLabel('Minimum peak distance(ms):', self)
        self.peak_rejection_threshold = QLabel('Peak rejection threshold', self)

        self.data_type_combobox = QComboBox()
        self.data_type_combobox.addItems(['Global field power', 'Raw EEG data'])
        self.data_type_combobox.setCurrentIndex(-1)

        self.min_peak_distance_lineedit = QLineEdit()
        self.peak_rejection_threshold_lineedit = QLineEdit()

        self.cancel_button = QPushButton('Cancel')
        self.ok_button = QPushButton('OK')

        layout = QGridLayout()
        layout.addWidget(self.data_type, 0, 0)
        layout.addWidget(self.data_type_combobox, 0, 1)
        layout.addWidget(self.settings, 2, 0)
        layout.addWidget(self.min_peak_distance, 3, 0)
        layout.addWidget(self.min_peak_distance_lineedit, 3, 1)
        layout.addWidget(self.peak_rejection_threshold, 4, 0)
        layout.addWidget(self.peak_rejection_threshold_lineedit, 4, 1)
        layout.addWidget(self.cancel_button, 5, 3)
        layout.addWidget(self.ok_button, 5, 4)
        self.setLayout(layout)

        self.setWindowTitle('Global field power')
        self.show()