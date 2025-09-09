import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from eeg_tool.interface.model.eeg import Eeg

class FilterData(QDialog):
    filter_signal = pyqtSignal(float, float)
    def __init__(self, parent=None):
        super(FilterData, self).__init__(parent)
        self.setWindowTitle('Filter the data')
        self.resize(300,100)
        self.eeg=Eeg()

        label1 = QLabel("Enter low frequency:")
        label2 = QLabel("Enter high frequency:")

        self.lineEdit1 = QLineEdit()
        self.lineEdit2 = QLineEdit()
        self.lineEdit1.resize(20,10)
        self.lineEdit2.resize(20,10)

        self.btnPress1 = QPushButton('Cancel')
        self.btnPress2 = QPushButton('OK')

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(label2, 1, 0)
        layout.addWidget(self.lineEdit1, 0, 1)
        layout.addWidget(self.lineEdit2, 1, 1)
        layout.addWidget(self.btnPress1, 2, 1)
        layout.addWidget(self.btnPress2, 2, 2)
        self.setLayout(layout)

        self.btnPress1.clicked.connect(self.btnPress1_clicked)
        self.btnPress2.clicked.connect(self.btnPress2_clicked)

    def btnPress1_clicked(self):
        self.close()

    def btnPress2_clicked(self):
        l_freq = float(self.lineEdit1.text())
        h_freq = float(self.lineEdit2.text())

        self.filter_signal.emit(l_freq,h_freq)
        # self.eeg.raw_eeg.raw_data.filter(l_freq, h_freq)
        # self.eeg.raw_eeg.plot()
        self.close()

        # number_int2 = self.lineEdit2.setText()
        # self.Signal_OneParameter.connect(slot)